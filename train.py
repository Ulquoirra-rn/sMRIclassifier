import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import MRISliceDataset, N_TABULAR, load_metadata
from model import HybridMRIClassifier


SCAN_FOLDER_TO_LABEL = {
    "smri_t1w": "t1",
    "smri_t2": "t2",
    "smri_t1ce": "t1ce",
    "smri_flair": "flair",
}


def load_samples(data_dir):
    """Load samples from a dataset root where each top-level entry is a patient.

    The scan-type folders (smri_t1w, smri_t2, smri_t1ce, smri_flair) can sit
    at any depth inside the patient directory — intermediate sub-directories
    (e.g. session IDs, visit labels, extra nesting) are handled transparently
    via os.walk.

    Example layouts (all supported):
        data_dir/patient_id/input/nifti/smri_t1w/scan.nii.gz
        data_dir/patient_id/ses-01/input/nifti/smri_t1w/scan.nii.gz
        data_dir/patient_id/visit_2023/extra/smri_t1w/scan.nii.gz

    Not all patients need all scan types.
    """
    samples = []
    for patient_id in sorted(os.listdir(data_dir)):
        patient_dir = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        # Walk the entire patient subtree once; collect all scan-type dirs found
        found: dict[str, dict] = {}  # scan_folder -> {nifti, json}
        for dirpath, dirnames, filenames in os.walk(patient_dir):
            folder_name = os.path.basename(dirpath)
            if folder_name not in SCAN_FOLDER_TO_LABEL:
                continue
            # Already found a hit for this scan type — keep the first one
            if folder_name in found:
                continue
            nifti_path = None
            json_path = None
            for fname in filenames:
                if fname.endswith((".nii", ".nii.gz")) and nifti_path is None:
                    nifti_path = os.path.join(dirpath, fname)
                elif fname.endswith(".json") and json_path is None:
                    json_path = os.path.join(dirpath, fname)
            if nifti_path is not None:
                found[folder_name] = {"nifti": nifti_path, "json": json_path}

        for scan_folder, paths in found.items():
            samples.append({
                "nifti": paths["nifti"],
                "json": paths["json"],
                "label": SCAN_FOLDER_TO_LABEL[scan_folder],
                "patient_id": patient_id,
            })

    return samples


def load_samples_from_scan_dirs(scan_dirs):
    """Load samples when a separate directory is provided per scan type.

    scan_dirs: dict mapping label string (e.g. 't1') to an input directory path.

    Two layouts are supported automatically:

    Flat layout — NIfTI files sit directly inside the scan dir.  The filename
    stem (before .nii/.nii.gz) is used as the patient_id.  An optional JSON
    sidecar with the same stem is loaded if present.
        scan_dir/
          patient001.nii.gz
          patient001.json   (optional)
          patient002.nii.gz

    Nested layout — Each patient has its own sub-directory.  The sub-directory
    name is used as the patient_id.  The first NIfTI file found inside is
    loaded, along with an optional JSON sidecar.
        scan_dir/
          patient001/
            scan.nii.gz
            scan.json        (optional)
          patient002/
            scan.nii.gz
    """
    samples = []
    for label, scan_dir in scan_dirs.items():
        if not os.path.isdir(scan_dir):
            print(f"Warning: {scan_dir} is not a directory, skipping label '{label}'.")
            continue

        entries = os.listdir(scan_dir)
        nifti_files = [e for e in entries if e.endswith((".nii", ".nii.gz"))]

        if nifti_files:
            # Flat layout: NIfTI files directly in scan_dir
            for fname in nifti_files:
                nifti_path = os.path.join(scan_dir, fname)
                patient_id = fname.replace(".nii.gz", "").replace(".nii", "")
                json_path = None
                candidate = os.path.join(scan_dir, patient_id + ".json")
                if os.path.exists(candidate):
                    json_path = candidate
                samples.append({
                    "nifti": nifti_path,
                    "json": json_path,
                    "label": label,
                    "patient_id": patient_id,
                })
        else:
            # Nested layout: one sub-directory per patient
            for patient_id in sorted(entries):
                patient_dir = os.path.join(scan_dir, patient_id)
                if not os.path.isdir(patient_dir):
                    continue
                nifti_path = None
                json_path = None
                for fname in os.listdir(patient_dir):
                    if fname.endswith((".nii", ".nii.gz")) and nifti_path is None:
                        nifti_path = os.path.join(patient_dir, fname)
                    elif fname.endswith(".json") and json_path is None:
                        json_path = os.path.join(patient_dir, fname)
                if nifti_path is None:
                    continue
                samples.append({
                    "nifti": nifti_path,
                    "json": json_path,
                    "label": label,
                    "patient_id": patient_id,
                })

    return samples


def compute_tabular_stats(samples):
    """Compute mean/std of tabular features for normalization."""
    all_values = []
    for s in samples:
        if s.get("json") and os.path.exists(s["json"]):
            values, mask = load_metadata(s["json"])
            if mask.sum() > 0:
                all_values.append(values)
    if not all_values:
        return np.zeros(N_TABULAR), np.ones(N_TABULAR)
    arr = np.stack(all_values)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    return mean, std


def normalize_tabular_in_samples(samples, mean, std):
    """Normalize tabular features in-place using precomputed stats.

    We modify the JSON files' loaded values at dataset level instead,
    so this just returns the scaler params for saving.
    """
    return {"mean": mean.tolist(), "std": std.tolist()}


def train_one_epoch(model, loader, criterion, optimizer, device, tab_mean, tab_std):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        tabular_mask = batch["tabular_mask"].to(device)
        labels = batch["label"].to(device)

        # Normalize tabular features
        tab_mean_t = torch.tensor(tab_mean, device=device)
        tab_std_t = torch.tensor(tab_std, device=device)
        tabular = (tabular - tab_mean_t) / tab_std_t

        logits = model(images, tabular, tabular_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, tab_mean, tab_std):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        tabular_mask = batch["tabular_mask"].to(device)
        labels = batch["label"].to(device)

        tab_mean_t = torch.tensor(tab_mean, device=device)
        tab_std_t = torch.tensor(tab_std, device=device)
        tabular = (tabular - tab_mean_t) / tab_std_t

        logits = model(images, tabular, tabular_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Train the hybrid MRI scan-type classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data input modes (mutually exclusive):

  --data_dir   Nested dataset root:
                 data_dir/patient_id/input/nifti/smri_t1w/...

  Per-type dirs  Provide one or more of --t1_dir, --t2_dir, --t1ce_dir,
                 --flair_dir pointing to directories that contain NIfTI
                 files for that scan type (flat or one-level patient subdirs).
        """,
    )
    # --- data input (two modes) ---
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Nested dataset root (patient_id/input/nifti/smri_*/)")
    parser.add_argument("--t1_dir", type=str, default=None,
                        help="Directory containing T1w NIfTI files")
    parser.add_argument("--t2_dir", type=str, default=None,
                        help="Directory containing T2 NIfTI files")
    parser.add_argument("--t1ce_dir", type=str, default=None,
                        help="Directory containing T1ce NIfTI files")
    parser.add_argument("--flair_dir", type=str, default=None,
                        help="Directory containing FLAIR NIfTI files")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--unfreeze_epoch", type=int, default=10,
                        help="Epoch at which to unfreeze all CNN layers")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4,
                        help="LR after unfreezing")
    parser.add_argument("--n_slices", type=int, default=15)
    parser.add_argument("--tabular_dropout", type=float, default=0.3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # Load and split data by patient (not by volume) to avoid leakage
    scan_dirs = {
        label: path
        for label, path in [
            ("t1", args.t1_dir),
            ("t2", args.t2_dir),
            ("t1ce", args.t1ce_dir),
            ("flair", args.flair_dir),
        ]
        if path is not None
    }
    if scan_dirs:
        samples = load_samples_from_scan_dirs(scan_dirs)
    elif args.data_dir:
        samples = load_samples(args.data_dir)
    else:
        parser.error(
            "Provide --data_dir or at least one of "
            "--t1_dir / --t2_dir / --t1ce_dir / --flair_dir"
        )
    print(f"Found {len(samples)} volumes")

    patient_ids = list(set(s["patient_id"] for s in samples))
    train_pids, val_pids = train_test_split(
        patient_ids, test_size=args.val_split, random_state=args.seed
    )
    train_pids, val_pids = set(train_pids), set(val_pids)
    train_samples = [s for s in samples if s["patient_id"] in train_pids]
    val_samples = [s for s in samples if s["patient_id"] in val_pids]
    print(f"Train: {len(train_samples)} volumes, Val: {len(val_samples)} volumes")

    # Compute tabular normalization from training set only
    tab_mean, tab_std = compute_tabular_stats(train_samples)
    tab_stats = {"mean": tab_mean.tolist(), "std": tab_std.tolist()}
    with open(os.path.join(args.output_dir, "tabular_stats.json"), "w") as f:
        json.dump(tab_stats, f)

    # Datasets
    train_ds = MRISliceDataset(
        train_samples, n_slices=args.n_slices, augment=True,
        tabular_dropout=args.tabular_dropout,
    )
    val_ds = MRISliceDataset(
        val_samples, n_slices=args.n_slices, augment=False,
        tabular_dropout=0.0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = HybridMRIClassifier(n_classes=4, freeze_early=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Unfreeze all layers at specified epoch
        if epoch == args.unfreeze_epoch:
            print(f"\n--- Unfreezing all CNN layers at epoch {epoch} ---")
            model.unfreeze_all()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, tab_mean, tab_std
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, tab_mean, tab_std
        )
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pth"))
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
