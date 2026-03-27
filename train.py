import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import MRISliceDataset, VolumeGroupedSampler, N_TABULAR, load_metadata
from model import HybridMRIClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


SCAN_FOLDER_TO_LABEL = {
    "smri_t1w": "t1",
    "smri_t2": "t2",
    "smri_t1ce": "t1ce",
    "smri_flair": "flair",
}


def load_samples(data_dir):
    """Walk the entire data_dir tree and collect all scan-type directories.

    Scan-type folders (smri_t1w, smri_t2, smri_t1ce, smri_flair) are found
    at any depth — no assumptions are made about the number of intermediate
    directory levels.

    Scans that share the same parent directory are treated as the same
    patient/visit for train-val splitting (all scan types under one 'nifti/'
    folder stay together in the same split partition).

    Example (all supported):
        data_dir/40/00/0011/Visit1/Glioma/input/nifti/smri_t1w/scan.nii.gz
        data_dir/patient_id/input/nifti/smri_t1w/scan.nii.gz
        data_dir/patient_id/ses-01/input/nifti/smri_flair/scan.nii.gz
    """
    # parent_dir -> {scan_folder_name -> {"nifti": ..., "json": ...}}
    groups = {}

    for dirpath, _, filenames in os.walk(data_dir):
        folder_name = os.path.basename(dirpath)
        if folder_name not in SCAN_FOLDER_TO_LABEL:
            continue

        parent = os.path.dirname(dirpath)
        if parent not in groups:
            groups[parent] = {}
        if folder_name in groups[parent]:
            continue  # duplicate scan type under same parent — keep first

        nifti_path = None
        json_path = None
        for fname in filenames:
            if fname.endswith((".nii", ".nii.gz")) and nifti_path is None:
                nifti_path = os.path.join(dirpath, fname)
            elif fname.endswith(".json") and json_path is None:
                json_path = os.path.join(dirpath, fname)

        if nifti_path is not None:
            groups[parent][folder_name] = {"nifti": nifti_path, "json": json_path}

    samples = []
    for parent, scan_map in groups.items():
        # Relative path of the parent dir acts as the patient/visit identifier
        patient_id = os.path.relpath(parent, data_dir)
        for scan_folder, paths in scan_map.items():
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


def train_one_epoch(model, loader, criterion, optimizer, device, tab_mean, tab_std,
                    log_interval=20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    n_batches = len(loader)
    t0 = time.time()

    tab_mean_t = torch.tensor(tab_mean, device=device)
    tab_std_t = torch.tensor(tab_std, device=device)

    for i, batch in enumerate(loader):
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        tabular_mask = batch["tabular_mask"].to(device)
        labels = batch["label"].to(device)

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

        if (i + 1) % log_interval == 0 or (i + 1) == n_batches:
            elapsed = time.time() - t0
            log.info(
                f"  [train] batch {i+1:4d}/{n_batches} | "
                f"loss {total_loss/total:.4f} | "
                f"acc {correct/total:.4f} | "
                f"{elapsed:.1f}s elapsed"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, tab_mean, tab_std, log_interval=20):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    n_batches = len(loader)
    t0 = time.time()

    tab_mean_t = torch.tensor(tab_mean, device=device)
    tab_std_t = torch.tensor(tab_std, device=device)

    for i, batch in enumerate(loader):
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        tabular_mask = batch["tabular_mask"].to(device)
        labels = batch["label"].to(device)

        tabular = (tabular - tab_mean_t) / tab_std_t

        logits = model(images, tabular, tabular_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (i + 1) % log_interval == 0 or (i + 1) == n_batches:
            elapsed = time.time() - t0
            log.info(
                f"  [val]   batch {i+1:4d}/{n_batches} | "
                f"loss {total_loss/total:.4f} | "
                f"acc {correct/total:.4f} | "
                f"{elapsed:.1f}s elapsed"
            )

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
    parser.add_argument("--cache_dir", type=str, default=".slice_cache",
                        help="Directory for cached extracted slices "
                             "(set to empty string to disable)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--unfreeze_epoch", type=int, default=50,
                        help="Epoch at which to unfreeze all CNN layers")
    parser.add_argument("--early_stopping_patience", type=int, default=20,
                        help="Stop after this many consecutive epochs without "
                             "val loss improvement (0 to disable)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4,
                        help="Minimum decrease in val loss to count as improvement")
    parser.add_argument("--batch_size", type=int, default=51,
                        help="Ideally n_slices+1 so each batch is exactly one volume")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Print batch-level progress every N batches")
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
    log.info(f"Using device: {device}")

    # Load and split data by patient (not by volume) to avoid leakage
    log.info("Scanning dataset directories...")
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
    log.info(f"Found {len(samples)} volumes")

    patient_ids = list(set(s["patient_id"] for s in samples))
    train_pids, val_pids = train_test_split(
        patient_ids, test_size=args.val_split, random_state=args.seed
    )
    train_pids, val_pids = set(train_pids), set(val_pids)
    train_samples = [s for s in samples if s["patient_id"] in train_pids]
    val_samples = [s for s in samples if s["patient_id"] in val_pids]
    log.info(f"Train: {len(train_samples)} volumes | Val: {len(val_samples)} volumes")

    # Compute tabular normalization from training set only
    log.info("Computing tabular feature statistics...")
    tab_mean, tab_std = compute_tabular_stats(train_samples)
    tab_stats = {"mean": tab_mean.tolist(), "std": tab_std.tolist()}
    with open(os.path.join(args.output_dir, "tabular_stats.json"), "w") as f:
        json.dump(tab_stats, f)
    log.info("Tabular stats saved.")

    # Datasets
    cache_dir = args.cache_dir if args.cache_dir else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        log.info(f"Slice cache: {os.path.abspath(cache_dir)}")
    else:
        log.info("Slice cache disabled.")

    log.info("Building datasets...")
    train_ds = MRISliceDataset(
        train_samples, n_slices=args.n_slices, augment=True,
        tabular_dropout=args.tabular_dropout, cache_dir=cache_dir,
    )
    val_ds = MRISliceDataset(
        val_samples, n_slices=args.n_slices, augment=False,
        tabular_dropout=0.0, cache_dir=cache_dir,
    )
    log.info(
        f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)} | "
        f"Batch size: {args.batch_size}"
    )
    train_sampler = VolumeGroupedSampler(train_ds, shuffle=True)
    val_sampler = VolumeGroupedSampler(val_ds, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=2, pin_memory=False)

    # Model
    log.info("Initialising model...")
    model = HybridMRIClassifier(n_classes=4, freeze_early=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {n_params:,} total | {n_trainable:,} trainable")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    log.info(f"Starting training: max {args.epochs} epochs | "
             f"early stopping patience {args.early_stopping_patience} consecutive epochs | "
             f"min delta {args.early_stopping_min_delta}")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    consecutive_no_improve = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Unfreeze all layers at specified epoch; reset patience so the loss
        # spike from newly unfrozen weights doesn't trigger early stopping
        if epoch == args.unfreeze_epoch:
            log.info(f"--- Unfreezing all CNN layers at epoch {epoch+1} ---")
            model.unfreeze_all()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"Trainable parameters after unfreeze: {n_trainable:,}")
            consecutive_no_improve = 0
            log.info("Early-stopping counter reset after unfreeze.")

        log.info(f"Epoch {epoch+1}/{args.epochs} — training...")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, tab_mean, tab_std,
            log_interval=args.log_interval,
        )

        log.info(f"Epoch {epoch+1}/{args.epochs} — validating...")
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, tab_mean, tab_std,
            log_interval=args.log_interval,
        )
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        log.info(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:.4f} | "
            f"LR {lr:.2e} | {elapsed:.1f}s"
        )

        # Improvement = val loss decreased by at least min_delta
        if val_loss < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = val_loss
            consecutive_no_improve = 0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pth"))
            log.info(f"  -> New best model saved "
                     f"(val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            consecutive_no_improve += 1
            log.info(f"  -> No improvement for {consecutive_no_improve} consecutive "
                     f"epoch(s) (patience {args.early_stopping_patience})")

        if (args.early_stopping_patience > 0
                and consecutive_no_improve >= args.early_stopping_patience):
            log.info(f"Early stopping: val loss did not improve by more than "
                     f"{args.early_stopping_min_delta} for "
                     f"{args.early_stopping_patience} consecutive epochs.")
            break

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f} | "
             f"Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
