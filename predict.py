import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import (MRIVolumeDataset, LABEL_NAMES, N_TABULAR,
                     load_metadata, extract_slices, normalize_slice)
from model import HybridMRIClassifier
from train import SCAN_FOLDER_TO_LABEL


def load_patient_samples(patient_dir):
    """Auto-discover scans from a patient directory.

    Accepts either:
      - Full path: patient_dir/input/nifti/smri_*/...
      - Direct nifti dir: patient_dir/smri_*/...
    """
    nifti_dir = os.path.join(patient_dir, "input", "nifti")
    if not os.path.isdir(nifti_dir):
        nifti_dir = patient_dir

    samples = []
    for scan_folder, label in SCAN_FOLDER_TO_LABEL.items():
        scan_dir = os.path.join(nifti_dir, scan_folder)
        if not os.path.isdir(scan_dir):
            continue

        nifti_path = None
        json_path = None
        for fname in os.listdir(scan_dir):
            if fname.endswith((".nii", ".nii.gz")):
                nifti_path = os.path.join(scan_dir, fname)
            elif fname.endswith(".json"):
                json_path = os.path.join(scan_dir, fname)

        if nifti_path:
            samples.append({
                "nifti": nifti_path,
                "json": json_path,
                "label": "t1",  # dummy, not used for prediction
            })
    return samples


@torch.no_grad()
def predict_volume(model, volume_data, device, tab_mean, tab_std):
    """Predict class for a single volume by averaging softmax across all slices + MIP."""
    model.eval()

    images = volume_data["images"].to(device)
    tabular = volume_data["tabular"].to(device)
    tabular_mask = volume_data["tabular_mask"].to(device)

    tab_mean_t = torch.tensor(tab_mean, device=device)
    tab_std_t = torch.tensor(tab_std, device=device)
    tabular = (tabular - tab_mean_t) / tab_std_t

    n = images.size(0)
    tabular = tabular.unsqueeze(0).expand(n, -1)
    tabular_mask = tabular_mask.unsqueeze(0).expand(n, -1)

    logits = model(images, tabular, tabular_mask)
    probs = F.softmax(logits, dim=1)

    avg_probs = probs.mean(dim=0)
    pred_class = avg_probs.argmax().item()
    confidence = avg_probs[pred_class].item()

    return {
        "predicted_label": LABEL_NAMES[pred_class],
        "confidence": confidence,
        "class_probabilities": {
            LABEL_NAMES[i]: avg_probs[i].item() for i in range(len(LABEL_NAMES))
        },
        "per_slice_predictions": [
            LABEL_NAMES[p.item()] for p in probs.argmax(dim=1)
        ],
    }


def predict_patient(model, sample_list, device, tab_mean, tab_std, n_slices=15):
    """Predict all volumes for a single patient."""
    dataset = MRIVolumeDataset(sample_list, n_slices=n_slices)
    results = []
    for i in range(len(dataset)):
        volume_data = dataset[i]
        result = predict_volume(model, volume_data, device, tab_mean, tab_std)
        result["file"] = sample_list[i]["nifti"]
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient_dir", type=str,
                       help="Patient directory (auto-discovers scans)")
    group.add_argument("--input", type=str, nargs="+",
                       help="Individual NIfTI file(s) to classify")
    parser.add_argument("--json", type=str, nargs="+", default=None,
                        help="JSON sidecar(s), same order as --input")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--tabular_stats", type=str,
                        default="checkpoints/tabular_stats.json")
    parser.add_argument("--n_slices", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    with open(args.tabular_stats, "r") as f:
        tab_stats = json.load(f)
    tab_mean = np.array(tab_stats["mean"], dtype=np.float32)
    tab_std = np.array(tab_stats["std"], dtype=np.float32)

    model = HybridMRIClassifier(n_classes=4, freeze_early=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.patient_dir:
        samples = load_patient_samples(args.patient_dir)
        if not samples:
            print(f"No scans found in {args.patient_dir}")
            return
    else:
        json_files = args.json or [None] * len(args.input)
        samples = [
            {"nifti": nii, "json": js, "label": "t1"}
            for nii, js in zip(args.input, json_files)
        ]

    results = predict_patient(model, samples, device, tab_mean, tab_std, args.n_slices)

    print("\n=== Predictions ===")
    for r in results:
        print(f"\n{r['file']}")
        print(f"  Predicted: {r['predicted_label']} "
              f"(confidence: {r['confidence']:.3f})")
        print(f"  Probabilities: {r['class_probabilities']}")
        print(f"  Per-slice votes: {r['per_slice_predictions']}")


if __name__ == "__main__":
    main()
