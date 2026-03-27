import os
import json
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP = {"t1": 0, "t2": 1, "flair": 2, "t1ce": 3}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
TABULAR_KEYS = ["RepetitionTime", "EchoTime", "InversionTime", "FlipAngle"]
N_TABULAR = len(TABULAR_KEYS)


def load_metadata(json_path):
    """Load DICOM metadata from dcm2niix JSON sidecar."""
    values = []
    mask = []
    with open(json_path, "r") as f:
        meta = json.load(f)
    for key in TABULAR_KEYS:
        if key in meta and meta[key] is not None:
            values.append(float(meta[key]))
            mask.append(1.0)
        else:
            values.append(0.0)
            mask.append(0.0)
    return np.array(values, dtype=np.float32), np.array(mask, dtype=np.float32)


def extract_slices(volume, n_slices=15):
    """Extract n evenly spaced axial slices + axial MIP from the volume.

    Returns a list of 2D arrays: n_slices regular slices followed by 1 MIP.
    """
    n_axial = volume.shape[2]
    # Skip top/bottom 10% to avoid empty slices
    start = int(n_axial * 0.1)
    end = int(n_axial * 0.9)
    indices = np.linspace(start, end, n_slices, dtype=int)

    slices = [volume[:, :, i] for i in indices]

    # Axial max-intensity projection
    mip = np.max(volume[:, :, start:end], axis=2)
    slices.append(mip)

    return slices


def normalize_slice(slice_2d):
    """Normalize a 2D slice to [0, 1]."""
    s = slice_2d.astype(np.float32)
    smin, smax = s.min(), s.max()
    if smax - smin > 0:
        s = (s - smin) / (smax - smin)
    return s


class MRISliceDataset(Dataset):
    """Training dataset: returns individual slices for per-slice classification.

    Each sample dict:
        {
            "nifti": "/path/to/volume.nii.gz",
            "json": "/path/to/volume.json",  # can be None
            "label": "t1"
        }
    """

    def __init__(self, samples, n_slices=15, image_size=224, augment=False,
                 tabular_dropout=0.3):
        self.samples = samples
        self.n_slices = n_slices
        self.slices_per_vol = n_slices + 1  # +1 for MIP
        self.image_size = image_size
        self.tabular_dropout = tabular_dropout

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples) * self.slices_per_vol

    def __getitem__(self, idx):
        vol_idx = idx // self.slices_per_vol
        slice_idx = idx % self.slices_per_vol
        sample = self.samples[vol_idx]

        vol = np.squeeze(nib.load(sample["nifti"]).get_fdata())
        all_slices = extract_slices(vol, self.n_slices)
        slice_2d = normalize_slice(all_slices[slice_idx])

        img = self.transform(slice_2d)
        img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img

        # Load metadata
        if sample.get("json") and os.path.exists(sample["json"]):
            tabular, mask = load_metadata(sample["json"])
        else:
            tabular = np.zeros(N_TABULAR, dtype=np.float32)
            mask = np.zeros(N_TABULAR, dtype=np.float32)

        # Modality dropout: randomly zero out all tabular features during training
        # so the CNN learns to classify standalone
        if self.tabular_dropout > 0 and np.random.rand() < self.tabular_dropout:
            mask = np.zeros(N_TABULAR, dtype=np.float32)

        label = LABEL_MAP[sample["label"].lower()]

        return {
            "image": img,
            "tabular": torch.tensor(tabular),
            "tabular_mask": torch.tensor(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "vol_idx": vol_idx,
            "is_mip": slice_idx == self.n_slices,
        }


class MRIVolumeDataset(Dataset):
    """Inference dataset: returns all slices + MIP for a volume at once."""

    def __init__(self, samples, n_slices=15, image_size=224):
        self.samples = samples
        self.n_slices = n_slices
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        vol = np.squeeze(nib.load(sample["nifti"]).get_fdata())
        all_slices = extract_slices(vol, self.n_slices)

        images = []
        for s in all_slices:
            img = self.transform(normalize_slice(s))
            img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img
            images.append(img)
        images = torch.stack(images)

        if sample.get("json") and os.path.exists(sample["json"]):
            tabular, mask = load_metadata(sample["json"])
        else:
            tabular = np.zeros(N_TABULAR, dtype=np.float32)
            mask = np.zeros(N_TABULAR, dtype=np.float32)

        label = LABEL_MAP[sample["label"].lower()] if "label" in sample else -1

        return {
            "images": images,
            "tabular": torch.tensor(tabular),
            "tabular_mask": torch.tensor(mask),
            "label": torch.tensor(label, dtype=torch.long),
        }
