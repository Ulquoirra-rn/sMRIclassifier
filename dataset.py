import hashlib
import logging
import os
import json
from collections import OrderedDict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image as PILImage

log = logging.getLogger(__name__)


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


def extract_slices(volume, n_slices=15, image_size=None):
    """Extract n evenly spaced axial slices + axial MIP from the volume.

    Returns a list of 2D float32 arrays: n_slices regular slices followed by
    1 MIP.  If image_size is given every slice is resized to
    (image_size, image_size) immediately, capping memory for high-res scans.
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

    if image_size is not None:
        resized = []
        for s in slices:
            arr = s.astype(np.float32)
            # Collapse any trailing dimensions (e.g. complex/multi-component)
            while arr.ndim > 2:
                arr = arr[..., 0]
            if arr.shape[0] != image_size or arr.shape[1] != image_size:
                arr = np.array(
                    PILImage.fromarray(arr, mode="F").resize(
                        (image_size, image_size), PILImage.BILINEAR
                    )
                )
            resized.append(arr)
        return resized

    return slices


def normalize_slice(slice_2d):
    """Normalize a 2D slice to [0, 1]."""
    s = slice_2d.astype(np.float32)
    smin, smax = s.min(), s.max()
    if smax - smin > 0:
        s = (s - smin) / (smax - smin)
    return s


def _cache_path(cache_dir, nifti_path, image_size):
    key = f"{os.path.abspath(nifti_path)}:{image_size}"
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{h}.npz")


def _get_volume_slices(nifti_path, n_slices, image_size, cache_dir):
    """Return extracted slices for a volume, using a disk cache when available.

    Cache files are stored as .npz under cache_dir and validated against the
    source file's mtime — if the NIfTI is replaced or modified the cache is
    automatically rebuilt.  Writes are atomic (tmp → rename) so a crashed
    worker never leaves a corrupt cache entry.
    """
    if cache_dir is not None:
        path = _cache_path(cache_dir, nifti_path, image_size)
        if os.path.exists(path):
            try:
                cached = np.load(path)
                if float(cached["mtime"]) == os.path.getmtime(nifti_path):
                    return list(cached["slices"])  # list of (image_size, image_size) arrays
            except Exception:
                pass  # corrupted cache — fall through to recompute

    vol = np.squeeze(nib.load(nifti_path).get_fdata())
    if vol.ndim > 3:
        vol = vol[..., 0]
    slices = extract_slices(vol, n_slices, image_size=image_size)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        mtime = os.path.getmtime(nifti_path)
        tmp = path + ".tmp"
        try:
            np.savez(tmp, slices=np.stack(slices), mtime=np.float64(mtime))
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)

    return slices


class MRISliceDataset(Dataset):
    """Training dataset: returns individual slices for per-slice classification.

    Uses a per-worker LRU volume cache (default size 8) so that consecutive
    requests for slices from the same volume hit RAM instead of disk, without
    holding the entire dataset in memory.  Set lru_size=0 to disable (pure
    lazy loading) or pass preload=True to load everything upfront if RAM allows.
    """

    def __init__(self, samples, n_slices=15, image_size=224, augment=False,
                 tabular_dropout=0.3, cache_dir=None, lru_size=8, preload=False):
        self.samples = samples
        self.n_slices = n_slices
        self.slices_per_vol = n_slices + 1  # +1 for MIP
        self.image_size = image_size
        self.tabular_dropout = tabular_dropout
        self.cache_dir = cache_dir
        self.lru_size = lru_size

        if preload:
            log.info(f"Preloading {len(samples)} volumes into RAM...")
            self._preloaded = []
            for i, s in enumerate(samples):
                slices = _get_volume_slices(s["nifti"], n_slices, image_size, cache_dir)
                self._preloaded.append(np.stack(slices))
                if (i + 1) % 100 == 0 or (i + 1) == len(samples):
                    log.info(f"  Preloaded {i+1}/{len(samples)} volumes")
        else:
            self._preloaded = None
            # Per-worker LRU: populated lazily in _get_slices()
            # (initialised in each worker via worker_init or on first access)
            self._lru: OrderedDict = None

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def _get_slices(self, vol_idx):
        """Return the (slices_per_vol, H, W) array for vol_idx, using LRU cache."""
        if self._preloaded is not None:
            return self._preloaded[vol_idx]

        # Initialise LRU dict on first access within this worker process
        if self._lru is None:
            self._lru = OrderedDict()

        if vol_idx in self._lru:
            self._lru.move_to_end(vol_idx)
            return self._lru[vol_idx]

        slices = _get_volume_slices(
            self.samples[vol_idx]["nifti"], self.n_slices,
            self.image_size, self.cache_dir,
        )
        arr = np.stack(slices)
        self._lru[vol_idx] = arr
        self._lru.move_to_end(vol_idx)
        if self.lru_size and len(self._lru) > self.lru_size:
            self._lru.popitem(last=False)
        return arr

    def __len__(self):
        return len(self.samples) * self.slices_per_vol

    def __getitem__(self, idx):
        vol_idx = idx // self.slices_per_vol
        slice_idx = idx % self.slices_per_vol
        sample = self.samples[vol_idx]

        slice_2d = normalize_slice(self._get_slices(vol_idx)[slice_idx])
        img = self.transform(slice_2d)
        img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img

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

    def __init__(self, samples, n_slices=15, image_size=224, cache_dir=None):
        self.samples = samples
        self.n_slices = n_slices
        self.image_size = image_size

        # Preload all slices into RAM
        self._slices = []
        for s in samples:
            slices = _get_volume_slices(s["nifti"], n_slices, image_size, cache_dir)
            self._slices.append(np.stack(slices))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images = []
        for s in self._slices[idx]:
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
