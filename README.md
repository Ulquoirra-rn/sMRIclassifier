# sMRIclassifier

Classifies structural MRI scans into four modality types — **T1w, T2, T1ce, FLAIR** — using a hybrid ResNet-18 + tabular model that optionally fuses DICOM acquisition metadata (TR, TE, TI, flip angle).

---

## Architecture

```
NIfTI volume
  └─ n evenly-spaced axial slices + 1 axial MIP
       └─ ResNet-18 (pretrained, ImageNet) → 512-d embedding
                                                   │
DICOM JSON sidecar (optional)                      │
  └─ [RepetitionTime, EchoTime,                    │
       InversionTime, FlipAngle]                   │
       └─ masked MLP → 64-d embedding ─────────────┤
                                                   │
                                             concat (576-d)
                                                   │
                                            FC → 4 classes
                                         (t1 / t2 / t1ce / flair)
```

**Training strategy:**
- ResNet-18 early layers are frozen for the first `--unfreeze_epoch` epochs; then all parameters are unfrozen for fine-tuning at a lower learning rate.
- Tabular dropout randomly zeros the metadata branch during training so the CNN learns to classify without it.
- Per-slice cross-entropy loss; predictions are aggregated across all slices via softmax averaging at inference time.
- Train/val split is done at the **patient level** to prevent data leakage.

---

## Data formats

### Per-scan-type directories (recommended)

Provide one directory per modality.  Two layouts are supported:

**Flat** — NIfTI files sit directly in the scan directory.  The filename stem is used as the patient ID.

```
t1_data/
  patient001.nii.gz
  patient001.json    # optional DICOM sidecar
  patient002.nii.gz
t2_data/
  patient001.nii.gz
  patient002.nii.gz
```

**Nested** — Each patient has its own sub-directory.

```
t1_data/
  patient001/
    anat.nii.gz
    anat.json         # optional
  patient002/
    anat.nii.gz
t2_data/
  patient001/
    anat.nii.gz
```

### Nested dataset root (`--data_dir`)

Each top-level entry under `data_dir` is treated as a patient.  The scan-type
folders (`smri_t1w`, `smri_t2`, `smri_t1ce`, `smri_flair`) are located by
walking the entire patient sub-tree, so intermediate directories at any depth
are handled transparently — session IDs, visit labels, extra nesting, etc.

The entire tree under `data_dir` is walked recursively — no assumptions are
made about the number of intermediate directory levels.  Scan-type folders
(`smri_t1w`, `smri_t2`, `smri_t1ce`, `smri_flair`) are located wherever they
appear.  Sibling scan-type folders sharing the same parent directory are
grouped as one patient/visit for train-val splitting.

```
# Shallow layout
data_dir/patient001/input/nifti/smri_t1w/scan.nii.gz

# Deep / segmented ID layout
data_dir/40/00/0011/Visit1_Dataentry_2/Glioma/input/nifti/smri_t1w/scan.nii.gz
data_dir/40/00/0011/Visit1_Dataentry_2/Glioma/input/nifti/smri_flair/scan.nii.gz

# Mixed depth — all handled automatically
data_dir/patient002/ses-01/input/nifti/smri_t2/scan.nii.gz
```

Not all scan types need to be present for every patient.

---

## Training

### Using per-scan-type directories

```bash
python train.py \
  --t1_dir    /data/t1 \
  --t2_dir    /data/t2 \
  --t1ce_dir  /data/t1ce \
  --flair_dir /data/flair \
  --output_dir checkpoints/
```

Any subset of scan-type dirs can be provided (e.g. only `--t1_dir` and `--t2_dir` for a two-class problem, though the model head is fixed at 4 classes).

### Using the legacy nested dataset root

```bash
python train.py \
  --data_dir /data/patients \
  --output_dir checkpoints/
```

### All training options

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | — | Legacy nested dataset root |
| `--t1_dir` | — | Directory of T1w NIfTI files |
| `--t2_dir` | — | Directory of T2 NIfTI files |
| `--t1ce_dir` | — | Directory of T1ce NIfTI files |
| `--flair_dir` | — | Directory of FLAIR NIfTI files |
| `--output_dir` | `checkpoints` | Where to save model + tabular stats |
| `--epochs` | `500` | Maximum training epochs |
| `--unfreeze_epoch` | `50` | Epoch at which to unfreeze all CNN layers |
| `--early_stopping_patience` | `20` | Stop if val accuracy does not improve for this many epochs (0 to disable) |
| `--batch_size` | `16` | Batch size |
| `--lr` | `1e-3` | Initial learning rate (frozen stage) |
| `--lr_finetune` | `1e-4` | Learning rate after unfreezing |
| `--n_slices` | `15` | Axial slices sampled per volume |
| `--tabular_dropout` | `0.3` | Probability of zeroing metadata branch |
| `--val_split` | `0.2` | Fraction of patients held out for validation |
| `--seed` | `42` | Random seed |

Outputs saved to `--output_dir`:
- `best_model.pth` — model weights at best validation accuracy
- `tabular_stats.json` — mean/std for tabular feature normalisation (required at inference)

---

## Inference

### Classify a patient directory (auto-discovery)

```bash
python predict.py \
  --patient_dir /data/patients/patient001 \
  --checkpoint  checkpoints/best_model.pth \
  --tabular_stats checkpoints/tabular_stats.json
```

### Classify individual NIfTI files

```bash
python predict.py \
  --input scan1.nii.gz scan2.nii.gz \
  --json  scan1.json   scan2.json \
  --checkpoint checkpoints/best_model.pth \
  --tabular_stats checkpoints/tabular_stats.json
```

`--json` is optional; omit it (or pass fewer entries) if no metadata is available.

---

## Dependencies

```
torch torchvision
nibabel
scikit-learn
numpy
```
