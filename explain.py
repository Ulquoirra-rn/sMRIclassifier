"""Grad-CAM explainability for the hybrid MRI classifier."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from torchvision import transforms

from dataset import (extract_slices, normalize_slice, load_metadata,
                     N_TABULAR, LABEL_NAMES)
from model import HybridMRIClassifier


class GradCAM:
    """Grad-CAM for the CNN branch of HybridMRIClassifier."""

    def __init__(self, model, target_layer=None):
        self.model = model
        # Default to last conv block of ResNet
        self.target_layer = target_layer or model.cnn.layer4[-1]
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image, tabular, tabular_mask, target_class=None):
        """Generate Grad-CAM heatmap for a single image.

        Args:
            image: (1, 3, H, W) tensor
            tabular: (1, N_TABULAR) tensor
            tabular_mask: (1, N_TABULAR) tensor
            target_class: class index to explain. If None, uses predicted class.

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
            predicted_class: int
        """
        self.model.eval()
        image.requires_grad_(True)

        logits = self.model(image, tabular, tabular_mask)
        pred_class = logits.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, pred_class


def visualize_gradcam(image_2d, heatmap, pred_label, confidence, save_path=None):
    """Plot original image with Grad-CAM overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_2d, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(image_2d, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title(f"Overlay — {pred_label} ({confidence:.2f})")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="NIfTI file to explain")
    parser.add_argument("--json", type=str, default=None,
                        help="JSON sidecar")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--tabular_stats", type=str,
                        default="checkpoints/tabular_stats.json")
    parser.add_argument("--output_dir", type=str, default="explanations")
    parser.add_argument("--n_slices", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # Load model
    model = HybridMRIClassifier(n_classes=4, freeze_early=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Load tabular stats
    with open(args.tabular_stats, "r") as f:
        tab_stats = json.load(f)
    tab_mean = torch.tensor(tab_stats["mean"], dtype=torch.float32, device=device)
    tab_std = torch.tensor(tab_stats["std"], dtype=torch.float32, device=device)

    # Load metadata
    if args.json and os.path.exists(args.json):
        tabular, mask = load_metadata(args.json)
    else:
        tabular = np.zeros(N_TABULAR, dtype=np.float32)
        mask = np.zeros(N_TABULAR, dtype=np.float32)
    tabular = torch.tensor(tabular, device=device).unsqueeze(0)
    tabular = (tabular - tab_mean) / tab_std
    mask = torch.tensor(mask, device=device).unsqueeze(0)

    # Load volume and extract slices
    vol = nib.load(args.input).get_fdata()
    all_slices = extract_slices(vol, args.n_slices)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    grad_cam = GradCAM(model)
    basename = os.path.splitext(os.path.basename(args.input))[0].replace(".nii", "")

    for i, s in enumerate(all_slices):
        slice_2d = normalize_slice(s)
        img = transform(slice_2d).repeat(3, 1, 1).unsqueeze(0).to(device)

        heatmap, pred_class = grad_cam.generate(img, tabular, mask)
        pred_label = LABEL_NAMES[pred_class]
        confidence = F.softmax(
            model(img, tabular, mask), dim=1
        )[0, pred_class].item()

        slice_name = f"slice_{i:02d}" if i < args.n_slices else "mip"
        save_path = os.path.join(args.output_dir, f"{basename}_{slice_name}.png")
        visualize_gradcam(slice_2d, heatmap, pred_label, confidence, save_path)

    print(f"\nAll explanations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
