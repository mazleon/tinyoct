"""
Attention Visualizer — Week 7 key figure generator.

Generates the 'acceptance figure': side-by-side GradCAM++ overlays
showing that each RLAP directional stream activates on the correct pathology:
  CNV    → oblique orientation stream dominant (Bruch's membrane)
  DME    → vertical stream dominant (fluid columns)
  Drusen → horizontal stream dominant (layer deposits)
  Normal → balanced low activation (no dominant direction)
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("pytorch-grad-cam not installed. pip install grad-cam")


CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


class Visualizer:
    def __init__(self, model, device, output_dir: str = "./outputs/figures"):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_rlap_streams(
        self,
        image: torch.Tensor,
        label: int,
        save_name: str = None,
    ):
        """
        Generate the 4-panel attention overlay figure for one image.
        Panels: original | H-stream | V-stream | oblique (45°+60° mean)

        This is the paper's key figure (Figure 3 in ISBI submission).
        """
        self.model.eval()
        with torch.no_grad():
            x = image.unsqueeze(0).to(self.device)
            attn_maps = self.model.get_attention_maps(x)

        # Convert image to HWC numpy for display (undo normalisation)
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = [
            "Original",
            "Horizontal\n(layer thickness)",
            "Vertical\n(lesion columns)",
            "Oblique 45°+60°\n(Bruch's membrane)"
        ]

        # Panel 0: Original
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title(f"Original — {CLASS_NAMES[label]}", fontsize=11, fontweight="bold")

        # Panel 1: Horizontal stream
        if "horizontal" in attn_maps:
            h_map = attn_maps["horizontal"][0].mean(0).cpu()  # [H, 1]
            h_resized = torch.nn.functional.interpolate(
                h_map.unsqueeze(0).unsqueeze(0),
                size=img_np.shape[:2], mode="bilinear"
            ).squeeze().numpy()
            axes[1].imshow(img_np, cmap="gray", alpha=0.4)
            axes[1].imshow(h_resized, cmap="hot", alpha=0.6)
        axes[1].set_title(titles[1], fontsize=10)

        # Panel 2: Vertical stream
        if "vertical" in attn_maps:
            v_map = attn_maps["vertical"][0].mean(0).cpu()  # [1, W]
            v_resized = torch.nn.functional.interpolate(
                v_map.unsqueeze(0).unsqueeze(0),
                size=img_np.shape[:2], mode="bilinear"
            ).squeeze().numpy()
            axes[2].imshow(img_np, cmap="gray", alpha=0.4)
            axes[2].imshow(v_resized, cmap="hot", alpha=0.6)
        axes[2].set_title(titles[2], fontsize=10)

        # Panel 3: Oblique bank (mean of 45° and 60° — most relevant for CNV)
        if "orientation_bank" in attn_maps:
            bank = attn_maps["orientation_bank"]
            oblique_keys = [k for k in bank if "45" in k or "60" in k]
            if oblique_keys:
                o_map = torch.stack([bank[k][0].mean(0) for k in oblique_keys]).mean(0)
                o_map = o_map.squeeze().cpu().numpy()
                o_map = (o_map - o_map.min()) / (o_map.max() - o_map.min() + 1e-8)
                o_resized = torch.nn.functional.interpolate(
                    torch.tensor(o_map).unsqueeze(0).unsqueeze(0),
                    size=img_np.shape[:2], mode="bilinear"
                ).squeeze().numpy()
                axes[3].imshow(img_np, cmap="gray", alpha=0.4)
                axes[3].imshow(o_resized, cmap="hot", alpha=0.6)
        axes[3].set_title(titles[3], fontsize=10)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        save_path = self.output_dir / (save_name or f"attention_{CLASS_NAMES[label]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved attention figure: {save_path}")
        return save_path
