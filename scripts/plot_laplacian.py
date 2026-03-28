import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def main():
    base_dir = "data/oct2017/train"
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    
    # Check if dir exists
    if not os.path.exists(base_dir):
        print(f"Warning: directory {base_dir} not found.")
        return
        
    random.seed(42) # Ensure reproducible "good" random images
    
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        # Pick a random file, or just use a fixed list if needed
        # We will iterate a few times if the image is too dark, but random is usually fine
        r_file = random.choice(files)
        img_path = os.path.join(cls_dir, r_file)
        
        # Load and resize
        img = Image.open(img_path).convert('L').resize((224, 224))
        img_arr = np.array(img).astype(np.float32) / 255.0
        
        # Apply Convolution
        laplacian_img = convolve2d(img_arr, laplacian_kernel, mode='same', boundary='symm')
        
        # Extract magnitude for visualization to highlight edges clearly against dark background
        laplacian_vis = np.abs(laplacian_img)
        
        # Normalize dynamically based on 99th percentile to avoid single hot-pixels washing it out
        p99 = np.percentile(laplacian_vis, 99)
        if p99 > 0:
            laplacian_vis = laplacian_vis / p99
        laplacian_vis = np.clip(laplacian_vis, 0, 1)
        
        # Plot Top Row (Original)
        axes[0, i].imshow(img_arr, cmap='gray')
        axes[0, i].set_title(f"Input: {cls}", fontsize=14, fontweight='bold')
        axes[0, i].axis('off')
        
        # Plot Bottom Row (Laplacian)
        axes[1, i].imshow(laplacian_vis, cmap='gray')
        axes[1, i].set_title(f"Laplacian Edge Map", fontsize=14, fontweight='bold')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    out_path = "paper/figures/laplacian_effect.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced 2x4 Laplacian grid to {out_path}")

if __name__ == "__main__":
    main()
