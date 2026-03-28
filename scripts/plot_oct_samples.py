import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def main():
    base_dir = "data/oct2017/train"
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    
    # Check if dir exists
    if not os.path.exists(base_dir):
        print(f"Warning: directory {base_dir} not found. Skipping plot generation.")
        return
        
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls)
        try:
            # find first valid image
            files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
            if not files:
                continue
            r_file = random.choice(files)
            img_path = os.path.join(cls_dir, r_file)
            img = Image.open(img_path).convert('L')
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(cls, fontsize=12, fontweight='bold')
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading class {cls}: {e}")
            axes[i].set_title(cls)
            axes[i].axis('off')
            
    plt.tight_layout()
    out_path = "paper/figures/oct_samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved dataset grid to {out_path}")

if __name__ == "__main__":
    main()
