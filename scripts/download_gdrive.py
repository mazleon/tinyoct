#!/usr/bin/env python3
"""
Google Drive Dataset Downloader for TinyOCT
============================================
Downloads OCT2017.zip and OCTID.zip from a private Google Drive folder,
extracts them to the correct data/ paths, and optionally auto-downloads OCTMNIST.

Designed for RunPod GPU servers where local disk space is limited
and datasets are pre-staged on Google Drive.

Usage
-----
    # Full download (recommended — runs everything)
    uv run scripts/download_gdrive.py

    # Skip OCTMNIST (if already downloaded or not needed)
    uv run scripts/download_gdrive.py --skip-octmnist

    # Override Google Drive folder ID
    uv run scripts/download_gdrive.py --folder-id <YOUR_FOLDER_ID>

    # Only verify existing data without downloading
    uv run scripts/download_gdrive.py --verify-only

    # Keep zip files after extraction (saves re-download time)
    uv run scripts/download_gdrive.py --keep-zips

Requirements
------------
    pip install gdown         (auto-installed if missing)
    pip install medmnist      (for OCTMNIST auto-download)

Expected output structure
--------------------------
    data/
    ├── OCT2017/
    │   ├── train/
    │   │   ├── CNV/       (*.jpeg)
    │   │   ├── DME/       (*.jpeg)
    │   │   ├── DRUSEN/    (*.jpeg)
    │   │   └── NORMAL/    (*.jpeg)
    │   └── test/
    │       ├── CNV/
    │       ├── DME/
    │       ├── DRUSEN/
    │       └── NORMAL/
    ├── OCTID/
    │   ├── NORMAL/        (*.jpg)
    │   ├── DR/            (*.jpg)
    │   └── AMD/           (*.jpg)
    └── medmnist/          (auto-downloaded)
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — update these if your Drive layout changes
# ─────────────────────────────────────────────────────────────────────────────

# Google Drive folder ID from URL: drive.google.com/drive/folders/<ID>
DEFAULT_FOLDER_ID = "1EpyO7CGtZnaqy_T5MS_s0TDaSOQu--ZK"

# Where to look for zips after gdown folder download
# gdown creates a subfolder named after the Drive folder by default
GDRIVE_FOLDER_NAME = "tinyoct-datasets"

# Target dataset directories (relative to project root)
DATA_DIR = Path("data")
OCT2017_DIR = DATA_DIR / "oct2017"    # lowercase — matches configs/base.yaml oct2017_path
OCTID_DIR = DATA_DIR / "OCTID"        # uppercase — matches configs/base.yaml octid_path
MEDMNIST_DIR = DATA_DIR / "medmnist"

# Expected zip filenames in Google Drive
ZIP_MAPPING = {
    "oct2017": {
        "candidates": ["OCT2017.zip", "oct2017.zip", "Kermany2018.zip", "kermany2018.zip"],
        "target_dir": OCT2017_DIR,
        "extract_subdir": None,   # will auto-detect after extraction
        "label": "OCT2017 (Kermany)",
    },
    "octid": {
        "candidates": ["OCTID.zip", "octid.zip", "OCTID_dataset.zip"],
        "target_dir": OCTID_DIR,
        "extract_subdir": None,
        "label": "OCTID (cross-scanner OOD)",
    },
}

# Validation: these class folders must exist after extraction
REQUIRED_OCT2017 = {
    "train": ["CNV", "DME", "DRUSEN", "NORMAL"],
    "test":  ["CNV", "DME", "DRUSEN", "NORMAL"],
}
REQUIRED_OCTID = ["NORMAL", "DR", "AMD"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str, char: str = "─", width: int = 60):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _ok(msg: str):   print(f"  ✅  {msg}")
def _warn(msg: str): print(f"  ⚠️   {msg}")
def _err(msg: str):  print(f"  ❌  {msg}", file=sys.stderr)
def _info(msg: str): print(f"  ℹ️   {msg}")


def ensure_gdown():
    """Import gdown, auto-installing it if missing."""
    try:
        import gdown
        return gdown
    except ImportError:
        _info("gdown not found — installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "--quiet"],
            stdout=subprocess.DEVNULL,
        )
        import gdown  # noqa: PLC0415
        _ok("gdown installed successfully")
        return gdown


def find_zip_in_dir(search_dir: Path, candidates: list[str]) -> Path | None:
    """Return the first matching zip found under search_dir (recursive)."""
    for name in candidates:
        matches = list(search_dir.rglob(name))
        if matches:
            return matches[0]
    # Fallback: any .zip file in the directory
    zips = list(search_dir.rglob("*.zip"))
    if zips:
        _warn(f"No named match found; using first zip found: {zips[0].name}")
        return zips[0]
    return None


def count_images(directory: Path) -> int:
    """Count JPEG/JPG/PNG images recursively in a directory."""
    total = 0
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        total += len(list(directory.rglob(ext)))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_gdrive_folder(gdown, folder_id: str, download_root: Path) -> Path:
    """
    Download the entire Google Drive folder into download_root.
    Returns the path to the downloaded folder.
    """
    download_root.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    _info(f"Downloading Drive folder → {download_root}/")
    _info(f"URL: {url}")
    _info("This may take a while for large datasets (~2+ GB)...\n")

    try:
        output = gdown.download_folder(
            url=url,
            output=str(download_root),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,  # don't fail if some files are skipped
        )
        if output is None:
            raise RuntimeError("gdown returned None — folder may be private or rate-limited.")
        downloaded_path = Path(output) if isinstance(output, str) else download_root
        _ok(f"Folder downloaded to: {downloaded_path}")
        return downloaded_path

    except Exception as exc:
        _err(f"gdown folder download failed: {exc}")
        _info("Tip: If the folder is private, make sure it is shared as 'Anyone with the link'.")
        _info("     You can also download individual files by ID:")
        _info("       gdown <FILE_ID> -O data/OCT2017.zip")
        _info("     Then re-run this script with --skip-download to only extract.")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_oct2017_root(extract_tmp: Path) -> Path | None:
    """
    After extracting OCT2017.zip, locate the train/ directory.
    Handles nested zip structures (e.g. OCT2017/OCT2017/train/).
    """
    # Common patterns Kaggle zips produce (case-insensitive search)
    candidates = [
        extract_tmp / "OCT2017",
        extract_tmp / "oct2017",
        extract_tmp / "Kermany2018",
        extract_tmp / "kermany2018",
        extract_tmp / "CellData" / "OCT",   # Kaggle alternate structure
        extract_tmp,                          # already at root
    ]
    for candidate in candidates:
        if (candidate / "train").exists():
            return candidate
    # Deep search
    for p in extract_tmp.rglob("train"):
        if p.is_dir() and any((p / cls).exists() for cls in ["CNV", "DME", "DRUSEN", "NORMAL"]):
            return p.parent
    return None


def _find_octid_root(extract_tmp: Path) -> Path | None:
    """
    Locate the root of OCTID after extraction.
    Expects folders like NORMAL/, DR/, AMD/ at the root.
    """
    octid_classes = {"NORMAL", "DR", "AMD", "CNV", "DME", "DRUSEN"}
    for p in [extract_tmp, *extract_tmp.iterdir() if extract_tmp.is_dir() else []]:
        if p.is_dir():
            subdirs = {d.name.upper() for d in p.iterdir() if d.is_dir()}
            if subdirs & octid_classes:  # at least one class matches
                return p
    return None


def extract_zip(zip_path: Path, target_dir: Path, dataset: str) -> bool:
    """
    Extract a zip file to target_dir, handling nested directory structures.
    Returns True on success.
    """
    if target_dir.exists() and count_images(target_dir) > 0:
        _warn(f"{dataset} already extracted at {target_dir} — skipping extraction.")
        _info("  (Delete the directory and re-run to force re-extraction.)")
        return True

    _info(f"Extracting {zip_path.name} ({zip_path.stat().st_size / 1e9:.2f} GB)...")

    # Extract to a temp directory first, then move to correct place
    tmp_dir = target_dir.parent / f"_tmp_{dataset}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        _ok(f"Extraction complete → {tmp_dir}")

        # Locate the actual data root inside the extraction
        if dataset == "oct2017":
            data_root = _find_oct2017_root(tmp_dir)
        elif dataset == "octid":
            data_root = _find_octid_root(tmp_dir)
        else:
            data_root = tmp_dir

        if data_root is None:
            _err(f"Could not locate data root inside extracted archive at {tmp_dir}")
            _info("  Please check the zip structure and set up manually:")
            _info(f"  Expected: train/CNV, train/DME, train/DRUSEN, train/NORMAL")
            return False

        # Move to final destination
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(data_root), str(target_dir))
        _ok(f"Moved to final path: {target_dir}")

    finally:
        # Clean up temp dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# OCTMNIST AUTO-DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_octmnist():
    """Auto-download OCTMNIST (224×224) via medmnist library."""
    _banner("OCTMNIST Auto-Download")

    MEDMNIST_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already present
    existing = list(MEDMNIST_DIR.rglob("*.npz"))
    if existing:
        _ok(f"OCTMNIST already present at {MEDMNIST_DIR} ({len(existing)} file(s))")
        return True

    try:
        import medmnist
        from medmnist import OCTMNIST
        import torchvision.transforms as T

        tf = T.ToTensor()
        for split in ["train", "val", "test"]:
            _info(f"Downloading OCTMNIST split: {split}...")
            OCTMNIST(split=split, size=224, transform=tf, download=True,
                     root=str(MEDMNIST_DIR))
        _ok(f"OCTMNIST ready at {MEDMNIST_DIR}")
        return True

    except ImportError:
        _warn("medmnist not installed. To download OCTMNIST, run:")
        _warn("  pip install medmnist")
        _warn("  python scripts/download_gdrive.py --octmnist-only")
        return False
    except Exception as exc:
        _err(f"OCTMNIST download failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_oct2017() -> bool:
    """Verify OCT2017 directory structure and report image counts."""
    _banner("Verifying OCT2017")
    ok = True

    for split, classes in REQUIRED_OCT2017.items():
        for cls in classes:
            cls_dir = OCT2017_DIR / split / cls
            if not cls_dir.exists():
                _err(f"Missing: {cls_dir}")
                ok = False
            else:
                n = count_images(cls_dir)
                status = _ok if n > 0 else _err
                status(f"OCT2017/{split}/{cls}: {n:,} images")
                if n == 0:
                    ok = False

    if ok:
        total = count_images(OCT2017_DIR)
        _ok(f"OCT2017 total: {total:,} images ✓")
    return ok


def verify_octid() -> bool:
    """Verify OCTID directory structure and report image counts."""
    _banner("Verifying OCTID")
    ok = True
    found_any = False

    for cls in REQUIRED_OCTID:
        cls_dir = OCTID_DIR / cls
        if cls_dir.exists():
            n = count_images(cls_dir)
            _ok(f"OCTID/{cls}: {n:,} images")
            found_any = True
        else:
            _warn(f"OCTID/{cls}: not found (may be named differently in your zip)")

    # Also check for any other class folders
    if OCTID_DIR.exists():
        extra = [d.name for d in OCTID_DIR.iterdir() if d.is_dir() and d.name not in REQUIRED_OCTID]
        if extra:
            _info(f"Additional OCTID folders found: {extra}")
            _info("  These will be mapped via OCTIDDataset class mapping in dataset.py")
            found_any = True

    if not found_any:
        _err(f"OCTID directory empty or missing at {OCTID_DIR}")
        ok = False

    return ok


def verify_octmnist() -> bool:
    """Check if OCTMNIST files are present."""
    _banner("Verifying OCTMNIST")
    npz_files = list(MEDMNIST_DIR.rglob("*.npz"))
    if npz_files:
        _ok(f"OCTMNIST: {len(npz_files)} file(s) at {MEDMNIST_DIR}")
        return True
    _warn(f"OCTMNIST not found at {MEDMNIST_DIR}")
    return False


def print_summary():
    """Print a final summary table of all dataset statuses."""
    _banner("Dataset Summary", char="═")

    rows = [
        ("OCT2017 (Kermany)",  OCT2017_DIR,   "~84K train images  → data/oct2017/"),
        ("OCTID",              OCTID_DIR,      "~500 images (OOD)  → data/OCTID/"),
        ("OCTMNIST",           MEDMNIST_DIR,   "~109K images       → data/medmnist/"),
    ]

    print(f"  {'Dataset':<22} {'Status':<12} {'Path'}")
    print(f"  {'─'*22} {'─'*12} {'─'*30}")
    for name, path, note in rows:
        n = count_images(path) if path.exists() else 0
        status = "✅ Ready" if n > 0 else "❌ Missing"
        print(f"  {name:<22} {status:<12} {path}  ({note})")

    print()
    _info("You are now ready to run:")
    _info("  uv run scripts/train.py --config configs/smoketest.yaml")
    _info("  uv run scripts/run_ablations.py")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download TinyOCT datasets from Google Drive to a RunPod server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--folder-id",
        default=DEFAULT_FOLDER_ID,
        help=f"Google Drive folder ID (default: {DEFAULT_FOLDER_ID})",
    )
    p.add_argument(
        "--download-dir",
        type=Path,
        default=DATA_DIR / "_gdrive_downloads",
        help="Temporary directory for raw downloads before extraction (default: data/_gdrive_downloads)",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Google Drive download; only extract from already-downloaded zips in --download-dir",
    )
    p.add_argument(
        "--skip-octmnist",
        action="store_true",
        help="Skip OCTMNIST auto-download (e.g. already present)",
    )
    p.add_argument(
        "--octmnist-only",
        action="store_true",
        help="Only download OCTMNIST, skip Google Drive datasets",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing datasets without downloading anything",
    )
    p.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep zip files after extraction (useful to avoid re-downloading)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    _banner("TinyOCT — Google Drive Dataset Downloader", char="═", width=60)
    print(f"  Project root: {Path.cwd()}")
    print(f"  Data root:    {DATA_DIR.resolve()}")
    print(f"  Drive folder: {args.folder_id}")

    # ── Verify-only mode ─────────────────────────────────────────────────────
    if args.verify_only:
        _banner("Verify-Only Mode")
        verify_oct2017()
        verify_octid()
        verify_octmnist()
        print_summary()
        return

    # ── OCTMNIST-only mode ───────────────────────────────────────────────────
    if args.octmnist_only:
        download_octmnist()
        print_summary()
        return

    # ── Google Drive download ─────────────────────────────────────────────────
    download_root = args.download_dir

    if not args.skip_download:
        gdown = ensure_gdown()
        _banner("Step 1 — Downloading from Google Drive")
        try:
            download_gdrive_folder(gdown, args.folder_id, download_root)
        except Exception:
            sys.exit(1)
    else:
        _banner("Step 1 — Skipping download (--skip-download)")
        if not download_root.exists():
            _err(f"Download directory not found: {download_root}")
            _err("Run without --skip-download first.")
            sys.exit(1)
        _info(f"Using existing downloads in: {download_root}")

    # ── Extraction ────────────────────────────────────────────────────────────
    _banner("Step 2 — Extracting Zip Archives")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extraction_ok = True

    for dataset_key, cfg in ZIP_MAPPING.items():
        _info(f"\nProcessing: {cfg['label']}")

        zip_path = find_zip_in_dir(download_root, cfg["candidates"])
        if zip_path is None:
            _err(f"Could not find zip for {cfg['label']} in {download_root}")
            _info(f"  Looked for: {cfg['candidates']}")
            _info("  Download the zip manually and place it in the download directory.")
            extraction_ok = False
            continue

        _ok(f"Found: {zip_path}")
        success = extract_zip(zip_path, cfg["target_dir"], dataset_key)
        if not success:
            extraction_ok = False

        # Clean up zip if requested
        if success and not args.keep_zips:
            _info(f"Removing zip to save disk space: {zip_path.name}")
            zip_path.unlink(missing_ok=True)
        elif args.keep_zips:
            _info(f"Keeping zip at: {zip_path}")

    if not extraction_ok:
        _warn("Some extractions failed. Check the errors above.")

    # ── OCTMNIST ──────────────────────────────────────────────────────────────
    if not args.skip_octmnist:
        _banner("Step 3 — OCTMNIST Auto-Download")
        download_octmnist()
    else:
        _info("Skipping OCTMNIST (--skip-octmnist)")

    # ── Verification ──────────────────────────────────────────────────────────
    _banner("Step 4 — Final Verification")
    oct2017_ok = verify_oct2017()
    octid_ok   = verify_octid()
    octmnist_ok = verify_octmnist() if not args.skip_octmnist else True

    # Clean up empty download dir
    if download_root.exists() and not any(download_root.rglob("*")):
        download_root.rmdir()

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary()

    if not (oct2017_ok and octid_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
