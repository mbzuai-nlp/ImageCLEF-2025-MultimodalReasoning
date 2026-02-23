"""
utils.py — Shared constants, logging configuration, I/O helpers, and output handler.
"""

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    """Return all image paths inside *folder* (non-recursive)."""
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    log.info("Found %d image(s) in '%s'.", len(paths), folder)
    return paths


def file_hash(path: Path, algorithm: str = "md5") -> str:
    """Compute hex digest of a file's raw bytes."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_image_size(path: Path) -> tuple[int, int]:
    """Return (height, width) via PIL without decoding pixel data."""
    with Image.open(path) as img:
        w, h = img.size
        return h, w


def load_gray(path: Path) -> np.ndarray:
    """Load image as a float32 grayscale array (values 0–1)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img.astype(np.float32) / 255.0


def compute_phash(path: Path) -> imagehash.ImageHash:
    """Compute perceptual hash (pHash) of an image."""
    with Image.open(path) as img:
        return imagehash.phash(img, hash_size=16)  # 256-bit hash


def handle_output(
    to_remove: set[Path],
    output_dir: Optional[Path],
    move: bool,
    dry_run: bool,
) -> None:
    """
    Depending on flags:
      • dry_run            → only print what would happen
      • output_dir + move  → move duplicates to output_dir/removed/, copy keepers
      • output_dir only    → copy keepers to output_dir
      • neither            → delete duplicates in-place
    """
    if dry_run:
        log.info("--- DRY RUN: no files will be modified ---")
        if to_remove:
            log.info("Files that WOULD be removed (%d):", len(to_remove))
            for p in sorted(to_remove):
                log.info("  ✗  %s", p.name)
        else:
            log.info("No duplicates found.")
        return

    if not to_remove:
        log.info("No duplicates found — nothing to do.")
        return

    removed_dir = output_dir / "removed" if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if removed_dir and move:
        removed_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(to_remove):
        if removed_dir and move:
            shutil.move(str(p), removed_dir / p.name)
            log.info("  MOVED   %s  →  %s", p.name, removed_dir / p.name)
        elif output_dir:
            log.info("  SKIP (dup)  %s", p.name)
        else:
            p.unlink()
            log.info("  DELETED  %s", p.name)

    if output_dir:
        input_dir = next(iter(to_remove)).parent
        for p in collect_images(input_dir):
            if p not in to_remove:
                shutil.copy2(str(p), output_dir / p.name)
                log.info("  KEPT    %s  →  %s", p.name, output_dir / p.name)
