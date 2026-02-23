"""
filters.py — Deduplication strategies and decision engine.

Strategies
----------
1. Exact duplicate detection   : MD5 hash comparison (byte-identical files)
2. Crop / sub-image detection  : Template matching + SSIM
3. Perceptual similarity       : pHash Hamming distance with user-defined threshold

All strategies support parallel execution via the `workers` argument.
Large-scale notes
-----------------
• Hashing and pHash computation are parallelised with ThreadPoolExecutor (I/O-bound).
• Pairwise pHash comparisons use scipy.spatial.distance.pdist (vectorised C), which
  handles tens of thousands of images in seconds.
• Crop detection pre-filters pairs by image dimensions (cheap PIL header read) before
  running expensive template matching in parallel threads (cv2 releases the GIL).
"""

import itertools
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist as sp_cdist
from skimage.metrics import structural_similarity as ssim

from utils import compute_phash, file_hash, get_image_size, load_gray

log = logging.getLogger(__name__)

_DEFAULT_WORKERS = os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Strategy 1 — Exact duplicate detection (parallel file hashing)
# ---------------------------------------------------------------------------

def find_exact_duplicates(
    images: list[Path],
    workers: int = _DEFAULT_WORKERS,
) -> dict[str, list[Path]]:
    """
    Group images by their MD5 hash.
    Returns {hash: [path, ...]} for groups with more than one file.
    Hashing is parallelised across `workers` threads.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(file_hash, images))

    hash_map: dict[str, list[Path]] = defaultdict(list)
    for path, h in zip(images, hashes):
        hash_map[h].append(path)

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    log.info(
        "Exact-duplicate groups found: %d  (total redundant files: %d)",
        len(duplicates),
        sum(len(v) - 1 for v in duplicates.values()),
    )
    return duplicates


# ---------------------------------------------------------------------------
# Strategy 2 — Crop / sub-image detection (parallel template matching)
# ---------------------------------------------------------------------------

def is_crop_of(small: np.ndarray, large: np.ndarray, ssim_threshold: float = 0.85) -> bool:
    """
    Return True if *small* appears to be a cropped region of *large*.

    Uses normalised cross-correlation template matching to locate the best
    matching patch, then verifies it with SSIM.
    """
    sh, sw = small.shape
    lh, lw = large.shape
    if sh > lh or sw > lw:
        return False

    small_u8 = (small * 255).astype(np.uint8)
    large_u8 = (large * 255).astype(np.uint8)
    _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(large_u8, small_u8, cv2.TM_CCOEFF_NORMED))

    if max_val < 0.70:  # quick reject — poor template match
        return False

    x, y = max_loc
    patch = large[y: y + sh, x: x + sw]
    return patch.shape == small.shape and ssim(small, patch, data_range=1.0) >= ssim_threshold


def find_cropped_duplicates(
    images: list[Path],
    ssim_threshold: float = 0.85,
    exact_duplicate_hashes: Optional[dict[Path, str]] = None,
    workers: int = _DEFAULT_WORKERS,
) -> list[tuple[Path, Path]]:
    """
    For every pair (a, b) check whether one is a strict crop of the other.
    Exact-duplicate pairs are skipped (template matching is meaningless for identical files).
    Returns a list of (crop_path, source_path) tuples.

    Large-scale strategy
    --------------------
    1. Read image dimensions in parallel using PIL (no pixel decoding).
    2. Pre-filter to only size-compatible pairs (small fits inside large).
    3. Run template matching on candidate pairs in parallel threads.
    """
    # Phase 1: Parallel dimension fetch (cheap PIL header read)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        shapes = list(pool.map(get_image_size, images))
    img_shapes = dict(zip(images, shapes))

    # Phase 2: Pre-filter pairs by size compatibility, skipping exact duplicates
    candidates: list[tuple[Path, Path]] = []   # (crop_candidate, source_candidate)
    for a, b in itertools.combinations(images, 2):
        if exact_duplicate_hashes:
            ha, hb = exact_duplicate_hashes.get(a), exact_duplicate_hashes.get(b)
            if ha and hb and ha == hb:
                log.debug("  Skipping crop check for exact-duplicate pair: '%s' / '%s'", a.name, b.name)
                continue
        sa, sb = img_shapes[a], img_shapes[b]
        if sa[0] <= sb[0] and sa[1] <= sb[1] and sa != sb:
            candidates.append((a, b))
        elif sb[0] <= sa[0] and sb[1] <= sa[1] and sa != sb:
            candidates.append((b, a))

    log.info("Crop candidate pairs after size pre-filter: %d", len(candidates))

    # Phase 3: Parallel template matching with a shared, thread-safe gray-image cache.
    # dict.setdefault is atomic in CPython; multiple threads may compute load_gray(p)
    # simultaneously for the same p, but all will store/return the same array.
    gray_cache: dict[Path, np.ndarray] = {}

    def get_gray(p: Path) -> np.ndarray:
        arr = gray_cache.get(p)
        if arr is None:
            arr = load_gray(p)
            gray_cache.setdefault(p, arr)
        return gray_cache[p]

    def check_pair(pair: tuple[Path, Path]) -> Optional[tuple[Path, Path]]:
        crop_path, src_path = pair
        return pair if is_crop_of(get_gray(crop_path), get_gray(src_path), ssim_threshold) else None

    crop_pairs: list[tuple[Path, Path]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(check_pair, candidates):
            if result:
                log.info("  Crop detected: '%s'  ⊂  '%s'", result[0].name, result[1].name)
                crop_pairs.append(result)

    log.info("Crop pairs found: %d", len(crop_pairs))
    return crop_pairs


# ---------------------------------------------------------------------------
# Strategy 3 — Perceptual similarity (parallel pHash + rotation-aware comparison)
# ---------------------------------------------------------------------------

def _phash_all_rotations(path: Path) -> list:
    """Return pHashes for 0°, 90°, 180°, 270° rotations of an image."""
    with Image.open(path) as img:
        return [imagehash.phash(img.rotate(angle), hash_size=16) for angle in (0, 90, 180, 270)]


def find_similar_images(
    images: list[Path],
    threshold: int = 10,
    workers: int = _DEFAULT_WORKERS,
) -> list[tuple[Path, Path, int]]:
    """
    Compare every pair by their pHash Hamming distance, checking all 90° rotations.
    Returns (img_a, img_b, distance) for pairs whose minimum rotational distance <= threshold.

    Rotation-aware strategy
    -----------------------
    Each image is hashed at 0°, 90°, 180°, and 270°.  For each pair (i, j) the
    distance is ``min over r: hamming(hash_i[0°], hash_j[r°])``, so rotated
    copies are detected at the same threshold as standard near-duplicates.

    Typical threshold guidance:
      ≤ 5   → near-identical (minor compression / resize artefacts)
      6–25  → visually similar or rotated (same question, different scan)
      26+   → different images
    """
    # Phase 1: Compute rotational pHashes in parallel
    with ThreadPoolExecutor(max_workers=workers) as pool:
        all_rotations = list(pool.map(_phash_all_rotations, images))

    n = len(images)
    n_bits = all_rotations[0][0].hash.size  # 256 for hash_size=16

    # base[i]: 0° hash for image i (reference orientation)
    base = np.array([rots[0].hash.flatten().astype(np.uint8) for rots in all_rotations])

    # Upper-triangle indices for all pairs (i < j)
    ii, jj = np.triu_indices(n, k=1)

    # For each rotation of image j, compute pairwise distances against base of image i.
    # Keep element-wise minimum so any matching rotation qualifies.
    min_dists = np.full(len(ii), np.inf)
    for rot_idx in range(4):
        rot_j = np.array([rots[rot_idx].hash.flatten().astype(np.uint8) for rots in all_rotations])
        dists = sp_cdist(base, rot_j, metric="hamming")[ii, jj] * n_bits
        np.minimum(min_dists, dists, out=min_dists)

    mask = min_dists <= threshold
    similar = [
        (images[int(i)], images[int(j)], int(round(d)))
        for i, j, d in zip(ii[mask], jj[mask], min_dists[mask])
    ]
    for a, b, dist in similar:
        log.info("  Similar pair (distance=%d): '%s'  ~  '%s'", dist, a.name, b.name)
    log.info("Similar pairs found (threshold ≤ %d): %d", threshold, len(similar))
    return similar


# ---------------------------------------------------------------------------
# Decision engine — select files to remove
# ---------------------------------------------------------------------------

def select_files_to_remove(
    exact_groups: dict[str, list[Path]],
    crop_pairs: list[tuple[Path, Path]],
    similar_pairs: list[tuple[Path, Path, int]],
) -> set[Path]:
    """
    Decide which files are redundant across all three strategies.

    Rules
    -----
    • Exact duplicates  → keep the largest file (best quality); remove the rest.
    • Cropped images    → remove the crop; keep the source.
    • Similar images    → within each connected cluster keep only the largest file.

    A union-find structure handles transitivity in similarity clusters.
    """
    def best(paths: list[Path]) -> Path:
        """Largest file by byte size; ties broken alphabetically."""
        return max(paths, key=lambda p: (p.stat().st_size, p.name))

    to_remove: set[Path] = set()

    for paths in exact_groups.values():
        to_remove.update(p for p in paths if p != best(paths))

    for crop, _ in crop_pairs:
        to_remove.add(crop)

    # Union-find over similar pairs, keeping the largest file per cluster
    parent: dict[Path, Path] = {}

    def find(x: Path) -> Path:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:  # path compression
            parent[x], x = root, parent.get(x, x)
        return root

    def union(x: Path, y: Path) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry if best([rx, ry]) == rx else rx] = best([rx, ry])

    all_similar: set[Path] = set()
    for a, b, _ in similar_pairs:
        all_similar.update([a, b])
        union(a, b)

    clusters: dict[Path, list[Path]] = defaultdict(list)
    for p in all_similar:
        clusters[find(p)].append(p)

    for rep, members in clusters.items():
        to_remove.update(m for m in members if m != rep)

    return to_remove
