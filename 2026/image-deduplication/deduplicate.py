#!/usr/bin/env python3
"""
deduplicate.py — Entrypoint for the image deduplication tool.

Strategies
----------
1. Exact duplicate detection  : MD5 hash comparison (byte-identical files)
2. Crop / sub-image detection : Template matching + SSIM
3. Perceptual similarity      : pHash Hamming distance with user-defined threshold

Usage:
    python deduplicate.py --input images/ [--output deduplicated/] [--threshold 10] [--dry-run] [--move]

Requirements:
    pip install Pillow imagehash scikit-image opencv-python numpy
"""

import logging
import sys

from args import build_parser
from filters import find_cropped_duplicates, find_exact_duplicates, find_similar_images, select_files_to_remove
from utils import collect_images, file_hash, handle_output

log = logging.getLogger(__name__)

SEP = "=" * 60


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.move and not args.output:
        parser.error("--move requires --output to be specified.")
    if not args.input.is_dir():
        parser.error(f"Input path is not a directory: {args.input}")

    images = collect_images(args.input)
    if len(images) < 2:
        log.info("Need at least 2 images to compare. Exiting.")
        sys.exit(0)

    log.info("%s\nSTRATEGY 1: Exact duplicate detection (MD5 hash)\n%s", SEP, SEP)
    exact_groups = find_exact_duplicates(images, workers=args.workers)

    log.info("%s\nSTRATEGY 2: Cropped / sub-image detection (template matching + SSIM)\n%s", SEP, SEP)
    all_hashes = {p: file_hash(p) for p in images}
    crop_pairs = find_cropped_duplicates(images, ssim_threshold=args.ssim_threshold, exact_duplicate_hashes=all_hashes, workers=args.workers)

    log.info("%s\nSTRATEGY 3: Perceptual similarity (pHash, threshold=%d)\n%s", SEP, args.threshold, SEP)
    similar_pairs = find_similar_images(images, threshold=args.threshold, workers=args.workers)

    log.info("%s\nDECISION: selecting files to remove\n%s", SEP, SEP)
    to_remove = select_files_to_remove(exact_groups, crop_pairs, similar_pairs)

    if to_remove:
        log.info("Files marked for removal (%d):", len(to_remove))
        for p in sorted(to_remove):
            log.info("  ✗  %s", p.name)
        kept = [p for p in images if p not in to_remove]
        log.info("Files kept (%d):", len(kept))
        for p in sorted(kept):
            log.info("  ✓  %s", p.name)
    else:
        log.info("No duplicates found.")

    log.info("%s\nOUTPUT\n%s", SEP, SEP)
    handle_output(to_remove, args.output, args.move, args.dry_run)
    log.info("Done.")


if __name__ == "__main__":
    main()
