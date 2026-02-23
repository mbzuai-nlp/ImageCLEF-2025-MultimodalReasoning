"""
args.py — CLI argument parser for the image deduplication tool.
"""

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate examination question images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be removed (no changes made):
  python deduplicate.py --input images/ --dry-run

  # Remove duplicates in-place with default similarity threshold:
  python deduplicate.py --input images/

  # Copy unique images to a separate folder; move duplicates to removed/:
  python deduplicate.py --input images/ --output deduplicated/ --move

  # Strict similarity (only near-identical scans removed, threshold=5):
  python deduplicate.py --input images/ --threshold 5

  # Aggressive deduplication (broader similarity match, threshold=20):
  python deduplicate.py --input images/ --threshold 20
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Folder containing images to deduplicate.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=(
            "If given, copy unique images here instead of deleting in-place. "
            "Combined with --move, duplicates are also moved to output/removed/."
        ),
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=25,
        help=(
            "Perceptual hash (pHash) Hamming distance threshold for 'similar' images. "
            "Range 0–256. Lower = stricter. Recommended: 5 (near-identical) to 25 "
            "(rotated/re-scanned). Default: 25."
        ),
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.85,
        help=(
            "SSIM score threshold for crop detection (0–1). "
            "Higher = stricter. Default: 0.85."
        ),
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move duplicates to output/removed/ instead of deleting them (requires --output).",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be removed without making any changes.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=os.cpu_count(),
        help=(
            f"Number of parallel worker threads (default: {os.cpu_count()} — all CPU cores). "
            "Controls parallelism for hashing, pHash computation, and template matching."
        ),
    )
    return parser
