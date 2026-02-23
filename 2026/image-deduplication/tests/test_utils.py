"""
test_utils.py — Tests for utility helpers in utils.py.
"""

from pathlib import Path

import numpy as np
import pytest

from utils import IMAGE_EXTENSIONS, collect_images, compute_phash, file_hash, load_gray


class TestCollectImages:
    def test_returns_all_images(self, example_dir: Path) -> None:
        images = collect_images(example_dir)
        assert len(images) == 7

    def test_returns_only_image_files(self, example_dir: Path) -> None:
        images = collect_images(example_dir)
        assert all(p.suffix.lower() in IMAGE_EXTENSIONS for p in images)

    def test_returns_sorted_paths(self, example_dir: Path) -> None:
        images = collect_images(example_dir)
        assert images == sorted(images)

    def test_empty_folder_returns_empty_list(self, tmp_path: Path) -> None:
        assert collect_images(tmp_path) == []

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "data.csv").write_text("a,b,c")
        assert collect_images(tmp_path) == []


class TestFileHash:
    def test_same_file_same_hash(self, img_original: Path) -> None:
        assert file_hash(img_original) == file_hash(img_original)

    def test_identical_files_same_hash(self, img_original: Path, img_exact_dup: Path) -> None:
        """q1.jpg and q3.jpg are byte-identical — must produce the same hash."""
        assert file_hash(img_original) == file_hash(img_exact_dup)

    def test_different_files_different_hash(self, img_original: Path, img_different: Path) -> None:
        assert file_hash(img_original) != file_hash(img_different)

    def test_hash_is_hex_string(self, img_original: Path) -> None:
        h = file_hash(img_original)
        assert isinstance(h, str)
        int(h, 16)                      # raises ValueError if not valid hex

    def test_sha256_algorithm(self, img_original: Path) -> None:
        h = file_hash(img_original, algorithm="sha256")
        assert len(h) == 64             # SHA-256 produces 64 hex chars


class TestLoadGray:
    def test_returns_float32_array(self, img_original: Path) -> None:
        arr = load_gray(img_original)
        assert arr.dtype == np.float32

    def test_values_in_unit_range(self, img_original: Path) -> None:
        arr = load_gray(img_original)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_returns_2d_array(self, img_original: Path) -> None:
        arr = load_gray(img_original)
        assert arr.ndim == 2

    def test_crop_smaller_than_source(self, img_original: Path, img_crop: Path) -> None:
        """The crop image must be strictly smaller than the source in at least one dimension."""
        source = load_gray(img_original)
        crop = load_gray(img_crop)
        assert crop.shape[0] <= source.shape[0] and crop.shape[1] <= source.shape[1]
        assert crop.shape != source.shape

    def test_invalid_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Cannot read image"):
            load_gray(tmp_path / "nonexistent.jpg")


class TestComputePhash:
    def test_returns_image_hash(self, img_original: Path) -> None:
        import imagehash
        result = compute_phash(img_original)
        assert isinstance(result, imagehash.ImageHash)

    def test_identical_files_zero_distance(self, img_original: Path, img_exact_dup: Path) -> None:
        """Byte-identical images must have Hamming distance 0."""
        assert (compute_phash(img_original) - compute_phash(img_exact_dup)) == 0

    def test_different_questions_high_distance(self, img_original: Path, img_different: Path) -> None:
        """Substantially different images should exceed a reasonable threshold."""
        dist = compute_phash(img_original) - compute_phash(img_different)
        assert dist > 10
