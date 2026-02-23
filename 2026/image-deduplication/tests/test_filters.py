"""
test_filters.py — Tests for each deduplication strategy in filters.py.
"""

import shutil
from pathlib import Path

import pytest

from filters import (
    find_cropped_duplicates,
    find_exact_duplicates,
    find_similar_images,
    is_crop_of,
    select_files_to_remove,
)
from utils import file_hash, handle_output, load_gray

# Default threshold used across tests — must match the CLI default in args.py
DEFAULT_THRESHOLD = 25


# ---------------------------------------------------------------------------
# Shared helper: compute to_remove from a list of work images
# ---------------------------------------------------------------------------

def _run_all_filters(images: list[Path]) -> set[Path]:
    exact_groups = find_exact_duplicates(images)
    all_hashes = {p: file_hash(p) for p in images}
    crop_pairs = find_cropped_duplicates(images, exact_duplicate_hashes=all_hashes)
    similar_pairs = find_similar_images(images, threshold=DEFAULT_THRESHOLD)
    return select_files_to_remove(exact_groups, crop_pairs, similar_pairs)


@pytest.fixture(scope="session")
def to_remove(all_images: list[Path]) -> set[Path]:
    """Redundant files identified from example_questions/ at the default threshold."""
    return _run_all_filters(all_images)


@pytest.fixture()
def work_images(all_images: list[Path], tmp_path: Path) -> list[Path]:
    """Copy of all_images in a temp dir so tests can safely mutate files."""
    work_dir = tmp_path / "images"
    work_dir.mkdir()
    for p in all_images:
        shutil.copy2(p, work_dir / p.name)
    return sorted(work_dir.iterdir())


# ---------------------------------------------------------------------------
# Strategy 1 — Exact duplicate detection
# ---------------------------------------------------------------------------

class TestFindExactDuplicates:
    def test_detects_identical_pair(self, all_images: list[Path]) -> None:
        """q1.jpeg and q6.jpeg are byte-identical and must appear in the same group."""
        groups = find_exact_duplicates(all_images)
        assert len(groups) == 1
        names = {p.name for p in next(iter(groups.values()))}
        assert {"q1.jpeg", "q6.jpeg"}.issubset(names)

    def test_non_duplicate_images_excluded(self, all_images: list[Path]) -> None:
        groups = find_exact_duplicates(all_images)
        grouped = {p.name for paths in groups.values() for p in paths}
        assert {"q2.jpeg", "q3.jpeg", "q5.jpeg", "q8.jpeg", "q9.jpeg"}.isdisjoint(grouped)

    def test_no_duplicates_returns_empty(self, img_original: Path, img_different: Path) -> None:
        assert find_exact_duplicates([img_original, img_different]) == {}

    def test_all_unique_images_no_groups(self, example_dir: Path) -> None:
        images = [example_dir / name for name in ("q2.jpeg", "q3.jpeg", "q8.jpeg")]
        assert find_exact_duplicates(images) == {}


# ---------------------------------------------------------------------------
# Strategy 2 — Crop / sub-image detection
# ---------------------------------------------------------------------------

class TestIsCropOf:
    def test_crop_is_detected(self, img_crop: Path, img_exact_dup: Path) -> None:
        """q5 is a crop of q6 (the kept original)."""
        assert is_crop_of(load_gray(img_crop), load_gray(img_exact_dup))

    def test_larger_is_not_crop_of_smaller(self, img_crop: Path, img_exact_dup: Path) -> None:
        assert not is_crop_of(load_gray(img_exact_dup), load_gray(img_crop))

    def test_identical_images_same_shape(self, img_original: Path, img_exact_dup: Path) -> None:
        """Same-size images have equal shapes, so neither is a strict crop of the other."""
        a, b = load_gray(img_original), load_gray(img_exact_dup)
        assert a.shape == b.shape

    def test_unrelated_images_not_crop(self, img_crop: Path, img_different: Path) -> None:
        crop, different = load_gray(img_crop), load_gray(img_different)
        if crop.shape[0] <= different.shape[0] and crop.shape[1] <= different.shape[1]:
            assert not is_crop_of(crop, different)


class TestFindCroppedDuplicates:
    def test_detects_crop(self, all_images: list[Path]) -> None:
        pairs = find_cropped_duplicates(all_images)
        assert "q5.jpeg" in {p.name for p, _ in pairs}

    def test_source_is_kept(self, all_images: list[Path]) -> None:
        for crop, source in find_cropped_duplicates(all_images):
            assert crop.name == "q5.jpeg"
            assert source.name in {"q1.jpeg", "q6.jpeg"}

    def test_skips_exact_duplicates(self, img_original: Path, img_exact_dup: Path) -> None:
        hashes = {img_original: file_hash(img_original), img_exact_dup: file_hash(img_exact_dup)}
        assert find_cropped_duplicates([img_original, img_exact_dup], exact_duplicate_hashes=hashes) == []

    def test_no_false_positives_between_different_questions(self, img_original: Path, img_different: Path) -> None:
        assert find_cropped_duplicates([img_original, img_different]) == []


# ---------------------------------------------------------------------------
# Strategy 3 — Perceptual similarity (rotation-aware)
# ---------------------------------------------------------------------------

class TestFindSimilarImages:
    def test_exact_duplicates_have_zero_distance(self, img_original: Path, img_exact_dup: Path) -> None:
        similar = find_similar_images([img_original, img_exact_dup], threshold=DEFAULT_THRESHOLD)
        assert len(similar) == 1 and similar[0][2] == 0

    def test_detects_rotated_duplicate(self, img_different: Path, img_rotation: Path) -> None:
        """q8 is q2 rotated — must be detected as similar within the default threshold."""
        similar = find_similar_images([img_different, img_rotation], threshold=DEFAULT_THRESHOLD)
        assert len(similar) == 1
        names = {similar[0][0].name, similar[0][1].name}
        assert names == {"q2.jpeg", "q8.jpeg"}

    def test_different_questions_not_similar(self, img_original: Path, img_different: Path) -> None:
        """q1 (a question) and q2 (a different question) should not be similar."""
        assert find_similar_images([img_original, img_different], threshold=DEFAULT_THRESHOLD) == []

    def test_threshold_zero_only_perceptually_identical(self, all_images: list[Path]) -> None:
        """At threshold=0 only byte-identical images (q1/q6, distance=0) should match."""
        similar = find_similar_images(all_images, threshold=0)
        assert all(dist == 0 for _, _, dist in similar)

    def test_high_threshold_finds_more_pairs(self, all_images: list[Path]) -> None:
        assert len(find_similar_images(all_images, threshold=50)) >= len(find_similar_images(all_images, threshold=5))

    def test_returns_3_tuple(self, img_original: Path, img_exact_dup: Path) -> None:
        similar = find_similar_images([img_original, img_exact_dup], threshold=DEFAULT_THRESHOLD)
        assert len(similar[0]) == 3


# ---------------------------------------------------------------------------
# Decision engine — select_files_to_remove
# ---------------------------------------------------------------------------

class TestSelectFilesToRemove:
    def test_removes_exact_duplicate_keeps_largest(self, img_original: Path, img_exact_dup: Path) -> None:
        """q1 and q6 are the same size; the kept file must have the larger (size, name)."""
        groups = find_exact_duplicates([img_original, img_exact_dup])
        to_remove = select_files_to_remove(groups, [], [])
        assert len(to_remove) == 1
        kept = ({img_original, img_exact_dup} - to_remove).pop()
        removed = to_remove.pop()
        assert kept.stat().st_size >= removed.stat().st_size

    def test_removes_crop(self, img_crop: Path, img_exact_dup: Path) -> None:
        to_remove = select_files_to_remove({}, [(img_crop, img_exact_dup)], [])
        assert img_crop in to_remove and img_exact_dup not in to_remove

    def test_removes_rotation_keeps_largest(self, img_different: Path, img_rotation: Path) -> None:
        """q8 (rotation of q2) should be removed; q2 is larger so it's kept."""
        similar = find_similar_images([img_different, img_rotation], threshold=DEFAULT_THRESHOLD)
        to_remove = select_files_to_remove({}, [], similar)
        assert img_rotation in to_remove
        assert img_different not in to_remove

    def test_integration_full_example(self, all_images: list[Path], to_remove: set[Path]) -> None:
        """End-to-end: exactly 4 redundant files from 7 example images."""
        assert len(to_remove) == 4
        assert len([p for p in all_images if p not in to_remove]) == 3
        removed_names = {p.name for p in to_remove}
        assert "q5.jpeg" in removed_names                       # crop always removed
        assert removed_names & {"q1.jpeg", "q6.jpeg"}          # one of the exact-dup pair removed
        assert "q8.jpeg" in removed_names                       # rotation removed
        assert "q9.jpeg" in removed_names                       # rotation removed

    def test_no_duplicates_returns_empty_set(self) -> None:
        assert select_files_to_remove({}, [], []) == set()


# ---------------------------------------------------------------------------
# Output handler
# ---------------------------------------------------------------------------

class TestHandleOutput:
    def test_dry_run_makes_no_changes(self, all_images: list[Path], to_remove: set[Path]) -> None:
        handle_output(to_remove, output_dir=None, move=False, dry_run=True)
        assert all(p.exists() for p in all_images)

    def test_output_dir_copies_keepers(self, work_images: list[Path], tmp_path: Path) -> None:
        to_remove = _run_all_filters(work_images)
        out_dir = tmp_path / "deduped"
        handle_output(to_remove, output_dir=out_dir, move=False, dry_run=False)
        assert {p.name for p in work_images if p not in to_remove} == {p.name for p in out_dir.iterdir()}

    def test_move_relocates_duplicates(self, work_images: list[Path], tmp_path: Path) -> None:
        to_remove = _run_all_filters(work_images)
        out_dir = tmp_path / "deduped"
        handle_output(to_remove, output_dir=out_dir, move=True, dry_run=False)
        removed_dir = out_dir / "removed"
        assert removed_dir.is_dir()
        assert {p.name for p in removed_dir.iterdir()} == {p.name for p in to_remove}
