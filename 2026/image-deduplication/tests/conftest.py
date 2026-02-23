"""
conftest.py — Shared pytest fixtures for the deduplication test suite.

Image inventory in example_questions/
--------------------------------------
q1.jpeg  — exam question scan               (1 447 131 bytes)  → exact duplicate of q6; removed
q2.jpeg  — different exam question          (1 110 018 bytes)  → kept (largest of q2/q8/q9 cluster)
q3.jpeg  — different exam question          (  878 044 bytes)  → kept
q5.jpeg  — cropped region of q1 / q6       (1 126 664 bytes)  → crop; removed
q6.jpeg  — byte-identical copy of q1.jpeg  (1 447 131 bytes)  → kept (largest name wins tie)
q8.jpeg  — q2 rotated 90°                  (  850 858 bytes)  → rotation; removed
q9.jpeg  — q2 rotated 180° or 270°         (  818 661 bytes)  → rotation; removed

Expected outcome: 4 removed (q1, q5, q8, q9), 3 kept (q2, q3, q6).
"""

from pathlib import Path

import pytest

from utils import collect_images

EXAMPLE_DIR = Path(__file__).parent.parent / "example_questions"


@pytest.fixture(scope="session")
def example_dir() -> Path:
    assert EXAMPLE_DIR.is_dir(), f"example_questions not found at {EXAMPLE_DIR}"
    return EXAMPLE_DIR


@pytest.fixture(scope="session")
def all_images(example_dir: Path) -> list[Path]:
    """All 7 image paths, sorted alphabetically."""
    return collect_images(example_dir)


@pytest.fixture(scope="session")
def img_original(example_dir: Path) -> Path:
    """One of the exact-duplicate pair (q1.jpeg)."""
    return example_dir / "q1.jpeg"


@pytest.fixture(scope="session")
def img_exact_dup(example_dir: Path) -> Path:
    """Byte-identical copy of q1.jpeg (q6.jpeg) — this one is kept."""
    return example_dir / "q6.jpeg"


@pytest.fixture(scope="session")
def img_different(example_dir: Path) -> Path:
    """A completely different question (q2.jpeg)."""
    return example_dir / "q2.jpeg"


@pytest.fixture(scope="session")
def img_crop(example_dir: Path) -> Path:
    """Cropped sub-image of q1 / q6 (q5.jpeg)."""
    return example_dir / "q5.jpeg"


@pytest.fixture(scope="session")
def img_rotation(example_dir: Path) -> Path:
    """q2 rotated 90° (q8.jpeg)."""
    return example_dir / "q8.jpeg"
