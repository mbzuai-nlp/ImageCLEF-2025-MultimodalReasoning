# Image Deduplication

Removes exact duplicates, cropped sub-images, and visually similar images from a folder of examination question images.

## How It Works

Three strategies run in sequence, each catching a different class of redundancy:

| # | Strategy | Method | Catches |
|---|---|---|---|
| 1 | **Exact duplicate** | MD5 file hash | Byte-identical copies |
| 2 | **Crop detection** | OpenCV template matching + SSIM | Images that are a cropped region of another |
| 3 | **Perceptual similarity** | Rotation-aware pHash (0°/90°/180°/270°) + Hamming distance | Near-identical or rotated scans of the same question |

When duplicates are found, the **largest file** in each group is kept as the highest-quality representative.

### Similarity Threshold (Strategy 3)

The `--threshold` flag controls how aggressively similar images are matched:

| Threshold | Behaviour |
|---|---|
| ≤ 5 | Near-identical (compression/resize artefacts only) |
| 6–25 | Same question, rotated or different scan quality *(default: 25)* |
| 26+ | Different images |

---

## Performance & Scalability

All three strategies are designed to scale to tens of thousands of images:

| Strategy | Parallelism | Complexity |
|---|---|---|
| Exact duplicate | Parallel MD5 hashing (`ThreadPoolExecutor`) | O(n) |
| Crop detection | Parallel PIL dimension fetch → size pre-filter → parallel template matching | O(k) where k ≪ n² |
| Perceptual similarity | Parallel pHash computation + vectorised `scipy.pdist` (C-level) | O(n²) at C speed |

Use `--workers` to control the thread count (defaults to all CPU cores).

---

## Setup

**Requires Python 3.12+** and [`uv`](https://github.com/astral-sh/uv).

```bash
# Create and activate a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Install with dev dependencies (includes pytest)
uv pip install -e ".[dev]"
```

---

## Usage

```bash
# Preview removals without making any changes (recommended first step)
python3 deduplicate.py --input images/ --dry-run

# Remove duplicates in-place
python3 deduplicate.py --input images/

# Copy unique images to a new folder; move duplicates to deduplicated/removed/
python3 deduplicate.py --input images/ --output deduplicated/ --move

# Stricter similarity (near-identical scans only)
python3 deduplicate.py --input images/ --threshold 5

# More aggressive similarity matching
python3 deduplicate.py --input images/ --threshold 20
```

### All Options

| Flag | Default | Description |
|---|---|---|
| `--input / -i` | *(required)* | Folder of images to process |
| `--output / -o` | — | Copy unique images here instead of deleting in-place |
| `--threshold / -t` | `25` | pHash Hamming distance threshold (0–256, lower = stricter) |
| `--ssim-threshold` | `0.85` | SSIM score for crop detection (0–1, higher = stricter) |
| `--move` | off | Move duplicates to `output/removed/` (requires `--output`) |
| `--dry-run / -n` | off | Preview changes without modifying any files |
| `--verbose / -v` | off | Enable debug logging |
| `--workers / -w` | all cores | Number of parallel worker threads |

---

## Testing

Install dev dependencies (includes pytest), then run the suite:

```bash
uv pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

The tests use the committed `example_questions/` images as fixtures and cover all three deduplication strategies, the decision engine, and the output handler (dry-run, copy, move).

---

## Project Structure

```
Image-dedup/
├── deduplicate.py   # Entrypoint — run this
├── filters.py       # Detection strategies & decision engine
├── utils.py         # I/O helpers, hashing, output handler
├── args.py          # CLI argument parser
└── images/          # Place your images here
```
