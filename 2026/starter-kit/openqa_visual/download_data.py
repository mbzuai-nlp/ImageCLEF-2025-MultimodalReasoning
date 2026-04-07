from datasets import load_dataset


DATASET_NAME = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"


def main():
    print(f"Downloading dataset {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
    except Exception as exc:
        print(f"Error downloading: {exc}")
        return

    print("Download complete.")
    print(f"Features: {dataset.features}")
    print("First example keys:")
    first_example = dataset[0]
    print(sorted(first_example.keys()))

    if "question" in first_example:
        print(f"Question: {first_example['question']}")
    if "answer" in first_example:
        print(f"Answer: {first_example['answer']}")
    if "image" in first_example:
        print(f"Image type: {type(first_example['image'])}")
        print(f"Image info: {first_example['image']}")


if __name__ == "__main__":
    main()
