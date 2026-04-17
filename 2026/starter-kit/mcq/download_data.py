from datasets import load_dataset
import json

def main():
    print("Downloading dataset MBZUAI/EXAMS-V...")
    # Using 'default' configuration if available or just loading properly
    try:
        dataset = load_dataset("MBZUAI/EXAMS-V", split="train")
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    print("Download complete.")
    print(f"Features: {dataset.features}")
    print("First example:")
    first_example = dataset[0]
    # Convert to string if there are non-serializable objects like images, just print keys
    print(first_example.keys())
    
    # Check image type
    if 'image' in first_example:
        print(f"Image type: {type(first_example['image'])}")
        print(f"Image info: {first_example['image']}")

if __name__ == "__main__":
    main()
