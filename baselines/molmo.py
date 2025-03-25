import os
import json
import re
import base64
from openai import OpenAI
import random
import concurrent.futures

# Initialize OpenAI-compatible client for vLLM
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8022/v1",  # vLLM vision server
)


def image_to_base64(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def process_image_chat_vllm(image_path):
    image_base64_url = image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze the image of a multiple-choice question. Identify the question, all answer options (even if there are more than four), and any relevant visuals like graphs or tables. Choose the correct answer based only on the image. Reply with just the letter of the correct option, no explanation.""",
                },
                {"type": "image_url", "image_url": {"url": image_base64_url}},
            ],
        },
    ]

    response = client.chat.completions.create(
        model="models/molmo",
        messages=messages,
        temperature=0,
        max_completion_tokens=50,
        extra_body={"guided_choice": ["A", "B", "C", "D", "E"]},
    )

    output = response.choices[0].message.content
    return output


def evaluate(dataset_path, image_folder):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    results = []
    correct = 0
    total = 0

    for item in metadata:
        image_id = item["id"]
        answer_key = item["answer_key"].upper()
        image_path = os.path.join(image_folder, f"{image_id}.png")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Processing {image_id}...")

        prediction = process_image_chat_vllm(image_path)
        is_correct = prediction == answer_key

        results.append({
            "id": image_id,
            "language": item["language"],
            "answer_key": prediction
        })

        correct += is_correct
        total += 1

        print(f"ID: {image_id}, Prediction: {prediction}, Actual: {answer_key}")

    accuracy = (correct / total) * 100 if total else 0
    print(f"\nFinal Accuracy: {accuracy:.2f}% on {total} images.")

    with open("molmo_vllm_results_short_prompt_1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return accuracy


if __name__ == "__main__":
    dataset_path = "data/metadata_labeled.json"
    image_folder = "data/images"
    evaluate(dataset_path, image_folder)
