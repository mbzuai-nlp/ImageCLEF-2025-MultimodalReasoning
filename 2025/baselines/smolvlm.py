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
    base_url="http://localhost:8024/v1",  # vLLM vision server
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
                    "text": """You are a sophisticated Vision-Language Model (VLM) capable of analyzing images containing multiple-choice questions, regardless of language. To guide your analysis, you may adopt the following process: 
                    1. Examine the image carefully for all textual and visual information. 
                    2. Identify the question text, even if it's in a different language. 
                    3. Extract all answer options (note: there may be more than four). 
                    4. Look for additional visual elements such as tables, diagrams, charts, or graphs. 
                    5. Ensure to consider any multilingual content present in the image. 
                    6. Analyze the complete context and data provided. 
                    7. Select the correct answer(s) based solely on your analysis. 
                    8. Respond by outputting only the corresponding letter(s) without any extra explanation.""",
                },
                {"type": "image_url", "image_url": {"url": image_base64_url}},
            ],
        },
    ]

    response = client.chat.completions.create(
        model="models/smolvlm",
        messages=messages,
        temperature=0,
        max_completion_tokens=20,
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
        language = item["language"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Processing {image_id}...")

        prediction = process_image_chat_vllm(image_path)
        is_correct = prediction == answer_key

        results.append(
            {"id": image_id, "language": item["language"], "answer_key": prediction}
        )

        correct += is_correct
        total += 1

        print(f"ID: {image_id}, Prediction: {prediction}, Actual: {answer_key}")

    accuracy = (correct / total) * 100 if total else 0
    print(f"\nFinal Accuracy: {accuracy:.2f}% on {total} images.")

    with open("smolvlm_vllm_results_prompt_2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return accuracy


if __name__ == "__main__":
    dataset_path = "data/validation_data.json"
    image_folder = "data/images"
    evaluate(dataset_path, image_folder)
