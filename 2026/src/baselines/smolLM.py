import os
import json
import re
from openai import OpenAI

# Setup VLLM chat client
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8018/v1",  # SmolLM2 VLLM chat port
)

# Folder containing precomputed SmolVLM-style captions
CAPTION_FOLDER = "../captions/SmolVLM"
# CAPTION_FOLDER = "/home/momina.ahsan/Documents/ExamsV/Llama-3.2-11B-Vision"


def load_caption(image_id):
    caption_path = os.path.join(CAPTION_FOLDER, f"{image_id}.json")
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Caption not found for image: {image_id}")
    with open(caption_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    return caption_data.get("caption", "").strip()


def run_llm_on_caption(image_id):
    caption = load_caption(image_id)

    # -------- PROMPT 1 --------
    # prompt_text = (
    #     f"You are given a multiple-choice question extracted from an exam. "
    #     f"The question is:\n{caption}\n"
    #     f"Identify the question and all answer options (even if there are more than four), and any relevant data related to graphs or tables. Choose the correct answer and reply with just the letter of the correct option, no explanation."
    # )

    # -------- PROMPT 2 --------
    prompt_text = (
        f"You are given a multiple-choice question extracted from an exam.\n"
        f"The question is:\n{caption}\n"
        f"Please follow the steps below to determine the correct answer:\n"
        f"1. Carefully read and interpret the full question text.\n"
        f"2. Identify the main question, even if it is in a different language.\n"
        f"3. Extract all available answer options (note: there may be more than four).\n"
        f"4. Pay attention to any references to data, including tables, diagrams, charts, or graphs mentioned in the text.\n"
        f"5. Take into account any multilingual elements present in the question.\n"
        f"6. Analyze all information in context, both textual and inferred data.\n"
        f"7. Select the correct answer based solely on your analysis.\n"
        f"8. Respond by outputting only the letter(s) of the correct answer option, with no additional explanation."
    )

    messages = [
        {"role": "user", "content": prompt_text},
    ]

    response = client.chat.completions.create(
        model="../models/smollm",
        messages=messages,
        temperature=0,
        max_completion_tokens=500,
        extra_body={"guided_choice": ["A", "B", "C", "D", "E"]},
    )

    output = response.choices[0].message.content.strip()
    return output, caption


def evaluate(dataset_path, model_name):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    correct = 0
    total = 0
    results = []

    for item in metadata:
        image_id = item["id"]
        answer_key = item["answer_key"].upper()

        print(f"Processing {image_id}...")

        prediction, caption = run_llm_on_caption(image_id)
        is_correct = prediction == answer_key

        results.append(
            {"id": image_id, "language": item["language"], "answer_key": prediction}
        )

        correct += is_correct
        total += 1

        print(f"ID: {image_id}, Predicted: {prediction}, Actual: {answer_key}")

    accuracy = (correct / total) * 100 if total else 0
    print(f"Final Accuracy: {accuracy:.2f}% on {total} items")

    output_file = f"{model_name.split('/')[-1]}_smolvlm_caption_long.json"  # or prompt2
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return accuracy


if __name__ == "__main__":
    dataset_path = "../data/validation_data.json"
    model_name = "../models/smollm"

    print("Running SmolLM2")
    evaluate(dataset_path, model_name)
