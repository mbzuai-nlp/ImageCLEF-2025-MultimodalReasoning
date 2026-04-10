import json
import re
from pathlib import Path
from typing import Any


QUESTION_EXTRACTION_PROMPT = "\n".join(
    (
        "Extract the full examination question text from the image.",
        "Return only the question text in the original language.",
        "Do not answer the question.",
        "Do not add labels such as 'Question:' or any extra commentary.",
        "Include short instructions or answer options only when they are visible and needed to preserve the question faithfully.",
    )
)


def normalize_item_id(item_id: Any) -> str:
    return str(item_id).strip()


def sanitize_extracted_question(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").strip()
    cleaned = re.sub(
        r"^\s*(question|question text|extracted question)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip().strip("\"'")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_answer_prompt(extracted_question: str) -> str:
    question_text = sanitize_extracted_question(extracted_question)
    if not question_text:
        raise ValueError("Extracted question text is empty.")

    return "\n".join(
        (
            "The image contains an examination question.",
            "Use the extracted question below as OCR guidance while answering.",
            "If the image contains details that the extracted text missed, rely on the image.",
            "",
            "Extracted question:",
            question_text,
            "",
            "Provide only the final answer, without any explanation or reasoning steps.",
        )
    )


def load_extracted_question_map(path: str | Path) -> dict[tuple[str, str], str]:
    question_path = Path(path)
    question_map: dict[tuple[str, str], str] = {}

    with question_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            question_map[(record["split"], normalize_item_id(record["id"]))] = sanitize_extracted_question(
                record["question"]
            )

    if not question_map:
        raise ValueError(f"No extracted questions were loaded from {question_path}.")

    return question_map


def get_extracted_question(
    question_map: dict[tuple[str, str], str],
    item_id: Any,
    split: str,
) -> str:
    return question_map[(split, normalize_item_id(item_id))]
