
import os
import csv
import json
import time
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIGURATION ----------------

TARGET_RECORDS = 5000
BATCH_SIZE = 25
MODEL = "grok-4.7"

OUTPUT_FILE = Path("systematic_similarity_search_dataset.csv")
CHECKPOINT_FILE = Path("systematic_similarity_search_checkpoint.csv")

COLUMNS = ["id", "title", "description", "keywords", "type"]

CATEGORIES = [
    "Technology",
    "Education",
    "Health",
    "Fitness",
    "Travel",
    "Food",
    "Finance",
    "Entertainment",
    "Environment",
    "Lifestyle",
    "Gaming",
    "Sports",
    "Career",
    "Relationships",
    "Science",
    "Fashion",
    "Parenting",
    "Business",
    "Photography",
    "Personal Development",
]

if not os.getenv("XAI_API_KEY"):
    raise RuntimeError(
        "XAI_API_KEY is missing. Set your Grok API key first."
    )

client = OpenAI(
    api_key=os.environ["XAI_API_KEY"],
    base_url="https://api.x.ai/v1",
    timeout=180,
    max_retries=3,
)

# ---------------- HELPERS ----------------

def normalize(text):
    """Normalize text for exact duplicate detection."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_existing():
    """Load saved records so interrupted runs can resume."""
    if not OUTPUT_FILE.exists():
        return []

    df = pd.read_csv(OUTPUT_FILE, dtype=str).fillna("")

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    records = df[COLUMNS].to_dict("records")

    # Check existing IDs and content before resuming.
    if len(records) > TARGET_RECORDS:
        raise ValueError(
            f"{OUTPUT_FILE} already has more than "
            f"{TARGET_RECORDS} rows. Review it first."
        )

    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs found in existing CSV.")

    return records


def save_records(records):
    """Write a consistent CSV and a checkpoint."""
    df = pd.DataFrame(records, columns=COLUMNS)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    df.to_csv(
        CHECKPOINT_FILE,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )


def generate_batch(batch_number, batch_size, existing_records):
    """Request a structured batch from Grok."""

    recent_titles = [
        r["title"] for r in existing_records[-100:]
    ]

    schema = {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "type": {
                            "type": "string",
                            "enum": CATEGORIES,
                        },
                    },
                    "required": [
                        "title",
                        "description",
                        "keywords",
                        "type",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["records"],
        "additionalProperties": False,
    }

    prompt = f"""
Generate exactly {batch_size} distinct, realistic, synthetic
social-media posts for a semantic similarity search project.

Batch number: {batch_number}

Every record must contain:
- title: a concise, specific post title
- description: a natural, informative post of 40-100 words
- keywords: 4-8 relevant, lowercase keyword phrases
- type: exactly one category from this list:
  {", ".join(CATEGORIES)}

Dataset quality requirements:
1. Cover a broad range of subjects and user intents.
2. Include questions, experiences, advice, opinions, tips,
   discussions, recommendations, and informational posts.
3. Vary sentence structures, writing styles, and topics.
4. Do not copy, quote, or imitate real people's identifiable posts.
5. Do not repeat titles or descriptions.
6. Make each record meaningfully different, not just a
   paraphrase of another record.
7. Return only the structured JSON requested by the schema.
8. Do not include record IDs; the Python program creates them.

Avoid repeating these recently used titles:
{json.dumps(recent_titles, ensure_ascii=False)}
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "social_media_dataset",
                "strict": True,
                "schema": schema,
            }
        },
    )

    data = json.loads(response.output_text)
    return data["records"]


def validate_record(record, existing_records):
    """Reject invalid records and exact duplicate content."""

    title = str(record.get("title", "")).strip()
    description = str(record.get("description", "")).strip()
    category = str(record.get("type", "")).strip()
    keywords = record.get("keywords", [])

    if not title or not description:
        return False

    if category not in CATEGORIES:
        return False

    if not isinstance(keywords, list):
        return False

    keywords = [
        str(k).strip().lower()
        for k in keywords
        if str(k).strip()
    ]

    if len(set(keywords)) < 4:
        return False

    if len(title) < 8 or len(description.split()) < 25:
        return False

    title_key = normalize(title)
    description_key = normalize(description)

    for existing in existing_records:
        if normalize(existing["title"]) == title_key:
            return False

        if normalize(existing["description"]) == description_key:
            return False

    return {
        "title": title,
        "description": description,
        "keywords": ", ".join(dict.fromkeys(keywords)),
        "type": category,
    }


# ---------------- MAIN GENERATION LOOP ----------------

def main():
    records = load_existing()

    print(f"Existing records: {len(records)}")
    print(f"Target records: {TARGET_RECORDS}")

    # Validate and resume existing content.
    used_titles = set()
    used_descriptions = set()

    for record in records:
        title_key = normalize(record["title"])
        description_key = normalize(record["description"])

        if title_key in used_titles:
            raise ValueError("Duplicate titles found in existing CSV.")

        if description_key in used_descriptions:
            raise ValueError(
                "Duplicate descriptions found in existing CSV."
            )

        used_titles.add(title_key)
        used_descriptions.add(description_key)

    batch_number = 0

    while len(records) < TARGET_RECORDS:
        remaining = TARGET_RECORDS - len(records)
        requested = min(BATCH_SIZE, remaining)
        batch_number += 1

        print(
            f"\nBatch {batch_number}: "
            f"{len(records)}/{TARGET_RECORDS} records"
        )

        try:
            candidates = generate_batch(
                batch_number,
                requested,
                records,
            )
        except Exception as error:
            print(f"API or parsing error: {error}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
            continue

        accepted = 0

        for candidate in candidates:
            if len(records) >= TARGET_RECORDS:
                break

            validated = validate_record(candidate, records)

            if not validated:
                continue

            # Guard against duplicates accepted in this batch.
            title_key = normalize(validated["title"])
            description_key = normalize(
                validated["description"]
            )

            if (
                title_key in used_titles
                or description_key in used_descriptions
            ):
                continue

            record_id = f"SSS{len(records) + 1:06d}"

            final_record = {
                "id": record_id,
                **validated,
            }

            records.append(final_record)
            used_titles.add(title_key)
            used_descriptions.add(description_key)
            accepted += 1

        # Save progress after every batch.
        save_records(records)

        print(
            f"Accepted: {accepted} | "
            f"Total saved: {len(records)}/{TARGET_RECORDS}"
        )

        if accepted == 0:
            print("No valid records accepted; waiting before retry.")
            time.sleep(3)

    # Final integrity checks.
    df = pd.read_csv(OUTPUT_FILE, dtype=str).fillna("")

    assert len(df) == TARGET_RECORDS
    assert df["id"].is_unique
    assert df["title"].map(normalize).is_unique
    assert df["description"].map(normalize).is_unique
    assert list(df.columns) == COLUMNS

    print("\nDataset generation completed.")
    print(f"CSV file: {OUTPUT_FILE.resolve()}")
    print(f"Total records: {len(df)}")
    print(f"Unique IDs: {df['id'].nunique()}")
    print(f"Categories: {df['type'].nunique()}")
    print("\nCategory distribution:")
    print(df["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
