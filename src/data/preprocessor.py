
import re
from pathlib import Path

import pandas as pd


# ============================================
# SYSTEMATIC SIMILARITY SEARCH
# Step 1: Data Preprocessing
# ============================================

# Get the directory containing this Python script
# SCRIPT_DIR = Path(__file__).resolve().parent

# Set the script directory explicitly
SCRIPT_DIR = Path(
    "/Users/javid-67239/mdj/SystematicSimilaritySearch/src/script"
)


# Input and output file paths
CSV_PATH = SCRIPT_DIR / "systematic_similarity_search_dataset.csv"
OUTPUT_PATH = SCRIPT_DIR / "processed_dataset.csv"


def clean_text(text: str) -> str:
    """
    Normalize text while preserving meaningful words,
    """
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_dataset() -> None:
    """Load, clean, validate, and save the CSV dataset."""

    # 1. Check whether the input file exists
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}"
        )

    # 2. Load CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    print("Original dataset shape:", df.shape)
    print("Available columns:", df.columns.tolist())

    # 3. Validate required columns
    required_columns = [
        "title",
        "description",
        "keywords",
        "type",
    ]

    # Handle accidental spaces in column names
    df.columns = df.columns.str.strip()

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 4. Keep the required columns
    df = df[required_columns].copy()

    # 5. Handle missing values
    for column in required_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    # 6. Remove rows with no meaningful title,
    # description, or keywords
    text_columns = ["title", "description", "keywords"]

    has_text = df[text_columns].apply(
        lambda row: any(value.strip() for value in row),
        axis=1,
    )

    df = df[has_text].copy()

    # 7. Remove exact duplicate records
    df = df.drop_duplicates(
        subset=required_columns
    ).reset_index(drop=True)

    # 8. Clean text fields
    for column in required_columns:
        df[column] = df[column].apply(clean_text)

    # 9. Combine fields into searchable text
    df["combined_text"] = (
        "title: " + df["title"] + ". "
        + "description: " + df["description"] + ". "
        + "keywords: " + df["keywords"] + ". "
        + "type: " + df["type"]
    ).str.strip()

    # 10. Remove empty combined text
    df = df[
        df["combined_text"].str.strip().ne("")
    ].reset_index(drop=True)

    # 11. Save processed dataset
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # 12. Print summary
    print("\n========== PREPROCESSING SUMMARY ==========")
    print("Original records:", len(pd.read_csv(CSV_PATH)))
    print("Processed records:", len(df))
    print("Output file:", OUTPUT_PATH)
    print("Output columns:", df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSample processed records:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    preprocess_dataset()
