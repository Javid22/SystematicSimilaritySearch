
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText


# ============================================
# SYSTEMATIC SIMILARITY SEARCH
# Step 3: FastText training
# ============================================

# Set the script directory explicitly
SCRIPT_DIR = Path(
    "/Users/javid-67239/mdj/SystematicSimilaritySearch/src/script"
)
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = SCRIPT_DIR / "processed_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PKL_PATH = MODEL_DIR / "systematic_similarity_fasttext.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_SIZE = 100
WINDOW = 5
EPOCHS = 10
SEED = 42


# 1. Tokenization
def tokenize(text):
    """Convert text into lowercase word and number tokens."""
    return re.findall(r"\b\w+\b", str(text).lower())


# 2. Create document vectors
def create_document_vector(tokens, model):
    """
    Average word vectors to create one document vector.

    FastText can generate vectors for many unseen words
    through character n-grams.
    """
    vectors = []

    for token in tokens:
        try:
            vectors.append(model.wv[token])
        except KeyError:
            continue

    if not vectors:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


# 3. Train FastText
def train_fasttext(sentences):
    print("\nTraining FastText...")

    model = FastText(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=1,
        workers=1,
        sg=1,
        epochs=EPOCHS,
        min_n=3,
        max_n=6,
        seed=SEED,
    )

    if len(model.wv) == 0:
        raise ValueError("FastText vocabulary is empty.")

    vectors = np.vstack([
        create_document_vector(tokens, model)
        for tokens in sentences
    ])

    if not np.any(np.linalg.norm(vectors, axis=1) > 0):
        raise ValueError(
            "FastText could not create usable document vectors."
        )

    print("FastText vocabulary:", len(model.wv))

    return model, vectors


# 4. Main training pipeline
def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run prepare_text.py first."
        )

    df = pd.read_csv(DATA_PATH).fillna("")

    if "prepared_tokens" in df.columns:
        text_column = "prepared_tokens"
    elif "combined_text" in df.columns:
        text_column = "combined_text"
    else:
        raise ValueError(
            "CSV must contain prepared_tokens "
            "or combined_text."
        )

    # Convert the prepared text into token lists.
    sentences = [
        tokenize(text)
        for text in df[text_column].astype(str)
    ]

    # Remove empty documents while preserving row alignment.
    valid_rows = [
        index for index, tokens in enumerate(sentences)
        if tokens
    ]

    df = df.iloc[valid_rows].reset_index(drop=True)
    sentences = [sentences[index] for index in valid_rows]

    if not sentences:
        raise ValueError("No usable text found in the dataset.")

    print("Documents for training:", len(sentences))
    print("Training input:", text_column)

    model, document_vectors = train_fasttext(sentences)
    model_type = "FastText"

    # Save searchable metadata alongside the vectors.
    metadata_columns = [
        column for column in [
            "title",
            "description",
            "keywords",
            "type",
            "combined_text",
        ]
        if column in df.columns
    ]

    documents = df[metadata_columns].copy()

    # Package the complete set of inference artifacts.
    model_package = {
        "model_type": model_type,
        "embedding_model": model,
        "document_vectors": document_vectors,
        "documents": documents,
        "vector_size": VECTOR_SIZE,
        "tokenization": "lowercase_regex_words_numbers",
        "document_vector_method": "mean_word_embeddings",
        "training_config": {
            "window": WINDOW,
            "epochs": EPOCHS,
            "seed": SEED,
        },
    }

    joblib.dump(model_package, PKL_PATH, compress=3)

    print("\n========== TRAINING SUMMARY ==========")
    print("Selected model:", model_type)
    print("Documents indexed:", len(documents))
    print("Document vectors:", document_vectors.shape)
    print("Saved model:", PKL_PATH)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as error:
        print(f"\nTraining failed: {error}")
        sys.exit(1)
