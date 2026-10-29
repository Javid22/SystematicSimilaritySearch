
import re
import sys
from pathlib import Path

import joblib
import numpy as np


# ============================================
# SYSTEMATIC SIMILARITY SEARCH
# Test saved Word2Vec model
# ============================================

# Set the script directory explicitly
SCRIPT_DIR = Path(
    "/Users/javid-67239/mdj/SystematicSimilaritySearch/src/script"
)

PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "systematic_similarity_word2vec.pkl"
)

# Reuse the exact cleaning step used to build the training data.
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
from preprocessor import clean_text  # noqa: E402

TOP_K = 5


def load_model(path):
    """Load the saved pickle artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    package = joblib.load(path)

    print("Pickle file loaded successfully.")

    return package


def extract_artifacts(package):
    """
    Support either a packaged model dictionary
    or a directly saved Gensim Word2Vec model.
    """
    if isinstance(package, dict):
        model = (
            package.get("word2vec_model")
            or package.get("embedding_model")
            or package.get("model")
        )

        # Handle a package using another common key.
        if model is None:
            model = package.get("fasttext_model")

        document_vectors = package.get("document_vectors")
        documents = package.get("documents")

    else:
        model = package
        document_vectors = None
        documents = None

    if model is None:
        # The package itself might be a Gensim model.
        if hasattr(package, "wv"):
            model = package

    if model is None or not hasattr(model, "wv"):
        raise ValueError(
            "Could not find a Gensim Word2Vec model. "
            "Check the keys and structure of your pickle file."
        )

    return model, document_vectors, documents


def tokenize(text):
    """Same tokenizer as train_word2vec.py."""
    return re.findall(r"\b\w+\b", str(text).lower())


def preprocess_query(query):
    """Clean the raw input and split it into tokens."""
    return tokenize(clean_text(query))


def create_query_vector(query, model):
    """Pre-process the query, then average the embeddings of known words."""
    tokens = preprocess_query(query)

    known_tokens = [
        token for token in tokens
        if token in model.wv.key_to_index
    ]

    if not known_tokens:
        return None, tokens, known_tokens

    vectors = np.array([
        model.wv[token] for token in known_tokens
    ])

    return np.mean(vectors, axis=0), tokens, known_tokens


def cosine_scores(query_vector, document_vectors):
    """Calculate cosine similarity safely."""
    query_norm = np.linalg.norm(query_vector)
    doc_norms = np.linalg.norm(document_vectors, axis=1)

    if query_norm == 0:
        raise ValueError("Query vector is zero.")

    denominator = doc_norms * query_norm

    scores = np.zeros(len(document_vectors), dtype=float)

    valid = denominator > 0

    scores[valid] = (
        document_vectors[valid] @ query_vector
    ) / denominator[valid]

    return scores


def main():
    print("\n========== WORD2VEC MODEL TEST ==========")

    # 1. Load the saved model
    package = load_model(MODEL_PATH)

    # 2. Extract model and search artifacts
    model, document_vectors, documents = extract_artifacts(
        package
    )

    print("Model class:", type(model).__name__)
    print("Vocabulary size:", len(model.wv))
    print("Embedding dimensions:", model.wv.vector_size)

    if len(model.wv) == 0:
        raise ValueError("The model vocabulary is empty.")

    # 3. Test a known word embedding
    sample_word = next(iter(model.wv.key_to_index))

    sample_vector = model.wv[sample_word]

    print("\nEmbedding test:")
    print("Sample word:", sample_word)
    print("Vector shape:", sample_vector.shape)
    print("Vector contains finite values:",
          bool(np.isfinite(sample_vector).all()))

    if not np.isfinite(sample_vector).all():
        raise ValueError("The model contains invalid vector values.")

    # 4. Interactive query loop
    print("\nType words or a sentence to search (blank / 'quit' to exit).")

    while True:
        try:
            query = input("\nEnter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in {"quit", "exit", "q"}:
            break

        run_query(query, model, document_vectors, documents)

    print("\nWord2Vec search test completed.")


def run_query(query, model, document_vectors, documents):
    query_vector, tokens, known_tokens = create_query_vector(
        query, model
    )

    print("Pre-processed tokens:", tokens)
    print("Known query words:", known_tokens)

    if query_vector is None:
        print("No query words found in the Word2Vec vocabulary.")
        return

    print("Query vector shape:", x.shape)
    print("Query vector (first 10 dims):", np.round(query_vector[:10], 4))

    print("\nSimilar words:")
    for word in known_tokens:
        print(f"  For '{word}':")
        for similar_word, score in model.wv.most_similar(word, topn=3):
            print(f"    {similar_word}: {score:.4f}")

    if document_vectors is None or documents is None:
        print("\nNo document_vectors/documents in pickle; skipping ranking.")
        return

    document_vectors = np.asarray(document_vectors)

    if document_vectors.ndim != 2:
        raise ValueError("Document vectors must be a 2D matrix.")

    if document_vectors.shape[1] != len(query_vector):
        raise ValueError(
            "Query and document vector dimensions do not match."
        )

    if len(documents) != len(document_vectors):
        raise ValueError(
            "Document metadata and vector counts do not match."
        )

    scores = cosine_scores(query_vector, document_vectors)
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    print("\n========== TOP SEARCH RESULTS ==========")

    for rank, index in enumerate(top_indices, start=1):
        if hasattr(documents, "iloc"):
            document = documents.iloc[index]
        else:
            document = documents[index]

        print(f"\nRank: {rank}")
        print(f"Title: {document.get('title', 'Untitled')}")
        print(f"Type: {document.get('type', '')}")
        print(f"Similarity: {scores[index]:.4f}")
        print(f"Description: {document.get('description', '')}")


if __name__ == "__main__":
    main()
