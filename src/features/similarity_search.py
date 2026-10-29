import numpy as np

from src.utils.model_loader import LoadedModel
from src.utils.text import preprocess_query


def create_query_vector(query: str, loaded: LoadedModel):
    """Average the embeddings of every token the model can represent."""
    tokens = preprocess_query(query)

    vectors = []
    used_tokens = []
    for token in tokens:
        try:
            vectors.append(loaded.embedding_model.wv[token])
            used_tokens.append(token)
        except KeyError:
            continue

    if not vectors:
        return None, tokens, used_tokens

    query_vector = np.mean(np.array(vectors), axis=0)
    return query_vector, tokens, used_tokens


def cosine_scores(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("Query vector is zero.")

    doc_norms = np.linalg.norm(document_vectors, axis=1)
    denominator = doc_norms * query_norm

    scores = np.zeros(len(document_vectors), dtype=float)
    valid = denominator > 0
    scores[valid] = (document_vectors[valid] @ query_vector) / denominator[valid]
    return scores


def search(query: str, loaded: LoadedModel, top_k: int = 5):
    query_vector, tokens, used_tokens = create_query_vector(query, loaded)

    if query_vector is None:
        return [], tokens, used_tokens

    if loaded.document_vectors.shape[1] != len(query_vector):
        raise ValueError("Query and document vector dimensions do not match.")

    scores = cosine_scores(query_vector, loaded.document_vectors)
    top_indices = np.argsort(scores)[::-1][:top_k]

    documents = loaded.documents
    results = []
    for rank, index in enumerate(top_indices, start=1):
        document = documents.iloc[index]
        results.append(
            {
                "rank": rank,
                "title": document.get("title", "Untitled"),
                "type": document.get("type", ""),
                "description": document.get("description", ""),
                "similarity": float(scores[index]),
            }
        )

    return results, tokens, used_tokens
