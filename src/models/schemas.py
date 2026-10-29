from enum import Enum

from pydantic import BaseModel, Field


class ModelName(str, Enum):
    word2vec = "word2vec"
    fasttext = "fasttext"


class SearchResult(BaseModel):
    rank: int
    title: str
    type: str
    description: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    model: ModelName
    tokens: list[str]
    known_tokens: list[str]
    results: list[SearchResult]


class ModelInfo(BaseModel):
    name: ModelName
    vocabulary_size: int
    vector_size: int
    documents_indexed: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    model: ModelName = ModelName.word2vec
    top_k: int = Field(default=5, ge=1, le=50)
