from fastapi import APIRouter, HTTPException, Query, Request

from src.features.similarity_search import search
from src.models.schemas import ModelInfo, ModelName, SearchRequest, SearchResponse

router = APIRouter()


def _get_loaded_model(request: Request, model: ModelName):
    loaded = request.app.state.models.get(model.value)
    if loaded is None:
        raise HTTPException(status_code=503, detail=f"Model '{model.value}' failed to load at startup.")
    return loaded


@router.get("/models", response_model=list[ModelInfo])
def list_models(request: Request):
    return [
        ModelInfo(
            name=ModelName(loaded.name),
            vocabulary_size=len(loaded.embedding_model.wv),
            vector_size=loaded.vector_size,
            documents_indexed=len(loaded.documents),
        )
        for loaded in request.app.state.models.values()
    ]


@router.get("/search", response_model=SearchResponse)
def search_get(
    request: Request,
    query: str = Query(min_length=1),
    model: ModelName = ModelName.word2vec,
    top_k: int = Query(default=5, ge=1, le=50),
):
    return _run_search(request, query, model, top_k)


@router.post("/search", response_model=SearchResponse)
def search_post(request: Request, payload: SearchRequest):
    return _run_search(request, payload.query, payload.model, payload.top_k)


def _run_search(request: Request, query: str, model: ModelName, top_k: int) -> SearchResponse:
    loaded = _get_loaded_model(request, model)

    try:
        results, tokens, known_tokens = search(query, loaded, top_k=top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return SearchResponse(
        query=query,
        model=model,
        tokens=tokens,
        known_tokens=known_tokens,
        results=results,
    )
