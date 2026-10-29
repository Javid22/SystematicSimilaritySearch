from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.routing.search import router as search_router
from src.utils.model_loader import MODEL_PATHS, load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    for name in MODEL_PATHS:
        try:
            app.state.models[name] = load_model(name)
            print(f"[startup] loaded '{name}' model")
        except (FileNotFoundError, ValueError) as error:
            print(f"[startup] skipping '{name}': {error}")
    yield
    app.state.models.clear()


app = FastAPI(
    title="Systematic Similarity Search API",
    description="Search indexed documents by semantic similarity using Word2Vec / FastText embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(search_router)


@app.get("/health")
def health():
    return {"status": "ok"}
