from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.routing.search import router as search_router
from src.utils.model_loader import MODEL_PATHS, load_model

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def demo():
    return FileResponse(STATIC_DIR / "demo.html")
