"""API de embeddings (FastAPI, assíncrona) para a KB de compliance do projeto DRX.

Um endpoint: POST /embed {"texts": [...], "kind": "query"|"passage"} -> vetores normalizados (e5-large, 1024d).
O servidor aplica os prefixos do e5 ("query: " / "passage: "); o cliente só manda o texto.

Concorrência: o event loop nunca roda o modelo. A inferência vai para um ThreadPoolExecutor
(torch libera o GIL), então /health e novas conexões continuam respondendo enquanto há inferência.
Por padrão há 1 worker de inferência (a CPU já é toda usada por uma inferência); pedidos concorrentes
esperam na fila. Fila cheia (EMBED_MAX_PENDING) -> 503 + Retry-After, em vez de acumular sem limite.

Rodar (da raiz do repositório), UM processo uvicorn (cada processo carrega ~2,2 GB de modelo):
    EMBED_API_KEY=... uvicorn src.app:app --host 0.0.0.0 --port 8080

Variáveis: EMBED_API_KEY (obrigatória; EMBED_ALLOW_NO_AUTH=1 só para dev local), EMBED_MODEL, EMBED_WORKERS,
EMBED_MAX_PENDING, EMBED_MAX_TEXTS, EMBED_MAX_CHARS, EMBED_TORCH_THREADS.
O conteúdo dos textos nunca é logado.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Callable, Literal, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger("embed_api")

MODEL_NAME = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large")
MAX_TEXTS = int(os.environ.get("EMBED_MAX_TEXTS", "64"))
MAX_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "8000"))  # e5 trunca em 512 tokens (~2000 chars)
MAX_PENDING = int(os.environ.get("EMBED_MAX_PENDING", "32"))
WORKERS = int(os.environ.get("EMBED_WORKERS", "1"))


class Encoder(Protocol):
    dim: int

    def encode(self, texts: list[str], kind: str) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """e5 via sentence-transformers, em CPU. Vetores normalizados (cosseno == produto interno)."""

    def __init__(self, name: str) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if "e5" not in name.lower():
            log.warning("modelo %s não parece e5: os prefixos 'query: '/'passage: ' podem não se aplicar", name)
        torch.set_num_threads(int(os.environ.get("EMBED_TORCH_THREADS", os.cpu_count() or 1)))
        self.model = SentenceTransformer(name, device="cpu")
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], kind: str) -> list[list[float]]:
        prefix = "query: " if kind == "query" else "passage: "
        vecs = self.model.encode(
            [prefix + t for t in texts], normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return vecs.tolist()


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)
    kind: Literal["query", "passage"] = "query"

    @field_validator("texts")
    @classmethod
    def _texts_ok(cls, texts: list[str]) -> list[str]:
        for t in texts:
            if not t.strip():
                raise ValueError("texto vazio")
            if len(t) > MAX_CHARS:
                raise ValueError(f"texto com mais de {MAX_CHARS} caracteres")
        return texts


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[float]]


def create_app(encoder_factory: Callable[[], Encoder] | None = None, model_name: str = MODEL_NAME) -> FastAPI:
    api_key = os.environ.get("EMBED_API_KEY", "")
    if not api_key and os.environ.get("EMBED_ALLOW_NO_AUTH") != "1":
        raise RuntimeError("defina EMBED_API_KEY (ou EMBED_ALLOW_NO_AUTH=1 apenas para desenvolvimento local)")
    factory = encoder_factory or (lambda: SentenceTransformerEncoder(model_name))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        app.state.executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="embed")
        # carrega o modelo fora do event loop; só aceita tráfego depois de pronto
        app.state.encoder = await loop.run_in_executor(app.state.executor, factory)
        await loop.run_in_executor(app.state.executor, app.state.encoder.encode, ["warmup"], "query")
        app.state.pending = 0
        app.state.max_pending = MAX_PENDING
        log.info("modelo %s pronto (dim=%d, workers=%d)", model_name, app.state.encoder.dim, WORKERS)
        yield
        app.state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="KB embeddings", lifespan=lifespan)

    async def require_key(authorization: str | None = Header(default=None)) -> None:
        if not api_key:  # EMBED_ALLOW_NO_AUTH=1
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="não autorizado", headers={"WWW-Authenticate": "Bearer"})

    @app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_key)])
    async def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
        st = request.app.state
        if st.pending >= st.max_pending:
            raise HTTPException(status_code=503, detail="fila cheia", headers={"Retry-After": "2"})
        st.pending += 1  # só o event loop mexe nisto: sem corrida
        t0 = time.perf_counter()
        try:
            vecs = await asyncio.get_running_loop().run_in_executor(st.executor, st.encoder.encode, req.texts, req.kind)
        finally:
            st.pending -= 1
        log.info("embed kind=%s n=%d ms=%.0f pending=%d", req.kind, len(req.texts), (time.perf_counter() - t0) * 1000, st.pending)
        return EmbedResponse(model=model_name, dim=st.encoder.dim, embeddings=vecs)

    @app.get("/health")
    async def health(request: Request) -> dict:
        st = request.app.state
        return {"status": "ok", "model": model_name, "dim": st.encoder.dim, "pending": st.pending}

    return app


# Instância para `uvicorn src.app:app`. O modelo só carrega no startup (lifespan), não no import.
app = create_app() if os.environ.get("EMBED_API_KEY") or os.environ.get("EMBED_ALLOW_NO_AUTH") == "1" else None
