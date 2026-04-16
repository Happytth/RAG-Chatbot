import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import SwiftShipRAG

# ── Shared pipeline instance ───────────────────────────────────────────────────
rag: Optional[SwiftShipRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the RAG pipeline once at startup."""
    global rag
    print("[app] Loading Swift Ship RAG pipeline ...")
    rag = SwiftShipRAG()
    print("[app] Server ready.")
    yield
    print("[app] Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Swift Ship Chatbot API",
    description=(
        "Customer support chatbot for Swift Ship powered by "
        "Retrieval-Augmented Generation (RAG) + HuggingFace."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend / mobile clients to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=500,
        example="How do I track my shipment?",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of knowledge-base chunks to retrieve (1–10).",
    )


class SourceChunk(BaseModel):
    text:  str
    score: float


class ChatResponse(BaseModel):
    question:    str
    answer:      str
    found_in_kb: bool
    sources:     List[SourceChunk]


class HealthResponse(BaseModel):
    status:        str
    chunks_loaded: int
    llm_model:     str


class RebuildResponse(BaseModel):
    message:       str
    chunks_loaded: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask the Swift Ship chatbot a question",
    tags=["Chatbot"],
)
async def chat(request: ChatRequest):
    """
    Send a customer question and receive an answer grounded in
    Swift Ship's knowledge base (data/data.txt).

    - **question**: The customer's question (2–500 characters)
    - **top_k**: How many knowledge base passages to retrieve (default 5)
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        result = rag.query(request.question.strip(), top_k=request.top_k)
    except RuntimeError as exc:
        # HuggingFace model still warming up
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except ValueError as exc:
        # Missing API token or bad config
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {exc}",
        )

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        found_in_kb=result["found_in_kb"],
        sources=[
            SourceChunk(text=text, score=round(score, 4))
            for text, score in result["sources"]
        ],
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    tags=["System"],
)
async def health():
    """Returns pipeline status and number of indexed chunks."""
    return HealthResponse(
        status="ok",
        chunks_loaded=rag.chunks if rag else 0,
        llm_model=rag.model_name if rag else "not loaded",
    )


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Rebuild knowledge base index",
    tags=["System"],
)
async def rebuild():
    """
    Re-reads data/data.txt, re-embeds all chunks, and rebuilds the
    FAISS index in-place. Use this after updating the knowledge base
    without restarting the server.
    """
    try:
        rag.rebuild()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rebuild failed: {exc}",
        )

    return RebuildResponse(
        message="Knowledge base rebuilt successfully.",
        chunks_loaded=len(rag.chunks),
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )