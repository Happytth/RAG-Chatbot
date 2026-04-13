"""
embeddings.py
-------------
Handles all vector store operations for Swift Ship RAG Chatbot:
  - Load and chunk data/data.txt
  - Generate embeddings via sentence-transformers
  - Build and persist FAISS index
  - Retrieve top-k chunks for a query
"""

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH        = os.path.join(BASE_DIR, "..", "data", "data.txt")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
PKL_PATH         = os.path.join(VECTOR_STORE_DIR, "index.pkl")

# ── Model — good balance of speed and accuracy for domain-specific text ────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking config — tuned for FAQ + policy document style content ────────────
CHUNK_SIZE = 120   # words per chunk (smaller = more precise retrieval)
OVERLAP    = 25    # word overlap between consecutive chunks


# ── Text loading ───────────────────────────────────────────────────────────────

def load_text(path: str = DATA_PATH) -> str:
    """Read the knowledge base file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Knowledge base not found at: {path}\n"
            "Make sure data/data.txt exists."
        )
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"[embeddings] Loaded {len(content):,} characters from {path}")
    return content


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[str]:
    """
    Split text into overlapping word-level windows.
    Smaller chunks work better for FAQ-style knowledge bases
    because each chunk stays focused on one topic.
    """
    words = [w for w in text.split() if w.strip()]
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())
        if start + chunk_size >= len(words):
            break

    print(f"[embeddings] Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


# ── Embedding ──────────────────────────────────────────────────────────────────

def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence-transformer embedding model."""
    print(f"[embeddings] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode chunks into float32 embeddings, L2-normalised for cosine similarity.
    Returns array of shape (N, 384).
    """
    print(f"[embeddings] Encoding {len(chunks)} chunks ...")
    vectors = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype(np.float32)


# ── FAISS index ────────────────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS inner-product index (cosine on L2-normalised vecs)."""
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[embeddings] FAISS index ready — {index.ntotal} vectors, dim={dim}")
    return index


def save_vector_store(index: faiss.Index, chunks: List[str]) -> None:
    """Persist FAISS index and chunk texts to disk."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(PKL_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[embeddings] Saved index  -> {FAISS_INDEX_PATH}")
    print(f"[embeddings] Saved chunks -> {PKL_PATH}")


def load_vector_store() -> Tuple[faiss.Index, List[str]]:
    """Load FAISS index and chunk texts from disk."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(PKL_PATH):
        raise FileNotFoundError(
            "Vector store not found. Run:  python embeddings.py"
        )
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(PKL_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"[embeddings] Loaded — {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


# ── One-shot builder ───────────────────────────────────────────────────────────

def build_and_save(
    data_path: str = DATA_PATH,
) -> Tuple[faiss.Index, List[str], SentenceTransformer]:
    """Full pipeline: load -> chunk -> embed -> index -> save."""
    text       = load_text(data_path)
    chunks     = chunk_text(text)
    model      = get_embedding_model()
    embeddings = embed_chunks(chunks, model)
    index      = build_index(embeddings)
    save_vector_store(index, chunks)
    return index, chunks, model


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = 5,
    min_score: float = 0.25,
) -> List[Tuple[str, float]]:
    """
    Retrieve the top-k most relevant chunks for a query.
    Chunks below min_score are filtered out to avoid noisy context.

    Returns:
        List of (chunk_text, similarity_score) sorted by score descending.
    """
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and float(score) >= min_score:
            results.append((chunks[idx], float(score)))

    return results


# ── CLI / smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Swift Ship -- Building Vector Store")
    print("=" * 55)

    index, chunks, model = build_and_save()
    print(f"\nDone. {len(chunks)} chunks indexed.\n")

    test_queries = [
        "How do I track my package?",
        "What is the price for express delivery?",
        "My shipment is damaged, what should I do?",
        "Do you offer international shipping?",
        "Can I change my delivery address?",
    ]

    print("-" * 55)
    print("Smoke test:")
    print("-" * 55)
    for q in test_queries:
        hits = retrieve(q, index, chunks, model, top_k=1)
        if hits:
            print(f"\nQ: {q}")
            print(f"   score={hits[0][1]:.4f}  {hits[0][0][:110]} ...")
        else:
            print(f"\nQ: {q}\n   No result above threshold.")