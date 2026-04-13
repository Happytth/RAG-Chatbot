"""
rag_pipeline.py
---------------
RAG pipeline for Swift Ship chatbot.

Uses huggingface_hub.InferenceClient (the modern, reliable API)
instead of raw requests to the legacy inference endpoint.

Flow per request:
  1. Retrieve top-k relevant chunks from FAISS (embeddings.py)
  2. Build a chat messages list
  3. Call HuggingFace InferenceClient.chat_completion()
  4. Return answer + source chunks
"""

import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from embeddings import (
    build_and_save,
    get_embedding_model,
    load_vector_store,
    retrieve,
    FAISS_INDEX_PATH,
    PKL_PATH,
)

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# Confirmed working on free HF Inference API (Mar 2026).
# Override via LLM_MODEL= in your .env file.
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Other working options:
#   "HuggingFaceH4/zephyr-7b-beta"
#   "microsoft/Phi-3-mini-4k-instruct"
#   "Qwen/Qwen2.5-7B-Instruct"
#   "google/gemma-2-2b-it"
#   "meta-llama/Llama-3.2-3B-Instruct"  (accept license on HF first)

TOP_K         = 5
MAX_CTX_CHARS = 2500

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a friendly and professional customer support assistant for Swift Ship, a small shipment and courier company.

Your job is to help customers with questions about:
- Shipment tracking and delivery status
- Pricing and available services
- Pickup and dropoff procedures
- Claims for lost or damaged packages
- Prohibited items and packaging requirements
- Account, billing, and volume discounts
- International shipping and customs

Rules you must always follow:
1. Answer ONLY using the provided context. Do not make up information.
2. If the answer is not in the context, say: "I don't have that information. Please contact our support team at support@swiftship.com or call +1-800-794-7447."
3. Be concise and helpful. Avoid long lists unless the question requires it.
4. Always refer to the company as "Swift Ship".
5. Never discuss competitors or make comparisons."""


# ── LLM call via InferenceClient ──────────────────────────────────────────────

def call_llm(context: str, question: str, model: str = LLM_MODEL) -> str:
    """
    Call HuggingFace InferenceClient using the chat_completion interface.
    This is the modern stable API — works with all current instruction models.
    """
    token = HF_API_TOKEN or os.getenv("HF_API_TOKEN", "")
    if not token or not token.startswith("hf_"):
        raise ValueError(
            "HF_API_TOKEN is missing or invalid.\n"
            "Add to your .env:  HF_API_TOKEN=hf_your_token_here\n"
            "Get a token at:    https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(token=token)

    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT FROM SWIFT SHIP KNOWLEDGE BASE:\n{context}\n\n"
                    f"CUSTOMER QUESTION: {question}"
                ),
            },
        ],
        max_tokens=400,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ── RAG Pipeline ───────────────────────────────────────────────────────────────

class SwiftShipRAG:
    """
    End-to-end RAG pipeline for Swift Ship customer support.

    Usage:
        rag = SwiftShipRAG()
        result = rag.query("How do I track my package?")
        print(result["answer"])
    """

    def __init__(self, rebuild_index: bool = False) -> None:
        if rebuild_index or not (
            os.path.exists(FAISS_INDEX_PATH) and os.path.exists(PKL_PATH)
        ):
            print("[RAG] Building vector store from data/data.txt ...")
            self.index, self.chunks, self.embed_model = build_and_save()
        else:
            print("[RAG] Loading existing vector store ...")
            self.index, self.chunks = load_vector_store()
            self.embed_model: SentenceTransformer = get_embedding_model()

        self.model_name = LLM_MODEL
        print(f"[RAG] Ready — {len(self.chunks)} chunks | LLM: {self.model_name}")

    def query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """
        Run retrieval + generation for a customer question.

        Returns:
            {
              "question":    str,
              "answer":      str,
              "found_in_kb": bool,
              "sources":     List[Tuple[str, float]],
              "context":     str,
            }
        """
        # 1. Retrieve relevant chunks
        sources: List[Tuple[str, float]] = retrieve(
            question, self.index, self.chunks, self.embed_model, top_k=top_k
        )

        # 2. No relevant chunks — return fallback immediately
        if not sources:
            return {
                "question":    question,
                "answer":      (
                    "I'm sorry, I don't have information about that. "
                    "Please contact our support team at support@swiftship.com "
                    "or call +1-800-794-7447 (Mon-Sat, 8AM-8PM EST)."
                ),
                "sources":     [],
                "context":     "",
                "found_in_kb": False,
            }

        # 3. Build context string (cap at MAX_CTX_CHARS)
        context_parts: List[str] = []
        total = 0
        for chunk, _ in sources:
            if total + len(chunk) > MAX_CTX_CHARS:
                break
            context_parts.append(chunk)
            total += len(chunk)
        context = "\n\n---\n\n".join(context_parts)

        # 4. Call LLM
        answer = call_llm(context, question, model=self.model_name)
        if not answer:
            answer = (
                "I couldn't generate a clear answer. Please contact "
                "support@swiftship.com for assistance."
            )

        return {
            "question":    question,
            "answer":      answer,
            "sources":     sources,
            "context":     context,
            "found_in_kb": True,
        }

    def rebuild(self) -> None:
        """Re-index data/data.txt without restarting the server."""
        print("[RAG] Rebuilding vector store ...")
        self.index, self.chunks, self.embed_model = build_and_save()
        print(f"[RAG] Rebuilt — {len(self.chunks)} chunks indexed.")


# ── CLI quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rag = SwiftShipRAG()

    if len(sys.argv) > 1:
        q      = " ".join(sys.argv[1:])
        result = rag.query(q)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
    else:
        print("Interactive mode (type 'quit' to exit)\n")
        while True:
            q = input("Customer: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            r = rag.query(q)
            print(f"\nSwift Ship Bot: {r['answer']}\n")