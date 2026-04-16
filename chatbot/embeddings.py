# embeddings.py  (LangChain version)

from pathlib import Path
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.txt"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# ── Embeddings ────────────────────────────────────────────────
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ── Build Vector Store ────────────────────────────────────────
def build_and_save() -> FAISS:
    print("[LC] Loading data...")
    loader = TextLoader(str(DATA_PATH))
    docs = loader.load()

    print("[LC] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks: List[Document] = splitter.split_documents(docs)

    print(f"[LC] {len(chunks)} chunks created")

    embeddings = get_embeddings()

    print("[LC] Creating FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_DIR))

    print("[LC] Vector store saved")
    return vectorstore


# ── Load Vector Store ─────────────────────────────────────────
def load_vector_store() -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── Retriever ─────────────────────────────────────────────────
def get_retriever(k: int = 5):
    vs = load_vector_store()
    return vs.as_retriever(search_kwargs={"k": k})


# ── CLI test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building vector store...")
    build_and_save()

    retriever = get_retriever()

    query = "How do I track my package?"
    docs = retriever.invoke(query)

    print("\nTop result:\n")
    print(docs[0].page_content)