from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.txt"
FAISS_PATH = BASE_DIR / "vector_store"


class SwiftShipRAG:
    def __init__(self):
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Build / Load FAISS
        if not FAISS_PATH.exists():
            loader = TextLoader(str(DATA_PATH))
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=120,
            )
            chunks = splitter.split_documents(docs)

            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(str(FAISS_PATH))

        self.vectorstore = FAISS.load_local(
            str(FAISS_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # LLM
        endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="conversational",
            temperature=0.2,
            max_new_tokens=400,
        )
        self.llm = ChatHuggingFace(llm=endpoint)

        self.model_name = "Qwen/Qwen2.5-7B-Instruct"

        # Prompt
        prompt = PromptTemplate(
            template="""
                            You are a Swift Ship support assistant.

                            Answer ONLY from the context.
                            If not found, say you don't know.

                            Context:
                            {context}

                            Question: {question}
                            """,
            input_variables=["context", "question"],
        )

        def retrieve_docs(question: str):
            docs = self.retriever.invoke(question)
            return "\n\n".join(doc.page_content for doc in docs)

        # Runnable chain
        self.parallel_chain=RunnableParallel({
                "context": RunnableLambda(retrieve_docs),
                "question": RunnablePassthrough(),
            })
        self.rag_chain = (
            self.parallel_chain
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Needed by FastAPI health endpoint
        self.chunks = self.vectorstore.index.ntotal

    # ✅ what FastAPI calls
    def query(self, question: str, top_k: int = 5):
        answer = self.rag_chain.invoke(question)

        docs = self.retriever.invoke(question)

        sources = [(d.page_content, 0.0) for d in docs]

        return {
            "question": question,
            "answer": answer,
            "found_in_kb": answer.lower()!= "i don't know" ,
            "sources": sources,
        }

    # ✅ what /rebuild calls
    def rebuild(self):
        if FAISS_PATH.exists():
            for f in FAISS_PATH.iterdir():
                f.unlink()
        self.__init__()