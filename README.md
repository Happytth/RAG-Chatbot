# 🚀 Swift Ship RAG Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** built using **FastAPI + HuggingFace + FAISS**.
This chatbot answers customer queries using a custom knowledge base.

---

## 📌 Features

* 🔍 Semantic search using embeddings
* 📚 Knowledge base from custom data
* 🤖 LLM-powered responses (HuggingFace)
* ⚡ FastAPI backend with interactive docs
* 🔄 Rebuild index without restarting server

---

## 🏗️ Project Structure

```
RAG-Chatbot/
│
├── chatbot/
│   ├── app.py              # FastAPI server
│   ├── rag_pipeline.py     # RAG logic
│   ├── embeddings.py       # Embedding + FAISS
│   ├── vector_store/       # Saved FAISS index
│   └── __pycache__/        # Ignored
│
├── data/
│   └── data.txt            # Knowledge base
│
├── .env                    # API keys (NOT committed)
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd RAG-Chatbot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

Using **uv (recommended)**:

```bash
pip install uv
uv sync
```

OR using pip:

```bash
pip install -e .
```

---

### 4️⃣ Setup Environment Variables

Create a `.env` file in root:

```env
HF_API_TOKEN=your_huggingface_token
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

---

### 5️⃣ Build Embeddings (First Time Only)

```bash
python chatbot/embeddings.py
```

---

## 🚀 Run the Application

### Option 1: Using Python

```bash
cd chatbot
python app.py
```

---

### Option 2: Using Uvicorn

```bash
cd chatbot
uvicorn app:app --reload
```

---

## 🌐 Access API

* Main API:
  http://127.0.0.1:8000

* Swagger Docs (Recommended):
  http://127.0.0.1:8000/docs

---

## 🧪 Example API Usage

### POST `/chat`

```json
{
  "question": "How do I track my shipment?",
  "top_k": 5
}
```

---

### GET `/health`

Returns:

* API status
* Number of indexed chunks
* Model name

---

### POST `/rebuild`

Rebuilds FAISS index after updating data.

---

## ⚠️ Common Issues

### ❌ ModuleNotFoundError

```bash
uv add <package-name>
```

---

### ❌ API not starting

* Check `.env` file
* Ensure dependencies installed

---

### ❌ Port already in use

```bash
uvicorn app:app --reload --port 8001
```

---

## 🔐 .gitignore

Make sure to ignore:

```
__pycache__/
*.pyc
.env
venv/
.venv/
```

---

## 📦 Tech Stack

* FastAPI
* HuggingFace Inference API
* FAISS
* Sentence Transformers
* Python

---

## 👨‍💻 Author

Soubhagya Nayak

---

## ⭐ Future Improvements

* Add frontend UI
* Add authentication
* Streaming responses
* Docker deployment

---
