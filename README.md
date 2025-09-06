# Mini RAG App: Context-Aware Q&A

[![Vercel Deploy](https://img.shields.io/badge/Frontend-Vercel-black?style=for-the-badge&logo=vercel)](https://mini-rag-website.vercel.app)
[![Render Deploy](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render)](https://mini-rag-backend-0xfd.onrender.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)

A fully-functional, web-based RAG (Retrieval-Augmented Generation) application that allows users to get fact-based, cited answers from their own documents.

## 🚀 Live Demo

* **Frontend (Vercel):** [https://mini-rag-website.vercel.app](https://mini-rag-website.vercel.app)
* **Backend (Render):** [https://mini-rag-backend-0xfd.onrender.com](https://mini-rag-backend-0xfd.onrender.com)

*(Note: The backend is on Render's free tier and may "spin down." The first request after a period of inactivity might take 30-60 seconds to respond.)*

---

## 📖 Table of Contents

* [Problem Statement](#-problem-statement)
* [Architecture Diagram](#-architecture-diagram)
* [Key Features](#-key-features)
* [Tech Stack](#-tech-stack)
* [Configuration Details](#-configuration-details)
* [Setup and Run Locally](#-setup-and-run-locally)
* [Dataset](#-dataset)
* [Remarks & Trade-offs](#-remarks--trade-offs)

---

## 🎯 Problem Statement

Large Language Models (LLMs) have revolutionized natural language understanding but suffer from critical limitations: their knowledge is static, they cannot access private information, and they are prone to "hallucination." This makes them unreliable for tasks requiring high factual accuracy.

This project, the "Mini RAG App," directly addresses these challenges by implementing the **Retrieval-Augmented Generation (RAG)** architecture. It provides an interface where users can ingest any text document, transforming the LLM from a generalist into a grounded expert on that specific content. The application ensures that all generated answers are verifiable, accurate, and directly cited from the source material.

---

## 🏗️ Architecture Diagram

The application follows a standard full-stack, cloud-native architecture:

```
User Browser --> Frontend (React on Vercel)
      |
      | (API Calls)
      v
Backend (FastAPI on Render) --> LLM & Services
      |                             |
      | (Vector Search)             | (Generate Answer)
      v                             v
Vector DB (Pinecone) <---------- Google AI (Gemini & Embeddings)
                                  |
                                  | (Reranking)
                                  v
                                Cohere API
```

---

## ✨ Key Features

* **Dynamic Document Ingestion:** Paste any text directly into the frontend for processing.
* **Vectorization & Storage:** Text is chunked, converted to vector embeddings, and stored in a cloud-hosted Pinecone database.
* **Retrieval & Reranking:** Implements a top-k similarity search followed by a Cohere reranker to find the most relevant context.
* **Grounded Generation:** Uses Google's Gemini model to generate answers based *only* on the retrieved context.
* **Inline Citations:** The final answer includes citations that link back to the specific source text chunks.

---

## 🛠️ Tech Stack

| Category      | Technology                                                                                                    |
|---------------|---------------------------------------------------------------------------------------------------------------|
| **Frontend** | [React](https://react.dev/), [Vite](https://vitejs.dev/), CSS                                                  |
| **Backend** | [Python](https://www.python.org/), [FastAPI](https://fastapi.tiangolo.com/)                                     |
| **AI/ML** | [LangChain](https://www.langchain.com/), [Pinecone](https://www.pinecone.io/), [Google GenAI](https://ai.google/), [Cohere](https://cohere.com/) |
| **Deployment**| [Vercel](https://vercel.com/) (Frontend), [Render](https://render.com/) (Backend)                               |

---

## ⚙️ Configuration Details

* **Vector Database:**
    * **Provider:** Pinecone
    * **Index Name:** `mini-rag-models`
    * **Dimensionality:** 768 (for `models/embedding-001`)
    * **Metric:** `cosine`
* **Chunking & Embeddings:**
    * **Strategy:** `RecursiveCharacterTextSplitter`
    * **Chunk Size:** 1000 characters
    * **Chunk Overlap:** 150 characters
    * **Embedding Model:** Google `models/embedding-001`
* **Retriever & Reranker:**
    * **Retriever:** Pinecone vector store `as_retriever(search_kwargs={"k": 20})`
    * **Reranker:** Cohere `rerank-english-v3.0` with `top_n=5`

---

## 🚀 Setup and Run Locally

To run this project on your local machine, follow these steps:

**Prerequisites:**
* Python 3.10+
* Node.js and npm
* Git

**1. Clone the repository:**
```bash
git clone [https://github.com/smreetyyadav37/mini-rag-website.git](https://github.com/smreetyyadav37/mini-rag-website.git)
cd mini-rag-website
```

**2. Backend Setup:**
```bash
# Navigate to the backend folder
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create a .env file and add your API keys
cp .env.example .env
# Now edit the .env file with your actual keys

# Run the backend server
uvicorn main:app --reload
```

**3. Frontend Setup:**
```bash
# Navigate to the frontend folder from the root
cd frontend

# Install dependencies
npm install

# Create a .env.local file and add the backend URL
# (You can copy from .env.example)
echo "VITE_API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)" > .env.local

# Run the frontend development server
npm run dev
```
The application should now be running locally.

---

## 📚 Dataset

The sample PDFs used for testing this RAG application can be downloaded from the link below.

* **[Download the Sample Dataset PDFs](https://drive.google.com/drive/folders/16clo1ajWQwB9e6UAr1jd_8cQjsYmIrBX?usp=sharing)**


---

## 📝 Remarks & Trade-offs

* **Cold Starts:** The backend is deployed on Render's free tier, which spins down after 15 minutes of inactivity. This results in a "cold start" delay of 30-60 seconds for the first request.
* **Model Selection:** The choice of Google's `embedding-001` and `gemini-1.5-flash` models was based on a balance of strong performance and free-tier availability.
* **Future Work:** A potential improvement would be to implement response streaming from the backend to the frontend. This would provide a better user experience by displaying the answer word-by-word as it's being generated.
