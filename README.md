# Research Paper RAG System

A Retrieval-Augmented Generation (RAG) system for querying and analyzing research papers using **Google Gemini**, **HuggingFace Embeddings**, **ChromaDB**, and **LangChain**, featuring both an interactive **Jupyter Notebook exploration** and a **Streamlit Web Chatbot**.

---

## 🚀 Overview

This repository provides an end-to-end implementation of a RAG pipeline designed to extract knowledge from complex research papers and deliver contextual, verifiable answers.

- **Frontend Application:** Interactive Streamlit web chat UI with ChatGPT-style message bubbles, expandable source citations, response time metrics, and sample questions.
- **Exploratory Notebook:** `rag.ipynb` contains the step-by-step development process: document loading, chunking strategies, embeddings generation, vector database indexing, MMR retrieval, and generation with Gemini.

---

## 🛠️ Architecture & Tech Stack

```text
[Research Paper (PDF)]
       │
       ▼
[PyPDFLoader / Text Splitter]
       │
       ▼ (Chunks: size ~1000, overlap ~200)
[HuggingFace Embeddings: BAAI/bge-base-en-v1.5]
       │
       ▼
[ChromaDB Vector Store]
       │
       ▼ (Maximal Marginal Relevance - MMR)
[Retrieved Context Chunks] ───► [Prompt Template]
                                        │
                                        ▼
                             [Google Gemini 2.5 Flash]
                                        │
                                        ▼
                                 [Generated Answer]
```

- **LLM:** Google Gemini 2.5 Flash (`google-genai` / `langchain-google-genai`)
- **Embedding Model:** `BAAI/bge-base-en-v1.5` via HuggingFace
- **Vector Database:** ChromaDB
- **Orchestration:** LangChain
- **Web App:** Streamlit
- **Document Ingestion:** PyPDFLoader

---

## 📁 Repository Structure

```text
.
├── RAG_Chatbot_Streamlit/
│   ├── app.py                 # Streamlit web application
│   ├── requirements.txt       # Dependencies for the Streamlit app
│   ├── .env.example           # Example environment configuration
│   ├── researchpaper.pdf      # Sample research paper for testing
│   ├── test_rag.py            # Unit/smoke tests for RAG pipeline
│   └── README.md              # App-specific documentation
├── rag.ipynb                  # Step-by-step Jupyter Notebook development
├── .gitignore                 # Standard Python, Jupyter, and secret exclusions
└── README.md                  # Main repository documentation
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Balasurya555/RAG-system-.git
cd RAG-system-
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
cd RAG_Chatbot_Streamlit
pip install -r requirements.txt
```

### 4. Configure API Key

Copy the `.env.example` file and insert your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=your_actual_google_api_key_here
```

Get a key at [Google AI Studio](https://aistudio.google.com/).

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`.

---

## 📓 Running the Jupyter Notebook

To explore the pipeline step by step:

```bash
pip install jupyter
jupyter notebook rag.ipynb
```

Ensure your `GOOGLE_API_KEY` is set in your environment or added to the notebook setup cell.

---

## ✨ Features

- **Conversational Memory:** Preserves multi-turn chat history.
- **Smart MMR Retrieval:** Balances relevance with diversity to avoid duplicate context chunks.
- **Source Citations:** Expandable context viewer showing the exact paper page and chunk used for the answer.
- **Performance Metrics:** Measures query-to-response latency.
- **Error Handling:** Gracefully handles missing API keys, document load errors, and rate limits.

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify it for your research and development.
