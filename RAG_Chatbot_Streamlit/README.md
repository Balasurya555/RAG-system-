# Research Paper RAG Chatbot

## Project Overview
This project is a modern, production-ready Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on a provided research paper. It uses a Streamlit frontend with a ChatGPT-style interface and a LangChain-powered backend.

## Features
- **ChatGPT-Style Chat History:** Preserves conversations using Streamlit session state.
- **Smart Retrieval:** Uses MMR (Maximal Marginal Relevance) for diverse document retrieval.
- **Modern UI:** Features answer cards, expandable retrieved context chunks, and metrics in the sidebar.
- **Fast Performance:** Caches the vector database, embeddings, and LLM to avoid rebuilding on every query. Includes a response time tracker.
- **Transparency:** Displays the exact pages and retrieved text chunks used to generate the answer.
- **Detailed Error Handling:** Guides users if API keys or PDFs are missing.
- **Interactive Examples:** One-click example questions populate and query the system instantly.

## Technologies Used
- **Frontend:** Streamlit
- **Framework:** LangChain
- **Embeddings:** HuggingFace (`BAAI/bge-base-en-v1.5`)
- **Vector Database:** Chroma
- **LLM:** Google Gemini 2.5 Flash

## Folder Structure
```text
RAG_Chatbot_Streamlit/
├── app.py
├── requirements.txt
├── .env
├── .env.example
└── researchpaper.pdf
```

## Installation
1. Navigate to the project directory:
   ```bash
   cd RAG_Chatbot_Streamlit
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables by copying `.env.example` to `.env` and adding your Google API Key:
   ```bash
   cp .env.example .env
   # Add your key to .env
   ```

## Usage
Run the Streamlit application:
```bash
streamlit run app.py
```
Open your browser and navigate to the provided localhost URL (usually `http://localhost:8501`).

## Screenshots
*(Add screenshots of your application here)*
