'''!pip install -q \
langchain \
langchain-community \
langchain-text-splitters \
langchain-google-genai \
langchain-huggingface \
langchain-chroma \
chromadb \
sentence-transformers \
pypdf \
python-dotenv'''
# ---
from langchain_community.document_loaders import PyPDFLoader
# ---
import os

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"

print("API Key Loaded:", os.environ["GOOGLE_API_KEY"][:8] + "...")
# ---
from pathlib import Path

pdf_files = list(Path.home().rglob("*.pdf"))

print(f"Found {len(pdf_files)} PDF(s):\n")

for i, pdf in enumerate(pdf_files):
    print(f"{i+1}. {pdf}")
    
# ---
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "C:/Users/asus/Downloads/ResearchPaperBot/ResearchPaperBot/RAG_Chatbot_Streamlit/researchpaper.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"✅ PDF Loaded Successfully!")
print(f"Pages Loaded: {len(documents)}")
# ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")
print(chunks[0].page_content[:500])
# ---
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)
# ---
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ Chroma Vector Database Created Successfully!")
print(f"Stored {len(chunks)} chunks.")
# ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("✅ Retriever is ready!")

# ---
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

print("✅ Gemini initialized!")
# ---
def ask(question):
    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.
If the answer is not found in the context, reply:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content
# ---
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

print("✅ Gemini initialized successfully!")
# ---
def ask(question):
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.
If the answer is not found in the context, reply:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content
# ---
from google import genai
import os

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("Available models:\n")

for model in client.models.list():
    print(model.name)
# ---
from google import genai
import os

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("Available Models:\n")

for model in client.models.list():
    print(model.name)
# ---
model="models/gemini-2.5-flash"
# ---
from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

print("✅ Gemini initialized successfully!")
# ---
import langchain_google_genai
import google.genai

print("LangChain Google GenAI:", langchain_google_genai.__version__)
print("Google GenAI:", google.genai.__version__)
# ---
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"
# ---
from google import genai
import os

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

for m in client.models.list():
    if "generateContent" in m.supported_actions:
        print(m.name)
# ---
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Hello"
)

print(response.text)
# ---
from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

print("✅ LLM initialized!")
# ---
response = llm.invoke("Say Hello")

print(response.content)
# ---
def ask(question):
    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    # Combine retrieved text
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful AI assistant.

Answer the question only using the context below.
If the answer is not present in the context, say:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    # Handle both string and structured responses
    if isinstance(response.content, str):
        return response.content

    return "".join(
        item.get("text", "")
        for item in response.content
        if isinstance(item, dict)
    )
# ---
question = "What is the main objective of this research paper?"

answer = ask(question)

print("Question:")
print(question)

print("\nAnswer:")
print(answer)
# ---
docs = retriever.invoke("What is the main objective of this research paper?")

print("Retrieved Documents:", len(docs))

for i, doc in enumerate(docs):
    print(f"\n----- Document {i+1} -----")
    print(doc.page_content[:500])
# ---
# %who
# ---
print(vectorstore._collection.count())
# ---
docs = retriever.invoke("What is the main objective of this research paper?")

print("Retrieved:", len(docs))

for i, doc in enumerate(docs):
    print(f"\n----- Chunk {i+1} -----")
    print(doc.page_content[:500])
# ---
retriever = vectorstore.as_retriever()
# ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)
# ---
docs = retriever.invoke("What is the main objective of this research paper?")

print("Number of documents:", len(docs))

for i, doc in enumerate(docs):
    print(f"\n----- Document {i+1} -----")
    print(doc.page_content[:1000])
# ---
for i, chunk in enumerate(chunks):
    text = chunk.page_content.lower()

    if "abstract" in text:
        print("="*80)
        print("Chunk:", i)
        print(chunk.page_content[:1500])
        break
# ---
for i, chunk in enumerate(chunks):
    text = chunk.page_content.lower()

    if "introduction" in text:
        print("="*80)
        print("Chunk:", i)
        print(chunk.page_content[:1500])
        break
# ---
from langchain_chroma import Chroma

# Delete the old vector database from memory
try:
    del vectorstore
except:
    pass

# Create a fresh vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Vector DB rebuilt!")
print("Documents:", vectorstore._collection.count())
# ---
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)

print("Retriever ready!")
# ---
docs = retriever.invoke("What is the main objective of this research paper?")

for i, doc in enumerate(docs):
    print("=" * 80)
    print(f"Document {i+1}")
    print(doc.page_content[:800])
# ---
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 50,
        "lambda_mult": 0.2
    }
)
# ---
docs = retriever.invoke("Introduction")

for i, doc in enumerate(docs):
    print("="*80)
    print(doc.page_content[:800])
# ---
results = vectorstore.similarity_search(
    "Introduction",
    k=5
)

for i, r in enumerate(results):
    print("="*80)
    print(f"Result {i+1}")
    print(r.page_content[:800])
# ---
from langchain_chroma import Chroma
import shutil
import os

# Delete the old database if it exists
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

print("Old Chroma database deleted.")
# ---
import os

print("Current working directory:")
print(os.getcwd())

print("\nCan write here?:", os.access(os.getcwd(), os.W_OK))

print("\nDirectory contents:")
print(os.listdir())
# ---
import chromadb
print(chromadb.__version__)
# ---
import sys
print(sys.version)
# ---
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="test_collection_2026"
)

print("Success!")
print(vectorstore._collection.count())
# ---
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 50,
        "lambda_mult": 0.2
    }
)

print("✅ Retriever ready!")
# ---
docs = retriever.invoke("What is the main objective of this research paper?")

for i, doc in enumerate(docs):
    print("=" * 80)
    print(f"Document {i+1}")
    print(doc.page_content[:800])
# ---
for i in range(10):
    print("=" * 80)
    print(f"Chunk {i}")
    print(chunks[i].page_content[:600])
# ---
print(text_splitter)
# ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print("✅ Total chunks:", len(chunks))
# ---
for i in range(10):
    print("=" * 80)
    print(f"Chunk {i}")
    print(chunks[i].page_content[:600])
# ---
question = "What is the main objective of this research paper?"

docs = retriever.invoke(question)

for i, d in enumerate(docs):
    print("=" * 80)
    print(f"Document {i+1}")
    print(d.page_content[:1000])
# ---
# !pip install sentence-transformers
# ---
def ask(question):
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful research assistant.

Answer ONLY using the context below.
If the answer is not available in the context, reply:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content
# ---
print(ask("What is the main objective of this paper?"))
# ---
docs = retriever.invoke("What is the main objective of this paper?")

print("Number of documents:", len(docs))

for i, doc in enumerate(docs):
    print("="*80)
    print(f"Document {i+1}")
    print(doc.page_content[:500])
# ---
docs = retriever.invoke("What is the main objective of this paper?")

context = "\n\n".join([doc.page_content for doc in docs])

print(context[:5000])
# ---
import os

def ask(question):
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a research paper assistant.

Answer the question ONLY using the retrieved context.

If the answer is not present in the context, reply:
"I couldn't find the answer in the document."

Retrieved Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\n==============================")
    print("Answer")
    print("==============================")

    # Extract Gemini response text
    if isinstance(response.content, list):
        answer = "".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
        )
    else:
        answer = response.content

    print(answer)

    print("\n==============================")
    print("Retrieved Sources")
    print("==============================")

    seen = set()

    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "Unknown")

        key = (source, page)

        if key not in seen:
            seen.add(key)
            print(f"📄 {source}  |  Page {page + 1}")
# ---
docs = retriever.invoke("What is the main objective of this paper?")

print("Retrieved", len(docs), "documents\n")

for i, doc in enumerate(docs):
    print("=" * 80)
    print(f"Document {i+1}")
    print(doc.page_content[:1000])
# ---
from langchain_huggingface import HuggingFaceEmbeddings

mini_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ MiniLM Embedding Loaded")
# ---
from langchain_chroma import Chroma

mini_vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=mini_embeddings,
    collection_name="minilm_vectorstore"
)

print("Documents:", mini_vectorstore._collection.count())
# ---
mini_retriever = mini_vectorstore.as_retriever(
    search_kwargs={"k":5}
)

print("Retriever Ready")
# ---
docs = mini_retriever.invoke(
    "What is the main objective of this paper?"
)

for i, doc in enumerate(docs):
    print("="*80)
    print(f"Document {i+1}")
    print(doc.page_content[:500])
# ---
retriever = vectorstore.as_retriever(
    search_kwargs={"k":5}
)
# ---
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)
# ---
docs = mmr_retriever.invoke("What is the main objective of this paper?")

for i, doc in enumerate(docs):
    print("=" * 80)
    print(f"Document {i+1}")
    print(doc.page_content[:500])
# ---
docs = retriever.invoke("What is the main objective of this paper?")

for i, doc in enumerate(docs):
    print("=" * 60)
    print(f"Document {i+1}")
    print(doc.metadata)
# ---
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)
# ---
ask("What is the main objective of this paper?")
# ---
questions = [
    "What is the main objective of this paper?",
    "What is SKAO?",
    "What are neutron stars?",
    "What is the equation of state?",
    "Why are pulsars important?",
    "What observational techniques are discussed?",
    "What future work is mentioned?",
    "Who are the authors of the paper?",
    "How many pages does the paper contain?",
    "Summarize the conclusion of the paper."
]

for i, q in enumerate(questions, 1):
    print("=" * 80)
    print(f"Question {i}: {q}")
    ask(q)
    print("\n")
# ---

# ---
