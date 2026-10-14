# pyrefly: ignore [missing-import]
import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# --- Page Config & Styling ---
st.set_page_config(
    page_title="Research Paper RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .stButton>button {
        font-weight: bold;
        border-radius: 8px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #dee2e6;
    }
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        font-size: 12px;
        color: #6c757d;
        background: white;
        z-index: 100;
        width: 100%;
        text-align: right;
        border-top: 1px solid #e9ecef;
    }
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "researchpaper.pdf")

@st.cache_resource(show_spinner=False)
def load_rag():
    """Initializes the RAG pipeline components. Cached so it doesn't rebuild on every run."""
    # 1. Check API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "Error: GOOGLE_API_KEY not found in environment variables. Please check your .env file or add the variable."
        
    # 2. Check PDF
    if not os.path.exists(PDF_PATH):
        return None, f"Error: '{PDF_PATH}' not found."

    try:
        persist_dir = os.path.join(BASE_DIR, "chroma_db")
        
        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

        # Vector Database
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print("Loaded existing Chroma database.")
            vectorstore = Chroma(
                embedding_function=embeddings,
                collection_name="streamlit_rag_collection_v2",
                persist_directory=persist_dir
            )
            # Fetch num_pages and num_chunks without reloading PDF
            try:
                num_chunks = vectorstore._collection.count()
                metadata_list = vectorstore.get(include=['metadatas'])['metadatas']
                # Unique page numbers (PyPDFLoader stores 'page' in metadata)
                num_pages = len(set(m.get("page", 0) for m in metadata_list if m))
            except Exception as e:
                import logging
                logging.error(f"Error fetching metadata: {str(e)}")
                num_chunks = "Unknown"
                num_pages = "Unknown"
        else:
            print("Created new Chroma database.")
            # Load PDF
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()
            num_pages = len(documents)

            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            num_chunks = len(chunks)

            # Create and persist Vector Database
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="streamlit_rag_collection_v2",
                persist_directory=persist_dir
            )

        # Retriever (MMR)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20
            }
        )

        # LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key
        )
        
        pipeline = {
            "retriever": retriever,
            "llm": llm,
            "num_pages": num_pages,
            "num_chunks": num_chunks
        }
        return pipeline, None
    except Exception as e:
        return None, f"An error occurred during initialization: {str(e)}"

def get_response(question, pipeline):
    """Retrieves context and generates an answer using the LLM."""
    retriever = pipeline["retriever"]
    llm = pipeline["llm"]
    
    # Retrieve relevant chunks
    try:
        docs = retriever.invoke(question)
    except Exception as e:
        import logging
        logging.error(f"Retriever Error: {str(e)}")
        return "Unable to retrieve documents right now. Please try again later.", []

    # Combine retrieved text
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.
If the answer is not found in the context, reply:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}
"""
    try:
        response = llm.invoke(prompt)

        # Handle both string and structured responses
        if isinstance(response.content, str):
            answer = response.content
        else:
            answer = "".join(
                item.get("text", "")
                for item in response.content
                if isinstance(item, dict)
            )
    except Exception as e:
        import logging
        logging.error(f"LLM Error: {str(e)}")
        answer = "Unable to generate a response right now. Please try again later."
        
    return answer, docs

def display_sources(docs):
    """Displays the retrieved sources in the UI using cards and expanders."""
    seen = set()
    sources = []
    
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "Unknown")
        key = (source, page)
        
        if key not in seen:
            seen.add(key)
            sources.append((source, page))
            
    if sources:
        st.markdown("#### 📚 Retrieved Sources")
        # Display sources as horizontal cards
        cols = st.columns(len(sources))
        for idx, (source, page) in enumerate(sources):
            with cols[idx % len(cols)]:
                with st.container(border=True):
                    page_num = page + 1 if isinstance(page, int) else page
                    st.markdown(f"**📄 {source}**")
                    st.markdown(f"Page {page_num}")
                    
        # Expandable Retrieved Context
        with st.expander("▼ Retrieved Context"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}**")
                st.info(doc.page_content)
                st.markdown("---")
    else:
        st.info("No sources retrieved.")

def display_sidebar(pipeline, status_msg, is_error=False):
    """Displays the sidebar with clear chat and metrics."""
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("---")
        
        st.markdown("### 📄 Document Information")
        if pipeline:
            col1, col2 = st.columns(2)
            col1.metric("Total Pages", pipeline['num_pages'])
            col2.metric("Total Chunks", pipeline['num_chunks'])
            
            st.markdown("#### Configuration")
            st.markdown("- **Embedding Model:** BAAI/bge-base-en-v1.5")
            st.markdown("- **Retriever:** MMR")
            st.markdown("- **Vector Database:** Chroma")
            st.markdown("- **LLM:** Gemini 2.5 Flash")
        
        st.markdown("---")
        st.markdown("### ℹ️ Status")
        if is_error:
            st.error(status_msg)
        else:
            st.success(status_msg)

def display_footer():
    st.markdown("""
    <div class="footer">
        Built With &bull; Streamlit &bull; LangChain &bull; Chroma &bull; Google Gemini &bull; HuggingFace Embeddings
    </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Setup loading
    pipeline = None
    error = None
    with st.spinner("Initializing RAG pipeline (this might take a moment on first run)..."):
        pipeline, error = load_rag()

    # Sidebar
    if error:
        display_sidebar(None, error, is_error=True)
    else:
        display_sidebar(pipeline, "Ready")

    # Main Area
    st.title("Research Paper RAG Chatbot")
    st.markdown("Ask any questions about the research paper, and the AI will find the answer from the document.")
    
    if error:
        st.error(error)
        st.info("If your API key is missing, please check the setup instructions in the README.")
        return

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if "I couldn't find the answer" in message["content"] or "I don't know" in message["content"]:
                    st.info("I couldn't find the answer in the document.")
                else:
                    with st.container(border=True):
                        st.markdown("### 📝 Answer")
                        st.markdown("---")
                        st.write(message["content"])
                
                if message.get("docs"):
                    display_sources(message["docs"])
                if message.get("response_time"):
                    st.caption(f"⚡ Response Time: {message['response_time']:.2f} seconds")
            else:
                st.markdown(message["content"])

    # Example Questions (only show if no messages yet)
    selected_example = None
    if not st.session_state.messages:
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What are neutron stars?",
            "What is SKAO?",
            "Why are pulsars important?",
            "What observational techniques are discussed?",
            "Summarize the paper."
        ]
        # Use columns to lay out buttons horizontally
        cols = st.columns(len(example_questions))
        for idx, q in enumerate(example_questions):
            if cols[idx].button(q, key=f"ex_{idx}"):
                selected_example = q
                
    # Chat Input
    prompt = st.chat_input("Ask a question about the document...")
    
    # Handle Input (Either from chat input or example button)
    user_query = prompt or selected_example

    if user_query:
        if not user_query.strip():
            st.warning("Please enter a valid question.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                start_time = time.time()
                answer, docs = get_response(user_query, pipeline)
            
            with st.spinner("🤖 Generating answer..."):
                end_time = time.time()
                response_time = end_time - start_time
                
                # Render answer UI
                if "I couldn't find the answer" in answer or "I don't know" in answer:
                    st.info("I couldn't find the answer in the document.")
                else:
                    with st.container(border=True):
                        st.markdown("### 📝 Answer")
                        st.markdown("---")
                        st.write(answer)
                
                display_sources(docs)
                st.caption(f"⚡ Response Time: {response_time:.2f} seconds")

        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "docs": docs,
            "response_time": response_time
        })

    # Footer
    display_footer()

if __name__ == "__main__":
    main()
