import os
import base64
import re
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==================================================
# CONFIG
# ==================================================
PAGE_TITLE = "Document Research Assistant"
PAGE_ICON = "🌐"
TEMP_UPLOAD_DIR = "./temp_uploads"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VISION_MODEL_NAME = "gemini-1.5-flash"

FALLBACK_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

# ==================================================
# THEME (Slate-Teal with 3D Glow Buttons)
# ==================================================
AQUA_THEME_CSS = """
<style>
    :root {
        --bg-main: #0F171A;
        --bg-card: #1A262B;
        --bg-user-card: #1E2D33;
        --sidebar-bg: #0B1013;
        --teal-accent: #3A7B7A;
        --teal-light: #529A98;
        --text-main: #E2E8F0;
        --text-muted: #94A3B8;
        --border-color: #26353C;
    }

    /* Main App Container */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-main);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Typography */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    /* File Uploader Container */
    [data-testid="stFileUploader"] {
        background-color: var(--bg-card);
        border: 1px dashed var(--teal-accent);
        border-radius: 10px;
        padding: 0.5rem;
    }

    /* Chat Messages - Unified Slate-Teal Aesthetic */
    [data-testid="stChatMessage"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.8rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    /* Target User Message specifically to distinguish visually */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: var(--bg-user-card) !important;
    }

    /* Fix Chat Input Field Styling */
    [data-testid="stChatInput"] {
        background-color: var(--bg-card) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--text-main) !important;
    }

    /* Button Polish - 3D Glow Effect */
    .stButton > button {
        background: linear-gradient(135deg, var(--teal-light), var(--teal-accent));
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        /* 3D and Glow combo */
        box-shadow: 0 4px 15px rgba(82, 154, 152, 0.35), 
                    inset 0 1px 1px rgba(255, 255, 255, 0.3), 
                    inset 0 -2px 4px rgba(0, 0, 0, 0.2) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        /* Enhanced glow on hover */
        box-shadow: 0 6px 20px rgba(82, 154, 152, 0.55), 
                    inset 0 1px 1px rgba(255, 255, 255, 0.4), 
                    inset 0 -2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px);
        /* Pressed state */
        box-shadow: 0 2px 8px rgba(82, 154, 152, 0.3), 
                    inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }

    /* Status Indicators / Alerts */
    .stAlert {
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }

    /* Thinking Animation Indicator */
    .thinking-dots {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 4px 0;
    }

    .thinking-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--teal-light);
        animation: pulse 1.2s infinite ease-in-out;
    }

    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes pulse {
        0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
        40% { opacity: 1; transform: scale(1.2); }
    }
</style>
"""

def apply_theme() -> None:
    st.markdown(AQUA_THEME_CSS, unsafe_allow_html=True)


# ==================================================
# ENVIRONMENT & SESSION STATE
# ==================================================
def load_api_keys():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if not groq_key:
        st.error("Missing GROQ_API_KEY in .env file.")
        st.stop()

    return groq_key, google_key


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


# ==================================================
# EMBEDDINGS & VISION OCR HELPERS
# ==================================================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def extract_text_via_gemini_vision(image_bytes, file_name, google_api_key):
    if not google_api_key:
        st.warning("GOOGLE_API_KEY missing. Image OCR skipped.")
        return ""

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    ext = os.path.splitext(file_name)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    llm = ChatGoogleGenerativeAI(
        model=VISION_MODEL_NAME,
        google_api_key=google_api_key,
        max_retries=3
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract and transcribe all readable text from this scanned image accurately. Do not add conversational commentary."},
            {"type": "image", "base64": base64_image, "mime_type": mime_type},
        ]
    )

    response = llm.invoke([message])
    return response.content if isinstance(response.content, str) else response.content[0]["text"]


# ==================================================
# INGESTION
# ==================================================
def process_and_store_files(uploaded_files, google_api_key):
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    all_docs = []

    for file in uploaded_files:
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        ext = os.path.splitext(file.name)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())

        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
            all_docs.extend(loader.load())

        elif ext in [".png", ".jpg", ".jpeg"]:
            try:
                extracted_text = extract_text_via_gemini_vision(file.getvalue(), file.name, google_api_key)
                if extracted_text.strip():
                    all_docs.append(Document(
                        page_content=extracted_text,
                        metadata={"source": file.name, "type": "vision"}
                    ))
            except Exception as e:
                st.warning(f"Vision transcription failed for '{file.name}': {e}")

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    embedding_model = get_embedding_model()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )


def load_existing_chroma():
    if os.path.exists(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embedding_model())
    return None


# ==================================================
# SIDEBAR UI
# ==================================================
def render_sidebar():
    with st.sidebar:
        st.title("Workspace")
        st.caption("Manage Document Knowledge Base")
        st.markdown("---")

        uploaded_files = st.file_uploader(
            "Upload Sources",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if st.button("Process & Index Files", use_container_width=True):
            if uploaded_files:
                with st.spinner("Indexing vector database..."):
                    st.session_state.vector_db = process_and_store_files(uploaded_files, st.session_state.google_api_key)
                    st.success("Indexing complete!")
            else:
                st.warning("Please upload at least one file.")

        st.markdown("---")
        st.subheader("Database Status")
        if st.session_state.vector_db:
            try:
                doc_count = st.session_state.vector_db._collection.count()
                st.info(f"Indexed Chunks: **{doc_count}**")
            except Exception:
                st.info("Database Active")
        else:
            st.warning("No Database Loaded")


# ==================================================
# CHAT UI
# ==================================================
def render_chat_history():
    for idx, message in enumerate(st.session_state.messages):
        avatar = "👤" if message["role"] == "user" else "🤖"
        
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            
            #follow-up button
            is_last_message = (idx == len(st.session_state.messages) - 1)
            if is_last_message and message.get("follow_up"):
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"✨ Ask: {message['follow_up']}", key=f"followup_{idx}"):
                    st.session_state.pending_question = message["follow_up"]
                    st.rerun()


def build_prompt(context, user_question):
    return f"""
You are an expert document research assistant.

Answer the user's question using ONLY the context provided below.
If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

At the very end of your response, always suggest exactly ONE highly relevant follow-up question that the user might want to ask next based on your answer.
Enclose this question in XML tags like this: <followup>Your question here?</followup>

Do not fabricate information.

Context:
{context}

User Question:
{user_question}
"""


THINKING_INDICATOR = """
<div class="thinking-dots"><span></span><span></span><span></span></div>
"""


def get_groq_response(prompt, groq_api_key, response_placeholder):
    preferred_model = os.getenv("GROQ_MODEL")
    active_models = [preferred_model] if preferred_model else FALLBACK_GROQ_MODELS

    full_response = ""
    last_error = None

    response_placeholder.markdown(THINKING_INDICATOR, unsafe_allow_html=True)

    for model_name in active_models:
        try:
            llm = ChatGroq(model=model_name, groq_api_key=groq_api_key, temperature=0.2)
            first_chunk_received = False

            for chunk in llm.stream(prompt):
                text_chunk = chunk.content if isinstance(chunk.content, str) else chunk.content[0]["text"]
                full_response += text_chunk
                first_chunk_received = True
                
                display_text = re.sub(r'<followup>.*?(</followup>)?', '', full_response, flags=re.IGNORECASE | re.DOTALL)
                response_placeholder.markdown(display_text + "▌")

            if not first_chunk_received:
                raise ValueError("Empty response from model")

            # Final cleanup of display text
            display_text = re.sub(r'<followup>.*?</followup>', '', full_response, flags=re.IGNORECASE | re.DOTALL).strip()
            response_placeholder.markdown(display_text)
            
            return full_response, True

        except Exception as e:
            last_error = f"{model_name}: {e}"
            full_response = ""
            continue

    return f"Error connecting to Groq API. Last error: {last_error}", False


def handle_user_question(user_question, groq_api_key):
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar="🤖"):
        if not st.session_state.vector_db:
            res_content = "Please upload and process documents first."
            st.markdown(res_content)
            st.session_state.messages.append({"role": "assistant", "content": res_content})
            return

        with st.spinner("Searching documents..."):
            retriever = st.session_state.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10}
            )
            retrieved_docs = retriever.invoke(user_question)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            prompt = build_prompt(context, user_question)

        response_placeholder = st.empty()
        raw_answer, success = get_groq_response(prompt, groq_api_key, response_placeholder)

        if not success:
            response_placeholder.error(raw_answer)
            st.session_state.messages.append({"role": "assistant", "content": raw_answer})
            return

        # Extract follow-up question
        follow_up_match = re.search(r'<followup>(.*?)</followup>', raw_answer, re.IGNORECASE | re.DOTALL)
        follow_up_q = follow_up_match.group(1).strip() if follow_up_match else None
        
        # Clean the final display text
        clean_answer = re.sub(r'<followup>.*?</followup>', '', raw_answer, flags=re.IGNORECASE | re.DOTALL).strip()
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": clean_answer,
            "follow_up": follow_up_q
        })
        
        st.rerun()


# ==================================================
# MAIN
# ==================================================
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    apply_theme()

    groq_api_key, google_api_key = load_api_keys()
    st.session_state.google_api_key = google_api_key

    init_session_state()
    if st.session_state.vector_db is None:
        st.session_state.vector_db = load_existing_chroma()

    render_sidebar()

    st.title("Document Research Companion")
    st.caption("Powered by Groq & Local Embeddings")

    # Handle pending follow-up question from button click
    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        handle_user_question(q, groq_api_key)

    render_chat_history()

    if user_question := st.chat_input("Ask a question about your documents..."):
        handle_user_question(user_question, groq_api_key)


if __name__ == "__main__":
    main()