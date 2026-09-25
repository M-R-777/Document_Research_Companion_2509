# 📄 Document Research Companion

An interactive RAG (Retrieval-Augmented Generation) application designed to search, analyze, and research across your documents. Upload PDFs, text files, or scanned document images, and ask questions through a real-time streaming chat interface.

---

## 🎬 Project Preview

![Application Preview](https://s4.ezgif.com/tmp/ezgif-4171d5bac8c4ff4d.gif)

---

## ✨ Features

* **Multi-Format Ingestion:** Process PDFs (`.pdf`), text files (`.txt`, `.md`), and scanned images (`.png`, `.jpg`, `.jpeg`).
* **Multimodal Vision OCR:** Automatically transcribes text from scanned document pictures using Gemini 1.5 Flash Vision.
* **Fast Local Vector Search:** Splits documents into 1,000-character chunks and indexes them locally with `ChromaDB` and `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
* **Interactive Chat with Guided Follow-ups:** Answers queries strictly based on provided context using Groq Llama models and automatically suggests a contextually relevant follow-up question.
* **Slate-Teal Dark UI:** Custom Streamlit theme built with custom CSS cards and 3D glowing buttons.

---

## 🛠️ How It Works

1. **Upload & Parse:** Uploaded files are ingested and converted into clean text documents.
2. **Chunk & Index:** Text is chunked with `RecursiveCharacterTextSplitter` (1,000 chunk size, 200 overlap) and embedded into a local `ChromaDB` vector database.
3. **Context Retrieval:** User questions trigger an MMR (Maximal Marginal Relevance) similarity search to extract the 4 most relevant text chunks.
4. **LLM Generation:** The context and question are passed to Groq Llama models to generate a streamed answer alongside an XML-parsed follow-up query.

---

## 🚀 Local Setup & Run Instructions

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your machine.

### 2. Clone Repository & Install Dependencies

```bash
git clone [https://github.com/M-R-777/Document_Research_Companion_2509.git](https://github.com/M-R-777/Document_Research_Companion_2509.git)
cd your-repo-name

python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt