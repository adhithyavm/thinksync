import os
import pymupdf  # PyMuPDF (modern import)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
ACTIVE_MODEL = "gemini-1.5-flash"


def get_pdf_text(path):
    """Extract all text from a PDF file."""
    pages = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            pages.append(page.get_text())
    return "\n".join(pages)


def create_vs(files):
    """Build a FAISS vector store from a list of PDF paths."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = []

    for f in files:
        raw = get_pdf_text(f)
        chunks.extend(splitter.split_text(raw))

    # Gemini embeddings via API — no local model download, no torch dependency
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    return FAISS.from_texts(chunks, embedding=embeddings)