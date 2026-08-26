import os
import pymupdf as fitz  # PyMuPDF (replaces deprecated 'import fitz')
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# load the env vars
load_dotenv()

# config settings
API_KEY = os.getenv("GEMINI_API_KEY")
ACTIVE_MODEL = "gemini-1.5-flash"

def get_pdf_text(path):
    """ extracts all text from a given pdf file path """
    content = []
    with fitz.open(path) as pdf:
        for pg in pdf:
            content.append(pg.get_text())
    return "\n".join(content)

def create_vs(files):
    """ builds the faiss vector store from a list of pdfs """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    all_chunks = []

    for f in files:
        raw_data = get_pdf_text(f)
        all_chunks.extend(text_splitter.split_text(raw_data))

    # using Google's text-embedding-004 (current stable model)
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )
    return FAISS.from_texts(all_chunks, embedding=embed_model)