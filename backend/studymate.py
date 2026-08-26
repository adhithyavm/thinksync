import os
import pymupdf as fitz  # PyMuPDF
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