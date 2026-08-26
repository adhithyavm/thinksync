import os
import logging
import pymupdf as fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# load the env vars
load_dotenv()

# config settings
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

def get_pdf_text(path):
    """ extracts all text from a given pdf file path """
    content = []
    with fitz.open(path) as pdf:
        for pg in pdf:
            content.append(pg.get_text())
    return "\n".join(content)

def generate_gemini_response(prompt: str, temperature: float = 0.2) -> str:
    """ Generate response using official Google Generative AI SDK with model fallbacks """
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    last_err = None
    for m_name in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name=m_name,
                generation_config={"temperature": temperature}
            )
            res = model.generate_content(prompt)
            if res and res.text:
                return res.text
        except Exception as e:
            logger.warning(f"Model {m_name} failed: {e}")
            last_err = e
            continue
            
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")