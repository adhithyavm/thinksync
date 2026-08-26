import os
import logging
import numpy as np
import pymupdf as fitz  # PyMuPDF
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

def get_pdf_text(path: str) -> str:
    """ extracts all text from a given pdf file path """
    content = []
    with fitz.open(path) as pdf:
        for pg in pdf:
            content.append(pg.get_text())
    return "\n".join(content)

def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """ splits raw text into overlapping semantic chunks """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

class SimpleVectorStore:
    """ High-performance Vector Store supporting dense embeddings & cosine similarity """
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.embeddings = None
        self.use_fallback = False
        self._build_index()

    def _build_index(self):
        if not self.chunks:
            return

        success = False
        if client:
            embedding_models = [
                "text-embedding-004",
                "embedding-001"
            ]
            for em_model in embedding_models:
                try:
                    temp_vectors = []
                    for chunk in self.chunks:
                        res = client.models.embed_content(
                            model=em_model,
                            contents=chunk,
                        )
                        # Extract embedding values from google.genai response
                        vec = res.embeddings[0].values if hasattr(res, 'embeddings') and res.embeddings else res.embedding.values
                        temp_vectors.append(vec)
                    self.embeddings = np.array(temp_vectors)
                    self.active_embed_model = em_model
                    success = True
                    logger.info(f"VectorStore successfully indexed {len(self.chunks)} chunks using {em_model}")
                    break
                except Exception as e:
                    logger.warning(f"Embedding model {em_model} unavailable: {e}")
                    continue

        if not success:
            logger.info("Using local TF-IDF vector space for vector similarity search.")
            self.use_fallback = True
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.embeddings = self.vectorizer.fit_transform(self.chunks)

    def similarity_search(self, query: str, k: int = 4) -> list[str]:
        """ Performs cosine similarity search against indexed vector space and returns top-k chunks """
        if not self.chunks or self.embeddings is None:
            return []

        k = min(k, len(self.chunks))

        if self.use_fallback:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.embeddings).flatten()
            top_indices = np.argsort(similarities)[::-1][:k]
            return [self.chunks[idx] for idx in top_indices]
        else:
            try:
                res = client.models.embed_content(
                    model=self.active_embed_model,
                    contents=query,
                )
                vec = res.embeddings[0].values if hasattr(res, 'embeddings') and res.embeddings else res.embedding.values
                query_vec = np.array(vec).reshape(1, -1)
                similarities = cosine_similarity(query_vec, self.embeddings).flatten()
                top_indices = np.argsort(similarities)[::-1][:k]
                return [self.chunks[idx] for idx in top_indices]
            except Exception as e:
                logger.warning(f"Query embedding error ({e}), falling back to local text similarity")
                vec = TfidfVectorizer(stop_words='english')
                all_texts = [query] + self.chunks
                tfidf = vec.fit_transform(all_texts)
                sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
                top_indices = np.argsort(sims)[::-1][:k]
                return [self.chunks[idx] for idx in top_indices]

def create_vector_store(file_paths: list[str]) -> SimpleVectorStore:
    """ Reads PDF files, splits them into semantic chunks, and builds a vector store """
    all_chunks = []
    for fp in file_paths:
        text = get_pdf_text(fp)
        chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=100)
        all_chunks.extend(chunks)
    return SimpleVectorStore(all_chunks)

def generate_gemini_response(prompt: str, temperature: float = 0.2) -> str:
    """ Generate response using modern official google-genai SDK with model fallbacks """
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not set or client initialization failed")

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-3.6-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
    ]
    last_err = None
    for m_name in models_to_try:
        try:
            res = client.models.generate_content(
                model=m_name,
                contents=prompt,
            )
            if res and res.text:
                return res.text
        except Exception as e:
            logger.warning(f"Model {m_name} failed: {e}")
            last_err = e
            continue
            
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")