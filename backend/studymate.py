import os
import logging
import numpy as np
import pymupdf as fitz  # PyMuPDF
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

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

        # Attempt to create dense embeddings via Google GenAI API
        embedded_vectors = []
        embedding_models = [
            "models/text-embedding-004",
            "models/embedding-001"
        ]
        
        success = False
        for em_model in embedding_models:
            try:
                temp_vectors = []
                for chunk in self.chunks:
                    res = genai.embed_content(
                        model=em_model,
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    temp_vectors.append(res['embedding'])
                self.embeddings = np.array(temp_vectors)
                self.active_embed_model = em_model
                success = True
                logger.info(f"VectorStore successfully indexed {len(self.chunks)} chunks using {em_model}")
                break
            except Exception as e:
                logger.warning(f"Embedding model {em_model} unavailable: {e}")
                continue

        # If external embedding API is unreachable/restricted, use high-precision TF-IDF vector embeddings
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
                res = genai.embed_content(
                    model=self.active_embed_model,
                    content=query,
                    task_type="retrieval_query"
                )
                query_vec = np.array(res['embedding']).reshape(1, -1)
                similarities = cosine_similarity(query_vec, self.embeddings).flatten()
                top_indices = np.argsort(similarities)[::-1][:k]
                return [self.chunks[idx] for idx in top_indices]
            except Exception as e:
                logger.warning(f"Query embedding failed ({e}), falling back to local text similarity")
                # Fallback on the fly
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