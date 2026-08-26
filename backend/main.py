from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import json
import logging
from studymate import (
    get_pdf_text,
    split_text_into_chunks,
    create_vector_store,
    SimpleVectorStore,
    generate_gemini_response
)
from dotenv import load_dotenv
from supabase import create_client, Client

# standard logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SB_URL, SB_KEY)

app = FastAPI(title="ThinkSync RAG Backend")

# enable dev CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# make sure uploads dir exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.post("/process-rag")
async def process_rag(student_name: str = Form(...), file: UploadFile = File(...)):
    try:
        # save local copy of file
        f_path = os.path.join("uploads", file.filename)
        with open(f_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # grab raw text for db storage
        full_text = get_pdf_text(f_path)

        # store kid info in Supabase
        k_key = student_name.lower().replace(" ", "_").strip()
        supabase.table("kids").upsert(
            {"name_key": k_key, "full_name": student_name}
        ).execute()

        # link document to kid
        payload = {
            "kid_name_key": k_key,
            "file_url": f"/uploads/{file.filename}",
            "file_type": file.content_type or "application/pdf",
            "raw_content": full_text
        }
        res = supabase.table("kid_documents").insert(payload).execute()
        doc_id = res.data[0]["id"]

        # === RAG: Vector Indexing & Similarity Retrieval ===
        # 1. Chunk document & build vector embeddings in Vector Store
        vector_store = create_vector_store([f_path])
        
        # 2. Similarity search for the most relevant context chunks
        retrieved_chunks = vector_store.similarity_search(query=f"{student_name} performance, behavior, academics, notes", k=4)
        c_text = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else full_text[:4000]

        # 3. Augmented Generation with retrieved context
        prompt = f"""You are an expert educational assistant synthesizing student observations.
Student: {student_name}

Retrieved Context Chunks (via Vector Search):
{c_text}

Instructions:
- Use only the provided context to extract verified insights.
- Return a valid JSON object strictly matching this format without markdown code fences or extra text:
{{
  "teacher_summary": "detailed academic analysis and pedagogical guidance for the teacher",
  "parent_summary": "encouraging update with a specific home activity suggestion for the parent",
  "admin_summary": "safety, logistics, and compliance perspective for school administrators",
  "priority": "low"
}}
Priority levels:
- high: safety risks, critical interventions, or major academic failure
- medium: noticeable behavioral changes or minor academic decline
- low: general updates, consistent progress, or positive feedback
"""

        text_out = generate_gemini_response(prompt, temperature=0.2)
        text_out = text_out.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(text_out)
        except Exception:
            parsed = json.loads(text_out.replace("\n", " "))

        # save synthesized insights to database
        insight_data = {
            "document_id": doc_id,
            "summary_teacher": parsed.get("teacher_summary", ""),
            "summary_parent": parsed.get("parent_summary", ""),
            "summary_admin": parsed.get("admin_summary", ""),
            "priority_level": parsed.get("priority", "low")
        }
        supabase.table("insights").insert(insight_data).execute()

        return parsed

    except Exception as err:
        logger.error(f"Error in RAG processing: {str(err)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process document via RAG")

class ChatRequest(BaseModel):
    student_name: str
    message: str

@app.post("/chat")
async def handle_chat(req: ChatRequest):
    try:
        s_name = req.student_name.strip()
        n_key = s_name.lower().replace(" ", "_")
        
        # find student in Supabase
        user_data = supabase.table("kids").select("*").or_(f"name_key.eq.{n_key},full_name.ilike.{s_name}").execute()
        
        if not user_data.data:
            return {"reply": "Student not found in our records.", "blocked": True}

        info = user_data.data[0]
        actual_name = info["full_name"]
        actual_key = info["name_key"]

        # pull all student documents from database
        docs = supabase.table("kid_documents").select("raw_content").eq("kid_name_key", actual_key).execute()
        all_doc_texts = [d["raw_content"] for d in docs.data if d.get("raw_content")]

        if not all_doc_texts:
            # fallback check in insights
            insights = supabase.table("insights").select("*").eq("document_id", actual_key).execute()
            if not insights.data:
                return {"reply": f"No data found for {actual_name}.", "blocked": False}
            i = insights.data[0]
            context_chunks = [f"Teacher Notes: {i.get('summary_teacher')}", f"Parent Notes: {i.get('summary_parent')}"]
        else:
            # === RAG: Vector Indexing & Semantic Search for Chat Query ===
            all_chunks = []
            for doc_txt in all_doc_texts:
                all_chunks.extend(split_text_into_chunks(doc_txt, chunk_size=500, chunk_overlap=100))
            
            chat_vector_store = SimpleVectorStore(all_chunks)
            context_chunks = chat_vector_store.similarity_search(query=req.message, k=4)

        retrieved_context = "\n\n---\n\n".join(context_chunks)

        chat_prompt = f"""
Role: Intelligent Education Assistant for {actual_name}
Retrieved Knowledge Context:
{retrieved_context}

User Query: "{req.message}"

Rules:
1. Answer the query strictly based on the retrieved knowledge context.
2. Be warm, supportive, and professional.
3. If trying to access unauthorized or unrelated records, start with [BLOCKED].
4. If the information is not contained in the records, state clearly that it is not yet recorded.
"""
        
        text_reply = generate_gemini_response(chat_prompt, temperature=0.1).strip()
        is_blocked = text_reply.startswith("[BLOCKED]")
        return {"reply": text_reply.replace("[BLOCKED]", "").strip(), "blocked": is_blocked}

    except Exception as e:
        logger.error(f"Chat RAG error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat server error")

class TransReq(BaseModel):
    text: str
    target_lang: str

@app.post("/translate")
async def do_translation(req: TransReq):
    try:
        p = f"Translate the following text into {req.target_lang}. Keep tone and structure intact.\n\nTEXT:\n{req.text}"
        res_text = generate_gemini_response(p, temperature=0.0).strip()
        return {"translated_text": res_text}
    except Exception as e:
        logger.error(f"Translate error: {str(e)}")
        raise HTTPException(status_code=500, detail="Translation failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)