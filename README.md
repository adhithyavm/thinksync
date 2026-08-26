# ThinkSync — AI Education Platform

[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)

ThinkSync is a multi-modal AI synthesis platform built for the **CIT Datathon**. It brings together RAG (Retrieval-Augmented Generation) and Google Gemini to help students, teachers, and parents stay aligned with personalized educational insights.

---

## Features

- **Educational Memory** — Student data is stored in Supabase so every interaction is context-aware and personalized over time.
- **RAG-Powered Chatbot** — Ask questions about uploaded study materials; answers are grounded via FAISS vector search.
- **Role-Based Dashboards** — Separate views for students (progress), teachers (summaries + alerts), and parents (growth tracking).
- **Multi-lingual Support** — AI responses and chatbot replies can be translated into Tamil, Hindi, Spanish, French, and German.
- **Multi-modal Ingestion** — Upload PDFs; the backend parses, embeds, and synthesizes them automatically.

---

## Tech Stack

### Frontend
- **React 19** + TypeScript
- **Vite** (build tool)
- **Tailwind CSS** (styling)
- **Lucide React** (icons)
- **React Router v7** (routing)

### Backend
- **FastAPI** (Python 3.11+)
- **LangChain** (AI orchestration)
- **Google Gemini 1.5 Flash** (LLM)
- **FAISS** (vector store, CPU)
- **HuggingFace all-MiniLM-L6-v2** (embeddings)
- **PyMuPDF** (PDF parsing)

### Database & Auth
- **Supabase** (PostgreSQL + Storage + Auth)
- **Python Dotenv** (environment management)

---

## Getting Started

### Prerequisites
- Node.js v18+
- Python 3.11+
- Google Gemini API Key
- Supabase Project URL & Anon Key

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/adhithyavm/thinksync.git
cd thinksync
```

**2. Backend setup:**
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:
```env
GEMINI_API_KEY=your_gemini_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
```

Start the backend server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**3. Frontend setup:**
```bash
cd ../ai-platform
npm install
```

Create a `.env` file inside `ai-platform/`:
```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_URL=http://localhost:8000
```

Start the dev server:
```bash
npm run dev
```

---

## Project Structure

```
thinksync/
├── ai-platform/              # React + TypeScript frontend
│   └── src/
│       ├── pages/            # Capture, Dashboard, ChatBot
│       ├── services/         # API & Supabase integration
│       └── lib/              # Supabase client
├── backend/                  # FastAPI Python server
│   ├── main.py               # API entry point (RAG, chat, translation)
│   ├── studymate.py          # Vector store & PDF extraction logic
│   └── requirements.txt      # Python dependencies
└── .gitignore
```

---

## License

Distributed under the MIT License.

---

Built for the CIT Datathon.
