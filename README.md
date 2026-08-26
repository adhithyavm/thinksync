# ThinkSync — AI Education Platform

[![Live App](https://img.shields.io/badge/Live_App-Vercel-black?style=for-the-badge&logo=vercel)](https://thinksynced.vercel.app)
[![API Status](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://thinksync-0kln.onrender.com)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)

ThinkSync is a multi-modal educational synthesis platform built for the **CIT Datathon**. It brings together Vector RAG (Retrieval-Augmented Generation) and Google Gemini to keep students, teachers, and parents aligned with personalized learning insights.

---

## 🌐 Live Deployments

- **Frontend (Web App):** [https://thinksynced.vercel.app](https://thinksynced.vercel.app)
- **Backend (FastAPI):** [https://thinksync-0kln.onrender.com](https://thinksync-0kln.onrender.com)

---

## Key Features

- **Multi-Modal Document Ingestion** — Upload student report cards, IEPs, or observation notes in PDF format for automated text extraction and vector indexing.
- **Vector RAG Engine** — Documents are chunked into semantic segments, indexed with vector embeddings, and retrieved via cosine similarity search.
- **Role-Based Synthesis** — Synthesizes structured insights tailored for teachers (pedagogy), parents (at-home guidance), and administrators (compliance & safety).
- **Secure Contextual Chat** — Interactive assistant that retrieves historical student documentation using vector search to answer targeted questions.
- **Multilingual Support** — Real-time translation into multiple languages (Tamil, Hindi, Spanish, French, German).
- **Educational Memory** — Persistent student tracking powered by Supabase PostgreSQL and Row Level Security.

---

## Tech Stack

### Frontend
- **Framework:** React 19 + TypeScript
- **Tooling:** Vite
- **Styling:** Tailwind CSS
- **Icons:** Lucide React
- **Routing:** React Router v7
- **Hosting:** Vercel

### Backend
- **Framework:** FastAPI (Python 3.11+)
- **LLM & Embeddings:** Google GenAI SDK (`google-genai`) — Gemini 2.5 / 3.6 Flash & Text Embeddings
- **Vector Retrieval:** In-memory Vector Store with Cosine Similarity Search
- **PDF Processing:** PyMuPDF (`pymupdf`)
- **Hosting:** Render

### Database & Storage
- **Database:** Supabase (PostgreSQL)
- **Storage:** Supabase Storage (`ai-ed` bucket)

---

## Getting Started

### Prerequisites
- Node.js (v18+)
- Python (v3.11+)
- Google Gemini API Key
- Supabase Project URL & Anon Key

### Local Installation

**1. Clone the repository:**
```bash
git clone https://github.com/adhithyavm/thinksync.git
cd thinksync
```

**2. Backend Setup:**
```bash
cd backend
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:
```env
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
```

Start the backend:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**3. Frontend Setup:**
```bash
cd ../ai-platform
npm install
```

Create a `.env` file in `ai-platform/`:
```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_URL=http://localhost:8000
```

Start the frontend:
```bash
npm run dev
```

---

## Project Structure

```
thinksync/
├── ai-platform/              # React + TypeScript frontend (Vite)
│   ├── src/
│   │   ├── pages/            # Capture, Dashboard, ChatBot
│   │   ├── services/         # API & Supabase services
│   │   └── lib/              # Supabase client configuration
│   └── vercel.json           # SPA rewrite routing for Vercel
├── backend/                  # FastAPI Python server
│   ├── main.py               # API endpoints (/process-rag, /chat, /translate)
│   ├── studymate.py          # Vector store, chunking, and Gemini integration
│   └── requirements.txt      # Python dependencies
└── .gitignore
```

---

## License

Distributed under the MIT License.

---

Built for the **CIT Datathon**.
