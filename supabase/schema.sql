-- ============================================================
--  ThinkSync Database Schema & Storage Configuration
-- ============================================================

-- Table 1: kids
-- Stores unique student identifiers and full names
CREATE TABLE IF NOT EXISTS public.kids (
    name_key  TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Table 2: kid_documents
-- Stores uploaded educational documents and raw extracted content
CREATE TABLE IF NOT EXISTS public.kid_documents (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kid_name_key TEXT NOT NULL REFERENCES public.kids(name_key) ON DELETE CASCADE,
    file_url     TEXT,
    file_type    TEXT DEFAULT 'application/pdf',
    raw_content  TEXT,
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- Table 3: insights
-- Stores multi-stakeholder synthesized insights generated via RAG
CREATE TABLE IF NOT EXISTS public.insights (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id      BIGINT NOT NULL REFERENCES public.kid_documents(id) ON DELETE CASCADE,
    summary_teacher  TEXT,
    summary_parent   TEXT,
    summary_admin    TEXT,
    priority_level   TEXT DEFAULT 'low' CHECK (priority_level IN ('low', 'medium', 'high')),
    created_at       TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
--  Row Level Security (RLS) Policies
-- ============================================================
ALTER TABLE public.kids          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.kid_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.insights      ENABLE ROW LEVEL SECURITY;

CREATE POLICY "allow_all_kids"     ON public.kids          FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_docs"     ON public.kid_documents FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_insights" ON public.insights      FOR ALL USING (true) WITH CHECK (true);

-- ============================================================
--  Supabase Storage Bucket Policies (Bucket name: 'ai-ed')
-- ============================================================
-- Ensure the 'ai-ed' bucket is created as Public in Supabase Storage UI
CREATE POLICY "storage_insert" ON storage.objects FOR INSERT TO anon, authenticated WITH CHECK (bucket_id = 'ai-ed');
CREATE POLICY "storage_select" ON storage.objects FOR SELECT TO anon, authenticated USING (bucket_id = 'ai-ed');
CREATE POLICY "storage_delete" ON storage.objects FOR DELETE TO anon, authenticated USING (bucket_id = 'ai-ed');
