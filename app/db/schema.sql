-- =============================================================
-- BKI Hull RAG — Initial Database Schema
-- Run this in your Supabase SQL Editor to create all tables.
-- =============================================================

-- Enable pgvector extension for embedding storage
create extension if not exists vector;

-- -----------------------------------------------------------
-- 1. documents — metadata about ingested source documents
-- -----------------------------------------------------------
create table if not exists documents (
    id         uuid primary key default gen_random_uuid(),
    doc_key    text unique not null,
    title      text not null,
    edition    text,
    source_file text,
    created_at timestamptz default now()
);

-- -----------------------------------------------------------
-- 2. chunks — document chunks with embeddings
--    Embedding dimension 1024 is the default for bge-m3.
--    Adjust if using a different embedding model.
-- -----------------------------------------------------------
create table if not exists chunks (
    id              uuid primary key default gen_random_uuid(),
    document_id     uuid references documents(id) on delete cascade,
    content         text not null,
    content_type    text not null default 'text',
    section_number  text,
    section_title   text,
    subsection      text,
    page_start      int,
    page_end        int,
    metadata        jsonb default '{}'::jsonb,
    embedding       vector(1024),
    created_at      timestamptz default now()
);

-- -----------------------------------------------------------
-- 3. chat_sessions — conversation session tracking
-- -----------------------------------------------------------
create table if not exists chat_sessions (
    id         uuid primary key default gen_random_uuid(),
    title      text,
    interface  text default 'cli',
    created_at timestamptz default now()
);

-- -----------------------------------------------------------
-- 4. chat_messages — individual messages in a session
-- -----------------------------------------------------------
create table if not exists chat_messages (
    id         uuid primary key default gen_random_uuid(),
    session_id uuid references chat_sessions(id) on delete cascade,
    role       text not null,
    content    text not null,
    language   text,
    metadata   jsonb default '{}'::jsonb,
    created_at timestamptz default now()
);

-- -----------------------------------------------------------
-- 5. retrieval_logs — debug/audit log for retrieval queries
-- -----------------------------------------------------------
create table if not exists retrieval_logs (
    id                 uuid primary key default gen_random_uuid(),
    session_id         uuid references chat_sessions(id) on delete set null,
    query              text not null,
    rewritten_query    text,
    language           text,
    intent             text,
    retrieved_chunk_ids uuid[],
    scores             jsonb default '{}'::jsonb,
    created_at         timestamptz default now()
);

-- -----------------------------------------------------------
-- 6. calculation_logs — audit log for calculation operations
-- -----------------------------------------------------------
create table if not exists calculation_logs (
    id                     uuid primary key default gen_random_uuid(),
    session_id             uuid references chat_sessions(id) on delete set null,
    formula_source_chunk_id uuid references chunks(id) on delete set null,
    formula_text           text,
    input_variables        jsonb default '{}'::jsonb,
    missing_variables      jsonb default '[]'::jsonb,
    result                 jsonb default '{}'::jsonb,
    created_at             timestamptz default now()
);

-- -----------------------------------------------------------
-- Indexes for common query patterns
-- -----------------------------------------------------------
create index if not exists idx_chunks_document_id on chunks(document_id);
create index if not exists idx_chunks_content_type on chunks(content_type);
create index if not exists idx_chunks_page_start on chunks(page_start);
create index if not exists idx_chat_messages_session_id on chat_messages(session_id);
create index if not exists idx_retrieval_logs_session_id on retrieval_logs(session_id);
