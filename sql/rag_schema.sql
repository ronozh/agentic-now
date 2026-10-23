CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          serial PRIMARY KEY,
    file_name   text UNIQUE NOT NULL,
    title       text NOT NULL,
    page_count  int NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  int NOT NULL REFERENCES documents ON DELETE CASCADE,
    page         int NOT NULL,  -- PDF page (1-based), what a PDF viewer jumps to
    printed_page int,           -- page number printed in the book footer, if found
    chunk_index  int NOT NULL,  -- order within the document
    content      text NOT NULL,
    embedding    vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
