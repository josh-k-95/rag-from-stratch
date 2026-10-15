CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    course           TEXT NOT NULL,
    source_file      TEXT NOT NULL,
    slide            INT  NOT NULL,
    chunk_type       TEXT NOT NULL CHECK (chunk_type IN ('text', 'table')),
    title            TEXT,
    learning_outcome TEXT,
    text             TEXT NOT NULL,        -- goes into the prompt
    embed_text       TEXT NOT NULL,        -- what was embedded
    unparsed_images  JSONB NOT NULL DEFAULT '[]',
    hidden           BOOLEAN NOT NULL DEFAULT false,
    embedding_model  TEXT NOT NULL,
    embedding        vector(384) NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);