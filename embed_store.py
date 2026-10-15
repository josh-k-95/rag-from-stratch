import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

UPSERT = """
INSERT INTO chunks (chunk_id, course, source_file, slide, chunk_type, title,
                    learning_outcome, text, embed_text, unparsed_images, hidden,
                    embedding_model, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    title = EXCLUDED.title,
    text = EXCLUDED.text,
    embed_text = EXCLUDED.embed_text,
    unparsed_images = EXCLUDED.unparsed_images,
    hidden = EXCLUDED.hidden,
    embedding_model = EXCLUDED.embedding_model,
    embedding = EXCLUDED.embedding
"""


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: python embed_store.py "chunks/<file>.jsonl"')
    lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines if line.strip()]

    model = SentenceTransformer(MODEL_NAME)  # first run downloads the model
    print(f"device: {model.device} | dim: {model.get_sentence_embedding_dimension()} "
          f"| max tokens: {model.max_seq_length}")

    # The model silently ignores everything past max_seq_length tokens.
    for c in chunks:
        n_tokens = len(model.tokenizer(c["embed_text"])["input_ids"])
        if n_tokens > model.max_seq_length:
            print(f"  TRUNCATED slide {c['slide']}: {n_tokens} tokens")

    vectors = model.encode(
        [c["embed_text"] for c in chunks],
        normalize_embeddings=True,   # unit length -> cosine distance behaves cleanly
        batch_size=32,
        show_progress_bar=True,
    )

    load_dotenv()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute(Path("schema.sql").read_text())
        register_vector(conn)

        rows = [
            (c["chunk_id"], c["course"], c["source_file"], c["slide"], c["chunk_type"],
             c["title"], c["learning_outcome"], c["text"], c["embed_text"],
             Jsonb(c["unparsed_images"]), c["hidden"], MODEL_NAME, v)
            for c, v in zip(chunks, vectors)
        ]
        with conn.cursor() as cur:
            cur.executemany(UPSERT, rows)

        total = conn.execute(
            "SELECT count(*) FROM chunks WHERE source_file = %s",
            (chunks[0]["source_file"],),
        ).fetchone()[0]
        print(f"Stored {total} chunks for {chunks[0]['source_file']}")

        # Sanity check: a slide's nearest neighbours should be about the same topic.
        probe = next((c for c in chunks if c["slide"] == 23), chunks[0])
        print(f"\nNearest neighbours of slide {probe['slide']} ({probe['title']}):")
        for slide, title, dist in conn.execute(
            """
            SELECT slide, title,
                   embedding <=> (SELECT embedding FROM chunks WHERE chunk_id = %s) AS dist
            FROM chunks
            ORDER BY dist
            LIMIT 5
            """,
            (probe["chunk_id"],),
        ):
            print(f"  slide {slide:>2}  {dist:.3f}  {title}")


if __name__ == "__main__":
    main()