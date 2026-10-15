import argparse
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SQL = """
SELECT chunk_id, slide, title, text, embedding <=> %s AS distance
FROM chunks
WHERE course = %s
  AND embedding_model = %s
ORDER BY distance
LIMIT %s
"""


def embed_query(model, query, use_prefix=True):
    text = QUERY_PREFIX + query if use_prefix else query
    return model.encode(text, normalize_embeddings=True)


def retrieve(conn, model, query, course, k=5, use_prefix=True):
    """Return [(chunk_id, slide, title, text, distance), ...], closest first."""
    qvec = embed_query(model, query, use_prefix)
    return conn.execute(SQL, (qvec, course, MODEL_NAME, k)).fetchall()


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks for a query.")
    parser.add_argument("query")
    parser.add_argument("--course", default="CSAI 351")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-prefix", action="store_true",
                        help="embed the query without BGE's instruction prefix")
    args = parser.parse_args()

    load_dotenv()
    model = SentenceTransformer(MODEL_NAME)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        rows = retrieve(conn, model, args.query, args.course, args.k,
                        use_prefix=not args.no_prefix)

    print(f"\nQuery: {args.query!r}  (prefix: {not args.no_prefix}, course: {args.course})")
    if not rows:
        print("No results: check the course code matches what you stored.")
    for rank, (chunk_id, slide, title, text, dist) in enumerate(rows, start=1):
        preview = " ".join(text.split())[:90]   # collapse whitespace for one-line display
        print(f"{rank}. slide {slide:>2}  dist {dist:.3f}  {preview}")


if __name__ == "__main__":
    main()