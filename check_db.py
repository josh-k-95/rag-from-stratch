import os
import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()
url = os.environ["DATABASE_URL"]

with psycopg.connect(url) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # once per database
    register_vector(conn)  # numpy <-> vector; must come after the extension exists

    ver = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()[0]
    print("pgvector version:", ver)

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])   # unrelated to a
    c = np.array([0.9, 0.1, 0.0])   # nearly same direction as a

    d_ab, d_ac = conn.execute(
        "SELECT %s::vector <=> %s::vector, %s::vector <=> %s::vector",
        (a, b, a, c),
    ).fetchone()
    print(f"cosine distance a<->b (unrelated): {d_ab:.3f}")
    print(f"cosine distance a<->c (similar):   {d_ac:.3f}")