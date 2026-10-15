import argparse
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from retrieve import MODEL_NAME, retrieve


def score(expected, retrieved_slides):
    exp = set(expected)
    rank = next((i for i, s in enumerate(retrieved_slides, start=1) if s in exp), None)
    recall = len(exp & set(retrieved_slides)) / len(exp)
    rr = 1 / rank if rank else 0.0
    return rank is not None, recall, rr, rank


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a labelled test set.")
    parser.add_argument("test_file", nargs="?", default="test_set.jsonl")
    parser.add_argument("--course", default="CSAI 351")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-prefix", action="store_true")
    args = parser.parse_args()

    lines = Path(args.test_file).read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]

    load_dotenv()
    model = SentenceTransformer(MODEL_NAME)
    positives, negatives = [], []

    print(f"\nk={args.k}  prefix={not args.no_prefix}  cases={len(cases)}\n")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        for c in cases:
            rows = retrieve(conn, model, c["query"], args.course, args.k,
                            use_prefix=not args.no_prefix)
            slides = [r[1] for r in rows]
            top1 = rows[0][4] if rows else float("nan")

            if c["expected_slides"]:
                hit, recall, rr, rank = score(c["expected_slides"], slides)
                positives.append({**c, "hit": hit, "recall": recall, "rr": rr, "top1": top1})
                print(f"{c['id']:<4} {c['kind']:<12} {'HIT ' if hit else 'MISS'} "
                      f"rank {rank or '-':>2}  recall {recall:.2f}  top1 {top1:.3f}  "
                      f"got {slides}  expected {c['expected_slides']}")
            else:
                negatives.append({**c, "top1": top1})
                print(f"{c['id']:<4} {c['kind']:<12} NEG              "
                      f"top1 {top1:.3f}  got {slides}")

    if positives:
        print(f"\nPositive queries: {len(positives)}")
        print(f"  Hit@{args.k}:    {mean([p['hit'] for p in positives]):.2f}")
        print(f"  Recall@{args.k}: {mean([p['recall'] for p in positives]):.2f}")
        print(f"  MRR:      {mean([p['rr'] for p in positives]):.2f}")
        for kind in sorted({p["kind"] for p in positives}):
            group = [p for p in positives if p["kind"] == kind]
            print(f"  {kind:<12} Hit@{args.k} {mean([p['hit'] for p in group]):.2f}  (n={len(group)})")

    if positives and negatives:
        lowest_neg = min(n["top1"] for n in negatives)
        overlapping = sum(1 for p in positives if p["top1"] >= lowest_neg)
        print(f"\nTop-1 distance  relevant: {mean([p['top1'] for p in positives]):.3f}"
              f"  |  negative: {mean([n['top1'] for n in negatives]):.3f}")
        print(f"Relevant queries whose top-1 is no closer than the best negative: "
              f"{overlapping}/{len(positives)}")


if __name__ == "__main__":
    main()