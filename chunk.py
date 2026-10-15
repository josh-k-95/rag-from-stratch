import json
import sys
from pathlib import Path

MAX_CHARS = 2000   # rough stand-in for the embedding model's token limit
MIN_CHARS = 60     # below this, a chunk probably carries no real content


def is_numeric(cell):
    try:
        float(cell.replace(",", ""))
        return True
    except ValueError:
        return False


def is_data_table(rows):
    """Heuristic: a real data table has >=2 rows and a non-numeric header."""
    if len(rows) < 2:
        return False
    header = [c for c in rows[0] if c]
    return bool(header) and not all(is_numeric(c) for c in header)


def to_markdown(rows):
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    esc = lambda c: c.replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |",
             "| " + " | ".join("---" for _ in range(width)) + " |"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def layout_table_to_text(rows):
    return "\n".join(" | ".join(r) for r in rows)


def make_chunks(parsed, course):
    source = parsed["source_file"]
    chunks = []
    last_title = ""

    for s in parsed["slides"]:
        n = s["slide"]
        if s["title"]:
            title = s["title"]
            last_title = s["title"]
        else:
            title = f"{last_title} (cont.)" if last_title else ""

        body_parts = list(s["text_blocks"])
        data_tables = []
        for rows in s["tables"]:
            if is_data_table(rows):
                data_tables.append(rows)
            else:
                body_parts.append(layout_table_to_text(rows))
        if s["notes"]:
            body_parts.append("Speaker notes: " + s["notes"])

        base = {
            "course": course,
            "source_file": source,
            "slide": n,
            "title": title,
            "learning_outcome": None,
            "hidden": s["hidden"],
            "unparsed_images": [img["file"] for img in s["images"]],
        }

        body = "\n\n".join(body_parts)
        text = f"# {title}\n\n{body}".strip() if title else body
        chunks.append({**base,
                       "chunk_id": f"{source}#slide{n}#text",
                       "chunk_type": "text",
                       "text": text,
                       "embed_text": text})

        for i, rows in enumerate(data_tables, start=1):
            table_text = f"Table from slide {n} ({title}):\n\n{to_markdown(rows)}"
            chunks.append({**base,
                           "chunk_id": f"{source}#slide{n}#table{i}",
                           "chunk_type": "table",
                           "text": table_text,
                           "embed_text": table_text,
                           "summary": None})
    return chunks


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python chunk.py "parsed/<file>.json" "<COURSE CODE>"')
    src, course = Path(sys.argv[1]), sys.argv[2]
    parsed = json.loads(src.read_text(encoding="utf-8"))
    chunks = make_chunks(parsed, course)

    out_dir = Path("chunks")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{src.stem}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_type = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1
    lengths = sorted(len(c["text"]) for c in chunks)
    print(f"Chunks: {len(chunks)}  {by_type}")
    print(f"Length (chars): min {lengths[0]}, median {lengths[len(lengths)//2]}, max {lengths[-1]}")

    for c in chunks:
        if len(c["text"]) > MAX_CHARS:
            print(f"  LONG  slide {c['slide']:>2} {c['chunk_type']}: {len(c['text'])} chars")
        if len(c["text"]) < MIN_CHARS:
            print(f"  TINY  slide {c['slide']:>2} {c['chunk_type']}: {c['text']!r}")

    with_images = sum(1 for c in chunks if c["chunk_type"] == "text" and c["unparsed_images"])
    print(f"Text chunks with unparsed images: {with_images}/{by_type.get('text', 0)}")
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()