import re
from typing import List, Dict

def split_text(
    documents: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict]:
    chunks = []

    for doc in documents:
        text = doc["content"]
        source = doc.get("source", "unknown")

        entries = re.split(r"(?=Pergunta:)", text)

        for entry_id, entry in enumerate(entries):
            entry = entry.strip()
            if not entry:
                continue

            if len(entry) <= chunk_size:
                chunks.append({
                    "content": entry,
                    "source": source,
                    "entry_id": entry_id,
                    "chunk_id": 0
                })
            else:
                start = 0
                end = chunk_size
                chunk_id = 0

                while start < len(entry):
                    snippet = entry[start:end]
                    chunks.append({
                        "content": snippet,
                        "source": source,
                        "entry_id": entry_id,
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1
                    start = end - chunk_overlap
                    end = start + chunk_size

    return chunks
