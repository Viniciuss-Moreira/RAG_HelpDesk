def split_text(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[dict]:

    chunks = []

    for doc in documents:
        content = doc["content"]
        source = doc.get("source", "unknown")

        start = 0
        end = chunk_size
        chunk_id = 0

        while start < len(content):
            chunk_text = content[start:end]

            chunks.append({
                "content": chunk_text,
                "source": source,
                "chunk_id": chunk_id
            })

            chunk_id += 1
            start = end - chunk_overlap
            end = start + chunk_size

    return chunks
