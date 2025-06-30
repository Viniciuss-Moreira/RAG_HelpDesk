def read_txt_chunks(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]