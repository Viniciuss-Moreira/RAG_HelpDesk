import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents_from_dir(directory: str) -> list[dict]:

    docs = []

    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append({
                    "content": content,
                    "source": str(file_path)
                })

    return docs
