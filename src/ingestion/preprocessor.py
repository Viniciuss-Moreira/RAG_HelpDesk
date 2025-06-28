import re

def clean_text(text: str) -> str:
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


def preprocess_documents(documents: list[dict]) -> list[dict]:
    return [
        {
            **doc,
            "content": clean_text(doc["content"])
        }
        for doc in documents
    ]