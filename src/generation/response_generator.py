from src.generation.llm_client import call_llm

def build_prompt(query: str, context: str) -> str:
    return (
        "tecnical context:\n"
        f"{context}\n\n"
        "with base this, response:\n"
        f"{query}"
    )

def generate_answer(query: str, context: str) -> str:
    prompt = build_prompt(query, context)
    return call_llm(prompt)
