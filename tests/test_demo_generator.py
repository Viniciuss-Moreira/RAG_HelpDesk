from src.generation.response_generator import generate_answer

def test_generate_answer_basic():
    query = "O que é um SSD?"
    context = "SSD é um tipo de armazenamento mais rápido que o HD tradicional."
    answer = generate_answer(query, context)
    assert "armazenamento" in answer.lower()