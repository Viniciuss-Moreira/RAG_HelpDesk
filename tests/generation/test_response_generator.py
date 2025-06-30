import pytest
from src.generation.response_generator import generate_answer
from unittest.mock import patch

@patch("src.generation.response_generator.call_llm")
def test_generate_answer(mock_call_llm):
    mock_call_llm.return_value = "Resposta gerada."

    query = "Como resolver erro de tela azul?"
    context = "A tela azul pode ser causada por problemas de driver, memória RAM, ou disco."

    resposta = generate_answer(query, context)

    assert resposta == "Resposta gerada."
    mock_call_llm.assert_called_once()
    args = mock_call_llm.call_args[0][0]
    assert query in args
    assert context in args