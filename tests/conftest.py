import pytest
import os
from dotenv import load_dotenv

@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Carrega variáveis de ambiente do .env antes dos testes"""
    load_dotenv()
    os.environ.setdefault("HUGGINGFACE_API_TOKEN", "fake-token-for-tests")