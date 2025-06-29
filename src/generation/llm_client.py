import os
import requests
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
REPO_ID = os.getenv("HUGGINGFACE_MODEL", "HuggingFaceH4/zephyr-7b-beta")
API_URL = f"https://api-inference.huggingface.co/models/{REPO_ID}"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def call_llm(prompt: str, max_length: int = 200) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_length, "temperature": 0.2}
    }
    try:
        print(f"[llm_client] POST {API_URL}")
        print(f"[llm_client] HEADERS: {HEADERS}")
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        print(f"[llm_client] Status code: {response.status_code}")
        text = response.text
        print(f"[llm_client] Response text: {text}")
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return str(data)
    except Exception as e:
        print(f"[llm_client] error HTTP HF: {e}")
        return f"error in generate response: {e}"