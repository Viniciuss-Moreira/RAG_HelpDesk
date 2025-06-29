from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

print("HF_TOKEN =", os.getenv("HUGGINGFACE_API_TOKEN"))
print("HF_MODEL =", os.getenv("HUGGINGFACE_MODEL"))
