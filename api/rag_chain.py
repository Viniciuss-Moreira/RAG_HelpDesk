import os
import json
import torch
from pathlib import Path
from dotenv import load_dotenv
from operator import itemgetter
from typing import List, Dict
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from src.retrieval.reranker import HybridReranker
from src.generation.prompt_templates import RAG_PROMPT_TEMPLATE

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_STORE_PATH = str(PROJECT_ROOT / "data" / "vector_store_faiss")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_REPO_ID = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("token api not found")

client = InferenceClient(model=LLM_REPO_ID, token=HF_TOKEN)

prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_docs(docs: List[Dict]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer_from_context(input_dict: Dict) -> str:
    context_docs = input_dict["context"]
    query_text = input_dict["query"]
    formatted_context = format_docs(context_docs)
    
    prompt_value = prompt_template.invoke({
        "context": formatted_context,
        "query": query_text
    })
    final_prompt_text = str(prompt_value)
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": final_prompt_text}],
            max_tokens=300,
            temperature=0.1
        )
        
        raw_answer = response.choices[0].message.content
        
        clean_answer = raw_answer.strip()
        
        if clean_answer.startswith('text="') and clean_answer.endswith('"'):
            clean_answer = clean_answer[6:-1]
        elif clean_answer.startswith("text='") and clean_answer.endswith("'"):
            clean_answer = clean_answer[6:-1]

        if clean_answer.startswith("Resposta:"):
            clean_answer = clean_answer.split("Resposta:", 1)[1]
        
        stop_phrases = ["<PERGUNTA>", "Pergunta:"]
        for phrase in stop_phrases:
            if phrase in clean_answer:
                clean_answer = clean_answer.split(phrase)[0]

        return clean_answer.strip()

    except Exception as e:
        print(f"error to call Huggingface API: {e}")
        return f"error to call llm: {e}"

def get_rag_chain():
    
    cache_dir = "/app/huggingface_cache"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=cache_dir
    )
    
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True
    )
    
    hybrid_reranker = HybridReranker(vector_store=vector_store, device=device, cache_dir=cache_dir)
        
    retrieval_chain = lambda query: hybrid_reranker.retrieve_and_rerank(query)

    rag_chain = {
        "context": retrieval_chain,
        "query": RunnablePassthrough()
    } | RunnableParallel({
        "source_chunks": itemgetter("context"),
        "answer": generate_answer_from_context
    })

    print("pipeline ready")
    return rag_chain