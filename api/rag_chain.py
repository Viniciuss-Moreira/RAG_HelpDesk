import os
from pathlib import Path
from dotenv import load_dotenv
from operator import itemgetter
from typing import List, Dict

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_STORE_PATH = str(PROJECT_ROOT / "data" / "vector_store_faiss")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_REPO_ID = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

client = InferenceClient(model=LLM_REPO_ID, token=HF_TOKEN)

prompt_template = PromptTemplate.from_template("""
<|system|>
Você é um assistente de helpdesk de TI especialista. Sua tarefa é responder à pergunta do usuário usando APENAS o contexto fornecido.
Responda em Português.
Se a resposta não estiver no contexto, responda EXATAMENTE: 'Não encontrei informações sobre isso na minha base de dados.'
Não adicione nenhuma informação que não esteja no texto de contexto.
</s>
<|user|>
Contexto:
{context}

Pergunta:
{query}
</s>
<|assistant|>
Resposta em Português:
""")

def format_docs(docs: List[Dict]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer_from_context(input_dict: Dict) -> str:
    context_docs = input_dict["context"]
    query_text = input_dict["query"]
    formatted_context = format_docs(context_docs)
    prompt_value = prompt_template.invoke({"context": formatted_context, "query": query_text})
    final_prompt_text = str(prompt_value)
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": final_prompt_text}],
            max_tokens=300,
            temperature=0.1
        )
        raw_answer = response.choices[0].message.content
        return raw_answer.strip()
    except Exception as e:
        print(f"error in huggingface API: {e}")
        return f"error in contact llm: {e}"

def get_rag_chain():
    print("loading pipeline")
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True
    )
    
    try:
        from src.retrieval.reranker import HybridReranker
        hybrid_reranker = HybridReranker(vector_store=vector_store)
        retrieval_chain = lambda query: hybrid_reranker.retrieve_and_rerank(query)
        print("using pipeline with reranker")
    except ImportError:
        retrieval_chain = vector_store.as_retriever(search_kwargs={"k": 5})
        print("reranker not found, using simple retriever")

    rag_chain = {
        "context": retrieval_chain,
        "query": RunnablePassthrough()
    } | RunnableParallel({
        "source_chunks": itemgetter("context"),
        "answer": generate_answer_from_context
    })

    print("pipeline ready.")
    return rag_chain