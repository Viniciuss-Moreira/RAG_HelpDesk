# src/evaluation/evaluator.py
import json
import numpy as np
from src.retrieval.retriever import Retriever

def precision_at_k(retrieved_texts, expected_keywords) -> float:
    hits = sum(any(kw.lower() in chunk.lower() for kw in expected_keywords) for chunk in retrieved_texts)
    return hits / len(retrieved_texts) if retrieved_texts else 0.0

def run_evaluation():
    print("🔎 Iniciando avaliação com Precision@3\n")
    
    # Carrega o benchmark
    with open("tests/benchmark.json", "r") as f:
        benchmark = json.load(f)
    
    # Carrega os textos dos chunks (na mesma ordem do vetor de embeddings)
    with open("data/raw/base_treinamento.txt", "r") as f:
        chunks = [line.strip() for line in f.readlines()]
    
    print(f"📋 Total de chunks carregados: {len(chunks)}")
    
    # Inicializa o retriever
    retriever = Retriever("data/embeddings/batch_000.npy")
    
    precisions = []
    
    for i, item in enumerate(benchmark, 1):
        query = item["query"]
        expected_keywords = item["expected_keywords"]
        
        print(f"\n🔍 Processando Query {i}: {query}")
        print(f"🎯 Keywords esperadas: {expected_keywords}")
        
        # CORREÇÃO: Sintaxe correta e captura ambos valores
        try:
            # Ajuste o método conforme seu retriever (retrieve ou search)
            idxs, scores = retriever.retrieve(query, top_k=3)  # CORRIGIDO: top_k em vez de top*k
            
            # Debug: mostrar o que foi retornado
            print(f"📊 Retornado - Type idxs: {type(idxs)}, Type scores: {type(scores)}")
            print(f"📊 Valores - idxs: {idxs}, scores: {scores}")
            
        except Exception as e:
            print(f"❌ Erro ao recuperar documentos: {e}")
            continue
        
        # Garante que idxs seja uma lista
        if isinstance(idxs, (np.int64, int)):
            idxs = [int(idxs)]
        elif isinstance(idxs, np.ndarray):
            idxs = idxs.flatten().tolist() if idxs.ndim > 1 else idxs.tolist()
        else:
            idxs = list(idxs)
        
        # Recupera os textos
        retrieved_texts = []
        for j in idxs:
            if int(j) < len(chunks):
                text = chunks[int(j)]
                retrieved_texts.append(text)
                print(f"📄 Documento {j}: {text[:100]}...")
            else:
                print(f"⚠️ Aviso: índice {j} está fora de alcance e será ignorado.")
        
        # Calcula precision
        score = precision_at_k(retrieved_texts, expected_keywords)
        precisions.append(score)
        
        print(f"✅ Query: {query}")
        print(f"📈 Precision@3: {score:.2f}")
        
        # Debug detalhado se score = 0
        if score == 0.0:
            print("🔍 DEBUG - Por que Precision = 0?")
            for idx, text in enumerate(retrieved_texts):
                keyword_matches = [kw for kw in expected_keywords if kw.lower() in text.lower()]
                print(f"  Texto {idx+1}: {keyword_matches if keyword_matches else 'Nenhuma keyword encontrada'}")
    
    avg_precision = np.mean(precisions)
    print(f"\n📊 Média Precision@3: {avg_precision:.2f}")

if __name__ == "__main__":
    run_evaluation()