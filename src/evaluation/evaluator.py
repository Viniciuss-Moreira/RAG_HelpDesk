import json
import numpy as np
from src.retrieval.retriever import Retriever
from difflib import SequenceMatcher

def has_similar_keyword(text, keywords, threshold=0.7):
    return any(SequenceMatcher(None, kw.lower(), text.lower()).ratio() > threshold for kw in keywords)

def precision_at_k(retrieved_texts, expected_keywords) -> float:
    hits = sum(has_similar_keyword(chunk, expected_keywords) for chunk in retrieved_texts)
    return hits / len(retrieved_texts) if retrieved_texts else 0.0

def run_evaluation():
    print("starting avaliation with precision@3\n")
    
    with open("tests/benchmark.json", "r") as f:
        benchmark = json.load(f)
    
    with open("data/raw/base_treinamento.txt", "r") as f:
        chunks = [line.strip() for line in f.readlines()]
    
    print(f"total chunks loaded: {len(chunks)}")
    
    retriever = Retriever("data/embeddings/batch_000.npy")
    
    precisions = []
    
    for i, item in enumerate(benchmark, 1):
        query = item["query"]
        expected_keywords = item["expected_keywords"]
        
        print(f"\nprocessing query {i}: {query}")
        print(f"expected keywords: {expected_keywords}")
        
        try:
            idxs, scores = retriever.retrieve(query, top_k=3)
            
            print(f"returned - Type idxs: {type(idxs)}, Type scores: {type(scores)}")
            print(f"valors - idxs: {idxs}, scores: {scores}")
            
        except Exception as e:
            print(f"error to rescue docs: {e}")
            continue
        
        if isinstance(idxs, (np.int64, int)):
            idxs = [int(idxs)]
        elif isinstance(idxs, np.ndarray):
            idxs = idxs.flatten().tolist() if idxs.ndim > 1 else idxs.tolist()
        else:
            idxs = list(idxs)
        
        retrieved_texts = []
        for j in idxs:
            if int(j) < len(chunks):
                text = chunks[int(j)]
                retrieved_texts.append(text)
                print(f"doc {j}: {text[:100]}...")
            else:
                print(f"warning, index {j} is out of range and will be ignored")
        
        # Calcula precision
        score = precision_at_k(retrieved_texts, expected_keywords)
        precisions.append(score)
        
        print(f"query: {query}")
        print(f"Precision@3: {score:.2f}")
        
        if score == 0.0:
            print("DEBUG - why precision = 0?")
            for idx, text in enumerate(retrieved_texts):
                keyword_matches = [kw for kw in expected_keywords if kw.lower() in text.lower()]
                print(f"  text {idx+1}: {keyword_matches if keyword_matches else 'none keyword found'}")
    
    avg_precision = np.mean(precisions)
    print(f"\nmédia Precision@3: {avg_precision:.2f}")

if __name__ == "__main__":
    run_evaluation()