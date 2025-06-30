import numpy as np
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def precision_at_k(retrieved_idxs: List[int], relevant_idxs: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_idxs[:k]
    hits = sum(1 for idx in top_k if idx in relevant_idxs)
    return hits / k


def recall_at_k(retrieved_idxs: List[int], relevant_idxs: List[int], k: int) -> float:
    if not relevant_idxs:
        return 0.0
    top_k = retrieved_idxs[:k]
    hits = sum(1 for idx in top_k if idx in relevant_idxs)
    return hits / len(relevant_idxs)


def mean_reciprocal_rank(retrieved_lists: List[List[int]], relevant_idxs_list: List[List[int]]) -> float:

    rr_scores = []
    for retrieved, relevant in zip(retrieved_lists, relevant_idxs_list):
        rr = 0.0
        for rank, idx in enumerate(retrieved, start=1):
            if idx in relevant:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def bleu_score(reference: str, candidate: str) -> float:
    smoothie = SmoothingFunction().method4
    weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference.split()], candidate.split(), weights=weights, smoothing_function=smoothie)
