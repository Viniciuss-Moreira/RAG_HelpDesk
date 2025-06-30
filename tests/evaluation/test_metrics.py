from src.evaluation.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank

def test_precision_at_k():
    retrieved = [1, 2, 3]
    relevant = [2, 3, 4]
    p = precision_at_k(retrieved, relevant, k=3)
    assert round(p, 2) == 0.67

def test_recall_at_k():
    retrieved = [1, 2, 3]
    relevant = [2, 3, 4]
    r = recall_at_k(retrieved, relevant, k=3)
    assert round(r, 2) == 0.67

def test_mrr():
    retrieved_lists = [[2, 3, 4], [1, 4, 5], [6, 7, 8]]
    relevant_lists = [[3], [4], [8]]
    mrr = mean_reciprocal_rank(retrieved_lists, relevant_lists)
    assert round(mrr, 3) == round((1/2 + 1/2 + 1/3) / 3, 3)
