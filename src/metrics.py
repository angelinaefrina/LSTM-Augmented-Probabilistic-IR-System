import numpy as np

def calculate_average_precision(retrieved_doc_ids, relevant_doc_ids):
    """
    Calculates AP for a single query.
    retrieved_doc_ids: list of document IDs returned by the system, sorted by score descending.
    relevant_doc_ids: set or list of ground-truth relevant document IDs from qrels.
    """
    hits = 0
    sum_precisions = 0.0
    relevant_set = set(relevant_doc_ids)
    
    if not relevant_set:
        return 0.0

    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_set:
            hits += 1
            precision_at_i = hits / (i + 1.0)
            sum_precisions += precision_at_i
            
    return sum_precisions / len(relevant_set)

def calculate_map(ap_scores):
    """Calculates Mean Average Precision (MAP) across all queries."""
    if not ap_scores:
        return 0.0
    return np.mean(ap_scores)