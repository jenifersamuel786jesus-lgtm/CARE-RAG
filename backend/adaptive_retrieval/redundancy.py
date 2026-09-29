from __future__ import annotations
from sklearn.metrics.pairwise import cosine_similarity

def score(candidate_index: int, selected_indices: list[int], matrix) -> float:
    if not selected_indices: return 0.0
    return float(max(cosine_similarity(matrix[candidate_index], matrix[selected_indices])[0]))
