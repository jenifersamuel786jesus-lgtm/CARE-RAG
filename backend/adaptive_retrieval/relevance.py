from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rank(query_vector, matrix, limit: int):
    scores = cosine_similarity(query_vector, matrix)[0]
    order = np.argsort(-scores)[:min(limit, len(scores))]
    return [(int(i), float(scores[int(i)])) for i in order]
