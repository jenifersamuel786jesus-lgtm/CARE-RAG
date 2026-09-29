def score(relevance: float, redundancy: float) -> float:
    return max(0.05, (1.0 - redundancy) * (0.65 + 0.35 * relevance))
