def detect_aspect(text: str, query_aspects: list[str], selected_count: int) -> str:
    lower=text.lower()
    for aspect in query_aspects:
        if aspect == 'Definition' and any(x in lower for x in ['is an','framework','system','distributed']): return aspect
        if aspect == 'Advantages' and any(x in lower for x in ['scale','efficient','memory','fault','performance','cost']): return aspect
        if aspect == 'Applications' and any(x in lower for x in ['application','pipeline','warehousing','workload','analytics']): return aspect
    return query_aspects[min(selected_count, len(query_aspects)-1)]

def contribution(aspect: str, covered: set[str]) -> float:
    return 1.0 if aspect not in covered else .15

def ratio(covered: set[str], aspects: list[str]) -> float:
    return len(covered.intersection(aspects)) / max(1, len(aspects))
