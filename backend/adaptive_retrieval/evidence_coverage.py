from __future__ import annotations
import re

def detect_aspect(text: str, query_aspects: list[str], selected_count: int) -> str:
    lower=text.lower()
    for aspect in query_aspects:
        if aspect == 'Definition' and (re.search(r'\b(is an?|framework|system|distributed|provides|keeps|coordinates)\b', lower)): return aspect
        if aspect == 'Applications' and (re.search(r'\b(application|applications|pipeline|warehousing|analytics|workload|streaming)\b', lower) or 'machine learning' in lower): return aspect
        if aspect == 'Advantages' and (re.search(r'(?<!-)\b(scale|scales|efficient|memory|fault|performance|cost|latency|suitable)\b', lower)): return aspect
    return 'Other'

def detect_topics(text: str, query_topics: list[str]) -> set[str]:
    lower=text.lower()
    return {topic for topic in query_topics if re.search(rf'\b{re.escape(topic)}\b', lower)}

def contribution(aspect: str, covered: set[str]) -> float:
    return 1.0 if aspect not in covered else .15

def ratio(covered: set[str], aspects: list[str]) -> float:
    return len(covered.intersection(aspects)) / max(1, len(aspects))

def topic_ratio(covered: set[str], topics: list[str]) -> float:
    return len(covered.intersection(topics)) / max(1, len(topics))
