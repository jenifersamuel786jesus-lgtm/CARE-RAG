from __future__ import annotations
import re

def analyze(query: str) -> tuple[float, str]:
    words = re.findall(r'\b\w+\b', query.lower())
    connectors = sum(w in {'and','or','compare','explain','why','how','advantages','applications','difference'} for w in words)
    score = min(.98, max(.12, .18 + len(words)*.025 + connectors*.07))
    return round(score,2), 'Simple' if score < .4 else 'Moderate' if score < .68 else 'Complex'

def aspects(query: str) -> list[str]:
    q=query.lower(); found=[]
    if any(x in q for x in ['what is','define','explain','compare','hadoop','spark']): found.append('Definition')
    if any(x in q for x in ['advantage','benefit','why','suitable','scale','performance']): found.append('Advantages')
    if any(x in q for x in ['application','used','use','where','workload']): found.append('Applications')
    return list(dict.fromkeys(found or ['Definition']))
