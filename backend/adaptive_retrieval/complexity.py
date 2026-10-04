from __future__ import annotations
import re

STOP = {
    'what','which','where','when','who','why','how','are','is','the','and','or','a','an','to','of','for','in','on','with','from','about','does','do','can','could','would','should','its','their','this','that','these','those','explain','compare','tell','me','please','give','describe','common','commonly'
}
INTENT = {'advantage','advantages','benefit','benefits','application','applications','used','use','usage','workload','performance','suitable','difference','differences','why','scale','scaling','definition','define'}

def analyze(query: str) -> tuple[float, str]:
    words = re.findall(r'\b\w+\b', query.lower())
    connectors = sum(w in {'and','or','compare','explain','why','how','advantages','applications','difference'} for w in words)
    score = min(.98, max(.12, .18 + len(words)*.025 + connectors*.07))
    return round(score,2), 'Simple' if score < .4 else 'Moderate' if score < .68 else 'Complex'

def aspects(query: str) -> list[str]:
    q=query.lower(); found=[]
    if any(x in q for x in ['what is','define','explain','compare']): found.append('Definition')
    if any(x in q for x in ['advantage','benefit','why','suitable','scale','performance']): found.append('Advantages')
    if any(x in q for x in ['application','used','use','where','workload']): found.append('Applications')
    return list(dict.fromkeys(found or ['Definition']))

def topics(query: str, chunks: list[dict]) -> list[str]:
    """Return query terms that are represented in the indexed corpus and behave like topics.

    This is deliberately corpus-aware: it avoids hard-coding a fixed product vocabulary while
    keeping generic question words and intent words out of the topic set.
    """
    q_terms = re.findall(r'\b[a-z][a-z0-9-]{2,}\b', query.lower())
    corpus = ' '.join(f"{c.get('document_name','')} {c.get('text','')}" for c in chunks).lower()
    found=[]
    for term in q_terms:
        if term in STOP or term in INTENT or term in found:
            continue
        if re.search(rf'\b{re.escape(term)}\b', corpus):
            found.append(term)
    return found[:6]
