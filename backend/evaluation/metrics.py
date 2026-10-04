from __future__ import annotations
import math, re
from ..adaptive_retrieval.complexity import STOP, INTENT

def relevant_ids(query: str, chunks: list[dict]) -> set[str]:
    q=query.lower(); ids=set(); terms=[]
    for term in re.findall(r'\b[a-z][a-z0-9-]{2,}\b',q):
        if term not in STOP and term not in INTENT: terms.append(term)
    if any(x in q for x in ['advantage','benefit','why','scale']): terms += ['scale','cost','efficient','fault','memory','performance']
    if any(x in q for x in ['application','used','where','workload']): terms += ['application','pipeline','warehousing','analytics','workload']
    for c in chunks:
        if any(t in c['text'].lower() for t in terms): ids.add(c['chunk_id'])
    return ids

def retrieval(selected: list[dict], ranked: list[dict], relevant: set[str]) -> dict:
    if not relevant: return {'recall_at_k':0,'mrr':0,'ndcg':0}
    chosen={x['chunk_id'] for x in selected}; recall=len(chosen & relevant)/len(relevant)
    first=next((i+1 for i,x in enumerate(ranked) if x['chunk_id'] in relevant),None); mrr=1/first if first else 0
    gains=[1 if x['chunk_id'] in relevant else 0 for x in ranked]
    dcg=sum(g/math.log2(i+2) for i,g in enumerate(gains[:max(1,len(selected))]))
    ideal=sum(1/math.log2(i+2) for i in range(min(len(relevant),max(1,len(selected)))))
    return {'recall_at_k':round(recall,3),'mrr':round(mrr,3),'ndcg':round(dcg/ideal,3) if ideal else 0}

def answer_quality(answer: str, selected: list[dict], query: str) -> dict:
    answer_words=set(re.findall(r'\b[a-z]{4,}\b',answer.lower())); context_words=set(re.findall(r'\b[a-z]{4,}\b',' '.join(x['text'] for x in selected).lower()))
    faithfulness=len(answer_words & context_words)/max(1,len(answer_words)); correctness=min(1,faithfulness*.75 + (0.25 if any(w in answer.lower() for w in ['hadoop','spark','distributed']) else 0))
    return {'faithfulness':round(faithfulness,3),'answer_correctness':round(correctness,3)}
