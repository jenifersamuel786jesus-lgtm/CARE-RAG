from __future__ import annotations
from .complexity import analyze, aspects
from .relevance import rank
from .redundancy import score as redundancy_score
from .information_gain import score as gain_score
from .evidence_coverage import detect_aspect, contribution, ratio
from .sufficiency import is_sufficient

WEIGHTS={'relevance':.40,'information_gain':.20,'evidence_coverage':.25,'low_redundancy':.15}

def select(query, chunks, vectorizer, matrix, candidate_pool=10, max_rounds=3, threshold=.85):
    qvec=vectorizer.transform([query]); complexity_score,complexity_level=analyze(query); query_aspects=aspects(query)
    query_lower=query.lower()
    named_topics=[topic for topic in ('hadoop','spark','mapreduce','kafka','nat','scriptcon') if topic in query_lower]
    ranked=rank(qvec,matrix,candidate_pool); selected_indices=[]; covered=set(); candidates=[]; rounds=0
    for position,(idx,relevance) in enumerate(ranked):
        c=chunks[idx]; topic_match=not named_topics or any(topic in f"{c.get('document_name','')} {c['text']}".lower() for topic in named_topics); redundancy=redundancy_score(idx,selected_indices,matrix); aspect=detect_aspect(c['text'],query_aspects,len(selected_indices)); coverage_contribution=contribution(aspect,covered); gain=gain_score(relevance,redundancy); adaptive=WEIGHTS['relevance']*relevance+WEIGHTS['information_gain']*gain+WEIGHTS['evidence_coverage']*coverage_contribution+WEIGHTS['low_redundancy']*(1-redundancy)
        selected=False
        if topic_match and relevance >= .08 and (aspect not in covered or (adaptive > .42 and redundancy < .7)):
            selected=True; selected_indices.append(idx); covered.add(aspect)
        reason=(f'Covers {aspect}; high gain with acceptable redundancy' if selected else ('Rejected: different topic from the question' if not topic_match else ('Rejected: highly redundant with selected evidence' if relevance >= .08 else 'Not relevant enough to support this question')))
        candidates.append({**c,'relevance':round(relevance,3),'normalized_relevance':round(relevance,3),'redundancy':round(redundancy,3),'information_gain':round(gain,3),'evidence_contribution':round(coverage_contribution,3),'adaptive_score':round(adaptive,3),'selected':selected,'aspect':aspect,'reason':reason})
        rounds=min(max_rounds,1+position//max(1,len(ranked)//max_rounds))
        if is_sufficient(ratio(covered,query_aspects),threshold,len(query_aspects),len(selected_indices)): break
    coverage=round(ratio(covered,query_aspects),3); selected=[c for c in candidates if c['selected']]
    return {'complexity_score':complexity_score,'complexity_level':complexity_level,'query_aspects':query_aspects,'candidates':candidates,'selected':selected,'rejected':[c for c in candidates if not c['selected']],'evidence_coverage':coverage,'sufficient':is_sufficient(coverage,threshold,len(query_aspects),len(selected)),'retrieval_rounds':rounds,'weights':WEIGHTS}
