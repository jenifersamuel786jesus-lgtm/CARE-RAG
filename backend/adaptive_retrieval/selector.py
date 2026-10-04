from __future__ import annotations
from .complexity import analyze, aspects, topics
from .relevance import rank
from .redundancy import score as redundancy_score
from .information_gain import score as gain_score
from .evidence_coverage import detect_aspect, detect_topics, contribution, ratio, topic_ratio

WEIGHTS={'relevance':.40,'information_gain':.18,'evidence_coverage':.22,'low_redundancy':.12,'source_diversity':.08}


def select(query, chunks, vectorizer, matrix, candidate_pool=10, max_rounds=3, threshold=.85):
    if not chunks or vectorizer is None or matrix is None:
        return {'complexity_score':.12,'complexity_level':'Simple','query_aspects':['Definition'],'query_topics':[], 'candidates':[],'selected':[],'rejected':[],'evidence_coverage':0,'topic_coverage':0,'aspect_coverage':0,'sufficient':False,'retrieval_rounds':0,'weights':WEIGHTS}
    qvec=vectorizer.transform([query]); complexity_score,complexity_level=analyze(query); query_aspects=aspects(query); query_topics=topics(query,chunks)
    # Rank the whole indexed corpus, then expose a bounded candidate view. This guarantees
    # cross-document coverage even when one document dominates the top-k scores.
    ranked=rank(qvec,matrix,len(chunks)); limit=min(len(ranked), max(candidate_pool, len(query_topics)*3, len(query_aspects)*3, 12))
    ranked=ranked[:limit]
    selected_indices=[]; covered_aspects=set(); covered_topics=set(); selected_docs=set(); candidates=[]

    def score_candidate(idx, relevance):
        c=chunks[idx]; text=f"{c.get('document_name','')} {c.get('text','')}".lower()
        matched_topics=detect_topics(text,query_topics)
        aspect=detect_aspect(c['text'],query_aspects,len(selected_indices))
        redundancy=redundancy_score(idx,selected_indices,matrix)
        gain=gain_score(relevance,redundancy)
        new_aspect=1.0 if aspect not in covered_aspects else .15
        new_topics=len(matched_topics-covered_topics)/max(1,len(query_topics)) if query_topics else 0
        diversity=0.0 if c.get('document_name') in selected_docs else 1.0
        adaptive=(WEIGHTS['relevance']*relevance + WEIGHTS['information_gain']*gain + WEIGHTS['evidence_coverage']*(.65*new_aspect+.35*new_topics) + WEIGHTS['low_redundancy']*(1-redundancy) + WEIGHTS['source_diversity']*diversity)
        return c,matched_topics,aspect,redundancy,gain,adaptive

    # First guarantee one best supporting chunk per explicit topic, if evidence exists.
    forced=[]
    for topic in query_topics:
        options=[(idx,rel) for idx,rel in ranked if topic in detect_topics(f"{chunks[idx].get('document_name','')} {chunks[idx]['text']}".lower(),[topic]) and rel >= .035]
        if options:
            forced.append(max(options,key=lambda pair: pair[1]))
    ordered=[]
    seen=set()
    for item in forced + ranked:
        if item[0] not in seen:
            ordered.append(item); seen.add(item[0])

    for position,(idx,relevance) in enumerate(ordered):
        c,matched_topics,aspect,redundancy,gain,adaptive=score_candidate(idx,relevance)
        selected=False
        has_new_topic=bool(matched_topics-covered_topics)
        has_new_aspect=aspect in query_aspects and aspect not in covered_aspects
        topic_ok=not query_topics or bool(matched_topics)
        # Keep a relevant chunk when it adds a missing topic/aspect; after coverage is met,
        # only admit strong, non-redundant supporting evidence from a new source.
        if relevance >= .035 and topic_ok and (has_new_topic or has_new_aspect or (adaptive >= .34 and redundancy < .65 and c.get('document_name') not in selected_docs)):
            selected=True; selected_indices.append(idx); covered_aspects.add(aspect); covered_topics.update(matched_topics); selected_docs.add(c.get('document_name'))
        reason=(f'Covers {sorted(matched_topics) or [aspect]}; adds topic/aspect coverage' if selected else ('Rejected: does not match a requested topic' if not topic_ok else ('Rejected: redundant or lower-value evidence' if relevance >= .035 else 'Not relevant enough to support this question')))
        candidates.append({**c,'relevance':round(relevance,3),'normalized_relevance':round(relevance,3),'redundancy':round(redundancy,3),'information_gain':round(gain,3),'evidence_contribution':round(1.0 if has_new_aspect or has_new_topic else .15,3),'adaptive_score':round(adaptive,3),'selected':selected,'aspect':aspect,'matched_topics':sorted(matched_topics),'reason':reason})
        # Stop only after all explicit topics and requested intents are covered.
        topic_cov=topic_ratio(covered_topics,query_topics); aspect_cov=ratio(covered_aspects,query_aspects)
        if (not query_topics or topic_cov >= 1) and aspect_cov >= threshold:
            break
        if len(selected_indices)>=max(6, len(query_topics)+len(query_aspects)+2) and position>=limit-1:
            break
    topic_cov=round(topic_ratio(covered_topics,query_topics),3); aspect_cov=round(ratio(covered_aspects,query_aspects),3)
    coverage=round(min(topic_cov if query_topics else 1.0, aspect_cov),3)
    sufficient=(not query_topics or topic_cov>=1) and (aspect_cov>=threshold or (len(query_aspects)==1 and aspect_cov>=.66)) and bool(selected_indices)
    return {'complexity_score':complexity_score,'complexity_level':complexity_level,'query_aspects':query_aspects,'query_topics':query_topics,'candidates':candidates,'selected':[c for c in candidates if c['selected']],'rejected':[c for c in candidates if not c['selected']],'evidence_coverage':coverage,'topic_coverage':topic_cov,'aspect_coverage':aspect_cov,'sufficient':sufficient,'retrieval_rounds':min(max_rounds,max(1,len(selected_indices))),'weights':WEIGHTS}
