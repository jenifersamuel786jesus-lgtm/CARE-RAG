from __future__ import annotations
import json, os, re, urllib.request

def extractive(query: str, chosen: list[dict], sufficient: bool) -> str:
    if not chosen:
        return 'I could not find supporting evidence for this question in the uploaded documents. Please upload a relevant document, then ask again.'
    if not sufficient:
        prefix='I found only partial evidence in the uploaded documents. The points below are supported, but the evidence is not sufficient to answer every part of the question:\n'
    else:
        prefix=''
    stop_words = {'what', 'which', 'where', 'when', 'that', 'this', 'with', 'from', 'into', 'about', 'explain', 'compare', 'advantages', 'benefits'}
    query_lower = query.lower()
    def citation(chunk):
        return f"({chunk.get('document_name', 'uploaded document')}, p. {chunk.get('page_number', '—')})"
    entity_terms = [term for term in ('hadoop', 'spark', 'mapreduce', 'kafka', 'nat') if term in query_lower]
    query_terms = (set(re.findall(r'\b[a-z]{4,}\b', query_lower)) - stop_words) | set(entity_terms)
    if 'advantage' in query_lower or 'benefit' in query_lower:
        query_terms.update({'scale', 'cost', 'fault', 'efficient', 'memory', 'performance'})
    if any(term in query_lower for term in ('application', 'used', 'where')):
        query_terms.update({'application', 'pipeline', 'warehousing', 'analytics', 'workload'})
    if 'nat' in query_lower and 'type' in query_lower:
        nat_points = []
        for chunk in chosen:
            for raw in chunk['text'].splitlines():
                line = re.sub(r'^[-•▪\s]+', '', raw).strip()
                if (re.search(r'\bstatic nat\b', line, re.I) or re.search(r'\bdynamic nat\b', line, re.I) or re.search(r'\bpat\b', line, re.I)) and line not in nat_points:
                    nat_points.append(f"- {line} {citation(chunk)}")
        if nat_points:
            return 'NAT types identified in the selected evidence:\n' + '\n'.join(nat_points[:4])
    scored = []
    for chunk in chosen:
        chunk_text = f"{chunk.get('document_name', '')} {chunk['text']}".lower()
        if entity_terms and not any(term in chunk_text for term in entity_terms):
            continue
        sentences = re.split(r'(?<=[.!?])\s+|\n+', chunk['text'].strip())
        for sentence in sentences:
            sentence = re.sub(r'^[-•▪\s]+', '', sentence).strip()
            if not sentence or len(sentence) < 30 or len(sentence) > 240:
                continue
            if len(sentence.split()) < 8 and not sentence.endswith(('.', '!', '?')):
                continue
            if sentence.count('•') >= 2 or sentence.lower().startswith(('unit ', 'chapter ', 'table of contents')) or sentence.isupper():
                continue
            words = set(re.findall(r'\b[a-z]{3,}\b', sentence.lower()))
            overlap = len(words & query_terms)
            if query_terms and overlap == 0:
                continue
            anchor_bonus = 2 if any(term in sentence.lower() for term in query_terms if len(term) >= 5) else 0
            scored.append((overlap + anchor_bonus, len(sentence), sentence, citation(chunk)))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    points = []
    seen = set()
    for _, _, sentence, source in scored:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        points.append(f'- {sentence} {source}')
        if len(points) == 4:
            break
    if not points:
        return 'This topic is not covered clearly by the uploaded documents. Please upload a relevant document, then ask again.'
    return prefix + 'Key points from the selected evidence:\n' + '\n'.join(points)

def generate(query: str, chosen: list[dict], sufficient: bool) -> tuple[str, str]:
    fallback=extractive(query,chosen,sufficient)
    key=os.getenv('OPENAI_API_KEY')
    base=os.getenv('OPENAI_API_BASE','https://api.openai.com/v1').rstrip('/')
    model=os.getenv('CARE_RAG_LLM_MODEL','gpt-4o-mini')
    if not key or not sufficient or not chosen: return fallback, 'extractive'
    context='\n\n'.join(f"[{c.get('document_name')} — page {c.get('page_number')}] {c['text']}" for c in chosen)
    body={'model':model,'temperature':0.1,'messages':[{'role':'system','content':'You are CARE-RAG. Answer only from the supplied evidence. If evidence is insufficient, say so. Return a concise answer in no more than 4 short bullet points. Cite the document name and page in plain language. Never expose internal chunk IDs or copy the full source text.'},{'role':'user','content':f'Question: {query}\n\nEvidence:\n{context}'}]}
    try:
        request=urllib.request.Request(base+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=30) as response:
            payload=json.load(response); answer=payload['choices'][0]['message']['content'].strip()
            return answer, 'llm'
    except Exception:
        return fallback, 'extractive-fallback'
