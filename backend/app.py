from __future__ import annotations
import hashlib, io, re, time
from pathlib import Path
from typing import Any
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from .storage.sqlite_store import SQLiteStore
from .adaptive_retrieval.selector import select
from .generation.generator import generate
from .evaluation.metrics import relevant_ids, retrieval, answer_quality

app=FastAPI(title='CARE-RAG API',version='2.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
STORE=SQLiteStore(); INDEX={'vectorizer':None,'matrix':None}
SEED_DOCUMENTS={'Hadoop Fundamentals.pdf':[('1','Hadoop is an open-source framework for distributed storage and processing of large datasets across clusters of commodity hardware.'),('3','Hadoop Distributed File System, or HDFS, splits files into blocks and replicates them across nodes for fault tolerance and high-throughput access.'),('5','Hadoop scales horizontally by adding inexpensive nodes. This makes it cost efficient for organizations whose data footprint grows over time.'),('8','YARN separates resource management from processing, allowing multiple data-processing engines to share a Hadoop cluster.'),('11','Common Hadoop applications include log processing, ETL pipelines, recommendation systems, and large-scale data warehousing.')],'Spark vs MapReduce.docx':[('2','MapReduce writes intermediate results to disk between stages, which improves fault tolerance but can introduce latency for iterative workloads.'),('4','Apache Spark keeps intermediate datasets in memory when possible, making it particularly effective for iterative machine-learning algorithms and interactive analytics.'),('5','Spark provides a unified engine for SQL, streaming, machine learning, and graph workloads, while MapReduce is centered on batch processing.')],'Distributed Systems Notes.txt':[('1','Distributed systems coordinate independent computers so they appear to users as a single coherent system.'),('2','Horizontal scaling allows teams to add inexpensive nodes as their data footprint grows, avoiding reliance on one high-end server.'),('4','Replication, partitioning, and retry mechanisms improve availability and help systems continue operating when individual machines fail.')]}
class AskRequest(BaseModel):
    query:str=Field(min_length=1); candidate_pool:int=Field(default=10,ge=3,le=50); max_rounds:int=Field(default=3,ge=1,le=5); sufficiency_threshold:float=Field(default=.85,ge=.4,le=1)

def make_chunk(name,page,text,index):
    doc_key=hashlib.sha1(name.encode('utf-8')).hexdigest()[:8].upper()
    return {'chunk_id':f'{doc_key}-{page}-{index:02d}','document_name':name,'page_number':page,'text':text.strip()}

def split_text(text, max_words=140, overlap=24):
    words=re.findall(r'\S+', text.strip())
    if not words: return []
    if len(words)<=max_words: return [' '.join(words)]
    out=[]; start=0
    while start<len(words):
        end=min(len(words),start+max_words); out.append(' '.join(words[start:end]))
        if end==len(words): break
        start=max(start+1,end-overlap)
    return out

def build_chunks(name,pairs):
    chunks=[]; seen=set(); index=1
    for page,text in pairs:
        for part in split_text(text):
            normalized=re.sub(r'\s+',' ',part).strip().lower()
            if len(normalized)<20 or normalized in seen: continue
            seen.add(normalized); chunks.append(make_chunk(name,page,part,index)); index+=1
    return chunks
def rebuild_index():
    chunks=STORE.chunks(); texts=[c['text'] for c in chunks]
    if not texts: INDEX['vectorizer']=INDEX['matrix']=None; return
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True); matrix=vectorizer.fit_transform(texts); INDEX.update(vectorizer=vectorizer,matrix=matrix)
    STORE.update_embeddings({c['chunk_id']:matrix[i].toarray()[0].tolist() for i,c in enumerate(chunks)})
def add_seed_data():
    if STORE.document_count(): return
    for name,pairs in SEED_DOCUMENTS.items(): STORE.add_document(name,Path(name).suffix[1:].upper(),len(pairs),[make_chunk(name,p,t,i) for i,(p,t) in enumerate(pairs,1)])
def extract_upload(name,data):
    suffix=Path(name).suffix.lower()
    if suffix=='.txt': return [('1',data.decode('utf-8',errors='ignore'))]
    if suffix=='.pdf':
        reader=PdfReader(io.BytesIO(data)); return [(str(i+1),page.extract_text() or '') for i,page in enumerate(reader.pages) if (page.extract_text() or '').strip()]
    if suffix=='.docx':
        doc=DocxDocument(io.BytesIO(data)); return [('1','\n'.join(p.text for p in doc.paragraphs if p.text.strip()))]
    raise HTTPException(415,'Unsupported document. Upload PDF, DOCX, or TXT.')
def run_retrieval(request:AskRequest):
    if not STORE.chunks(): raise HTTPException(400,'No processed evidence is available. Upload a document first.')
    started=time.perf_counter(); chunks=STORE.chunks(); trace=select(request.query.strip(),chunks,INDEX['vectorizer'],INDEX['matrix'],request.candidate_pool,request.max_rounds,request.sufficiency_threshold); answer,source=generate(request.query,trace['selected'],trace['sufficient']); trace.update(query=request.query.strip(),answer=answer,generation_source=source,context_tokens=len(' '.join(c['text'] for c in trace['selected']).split()),retrieval_latency_ms=round((time.perf_counter()-started)*1000,1)); STORE.save_trace(trace); STORE.add_query(request.query.strip(),len(trace['selected']),trace['evidence_coverage'],trace['retrieval_latency_ms'],trace['context_tokens']); return trace
@app.on_event('startup')
def startup(): STORE.normalize_chunk_ids(); add_seed_data(); rebuild_index()
@app.get('/api/health')
def health(): return {'status':'ok','service':'CARE-RAG','version':'2.0.0','persistent_storage':str(STORE.path)}
@app.get('/api/stats')
def stats(): return STORE.stats()
@app.get('/api/documents')
def documents(): return STORE.documents()
@app.post('/api/documents/upload')
async def upload_documents(files:list[UploadFile]=File(...)):
    results=[]
    pending=[]
    existing_names={d['name'] for d in STORE.documents()}
    incoming_names=set()
    for file in files:
        data=await file.read()
        if not data: raise HTTPException(400,f'{file.filename} is empty.')
        if len(data)>25*1024*1024: raise HTTPException(413,f'{file.filename} is larger than the 25 MB upload limit.')
        if file.filename in existing_names or file.filename in incoming_names: raise HTTPException(409,f'{file.filename} is already uploaded or duplicated in this batch.')
        try:
            pairs=extract_upload(file.filename,data)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422,f'Could not read {file.filename}. Check that it is a valid PDF, DOCX, or TXT file.') from exc
        if not pairs or not any(t.strip() for _,t in pairs): raise HTTPException(400,f'{file.filename} contains no extractable text.')
        chunks=build_chunks(file.filename,pairs)
        if not chunks: raise HTTPException(400,f'{file.filename} contains no usable text after chunking.')
        pending.append((file.filename,Path(file.filename).suffix[1:].upper(),len(set(str(p) for p,_ in pairs)),chunks)); incoming_names.add(file.filename)
    for name,doc_type,pages,chunks in pending:
        STORE.add_document(name,doc_type,pages,chunks); results.append(next(d for d in STORE.documents() if d['name']==name))
    rebuild_index(); return {'documents':results}
@app.delete('/api/documents/{name}')
def delete_document(name:str):
    if not any(d['name']==name for d in STORE.documents()): raise HTTPException(404,'Document not found.')
    STORE.delete_document(name); rebuild_index(); return {'ok':True}
@app.post('/api/ask')
def ask(request:AskRequest): return run_retrieval(request)
@app.get('/api/trace')
def trace(): return STORE.last_trace() or {'candidates':[],'selected':[],'rejected':[]}
@app.post('/api/evaluate')
def evaluate(request:AskRequest):
    trace=run_retrieval(request); allc=trace['candidates']; relevant=relevant_ids(request.query,STORE.chunks()); ranked=sorted(allc,key=lambda x:x['relevance'],reverse=True); fixed=ranked[:5]; similarity=ranked[:max(3,min(6,len(ranked)))]; methods=[]
    for name,selected in [('Fixed Top-K',fixed),('Similarity Adaptive',similarity),('CARE-RAG',trace['selected'])]:
        metrics={**retrieval(selected,ranked,relevant),**answer_quality(generate(request.query,selected,True)[0],selected,request.query)}; methods.append({'method':name,'selected_chunks':len(selected),'context_tokens':sum(len(c['text'].split()) for c in selected),'evidence_coverage':round(min(1,sum(c['evidence_contribution'] for c in selected)/max(1,len(trace['query_aspects']))),3),'redundancy':round(sum(c['redundancy'] for c in selected)/max(1,len(selected)),3),'retrieval_latency_ms':trace['retrieval_latency_ms'],'answer':generate(request.query,selected,True)[0],'sources':selected,'metrics':metrics})
    return {'query':request.query,'relevant_chunk_count':len(relevant),'results':methods}
