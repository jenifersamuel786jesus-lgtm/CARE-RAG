from __future__ import annotations
import hashlib, json, sqlite3, time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / 'care_rag.db'

class SQLiteStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()
    def init_schema(self):
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS documents(name TEXT PRIMARY KEY, type TEXT NOT NULL, pages INTEGER NOT NULL, chunks INTEGER NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, document_name TEXT NOT NULL REFERENCES documents(name) ON DELETE CASCADE, page_number TEXT NOT NULL, text TEXT NOT NULL, embedding TEXT);
        CREATE TABLE IF NOT EXISTS queries(id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, selected INTEGER NOT NULL, coverage REAL NOT NULL, latency REAL NOT NULL, context_tokens INTEGER NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, created_at REAL NOT NULL);
        ''')
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.commit()
    def document_count(self): return self.conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
    def add_document(self, name, doc_type, pages, chunks):
        self.conn.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?,?,?,?)', (name, doc_type, pages, len(chunks), 'Processed', time.time()))
        self.conn.execute('DELETE FROM chunks WHERE document_name=?', (name,))
        for c in chunks:
            self.conn.execute('INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?)', (c['chunk_id'], name, str(c['page_number']), c['text'], json.dumps(c.get('embedding')) if c.get('embedding') is not None else None))
        self.conn.commit()
    def delete_document(self, name):
        self.conn.execute('DELETE FROM chunks WHERE document_name=?', (name,)); self.conn.execute('DELETE FROM documents WHERE name=?', (name,)); self.conn.commit()
    def documents(self): return [dict(r) for r in self.conn.execute('SELECT name,type,pages,chunks,status FROM documents ORDER BY created_at DESC')]
    def chunks(self):
        rows = self.conn.execute('SELECT chunk_id,document_name,page_number,text,embedding FROM chunks ORDER BY rowid').fetchall()
        out=[]
        for r in rows:
            d=dict(r); d['page_number']=int(d['page_number']) if str(d['page_number']).isdigit() else d['page_number']; d['embedding']=json.loads(d['embedding']) if d['embedding'] else None; out.append(d)
        return out
    def normalize_chunk_ids(self):
        rows=self.conn.execute('SELECT chunk_id,document_name,page_number FROM chunks ORDER BY document_name,rowid').fetchall()
        if not rows: return
        mapping=[]; counters={}
        for r in rows:
            key=hashlib.sha1(r['document_name'].encode('utf-8')).hexdigest()[:8].upper()
            counter=counters.get((r['document_name'],str(r['page_number'])),0)+1; counters[(r['document_name'],str(r['page_number']))]=counter
            mapping.append((r['chunk_id'],f'{key}-{r["page_number"]}-{counter:02d}'))
        if len({new for _,new in mapping}) != len(mapping): return
        self.conn.executemany('UPDATE chunks SET chunk_id=? WHERE chunk_id=?', [(f'__migrate__{i}',old) for i,(old,_) in enumerate(mapping)])
        self.conn.executemany('UPDATE chunks SET chunk_id=? WHERE chunk_id=?', [(new,f'__migrate__{i}') for i,(_,new) in enumerate(mapping)])
        self.conn.commit()
    def update_embeddings(self, embeddings):
        self.conn.executemany('UPDATE chunks SET embedding=? WHERE chunk_id=?', [(json.dumps(vec), cid) for cid,vec in embeddings.items()]); self.conn.commit()
    def add_query(self, query, selected, coverage, latency, context_tokens):
        self.conn.execute('INSERT INTO queries(query,selected,coverage,latency,context_tokens,created_at) VALUES(?,?,?,?,?,?)',(query,selected,coverage,latency,context_tokens,time.time())); self.conn.commit()
    def stats(self):
        row=self.conn.execute('SELECT COUNT(*) documents, COALESCE(SUM(chunks),0) chunks FROM documents').fetchone()
        q=self.conn.execute('SELECT COUNT(*) queries, COALESCE(AVG(selected),0) avg_selected, COALESCE(AVG(coverage),0) avg_coverage, COALESCE(AVG(latency),0) avg_latency, COALESCE(AVG(context_tokens),0) avg_tokens FROM queries').fetchone()
        return {'documents':row['documents'],'chunks':row['chunks'],'queries':q['queries'],'avg_selected_chunks':round(q['avg_selected'],1) if q['queries'] else 0,'avg_context_tokens':round(q['avg_tokens']) if q['queries'] else 0,'avg_retrieval_latency':round(q['avg_latency'],1) if q['queries'] else 0,'avg_evidence_coverage':round(q['avg_coverage'],3) if q['queries'] else 0}
    def save_trace(self, trace): self.conn.execute('INSERT OR REPLACE INTO traces(id,payload,created_at) VALUES(1,?,?)',(json.dumps(trace),time.time())); self.conn.commit()
    def last_trace(self):
        r=self.conn.execute('SELECT payload FROM traces WHERE id=1').fetchone(); return json.loads(r['payload']) if r else None
