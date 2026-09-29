# CARE-RAG

**Context-Adaptive Retrieval with Evidence-Aware Selection for Question Answering**

CARE-RAG is a working local full-stack application. It ingests PDF, DOCX, and TXT files, persists documents and chunks in SQLite, builds a local TF-IDF embedding index, retrieves candidates with cosine similarity, scores redundancy and information gain, tracks evidence coverage, stops when evidence is sufficient, and returns grounded answers with citations.

## Run the application

Install Python dependencies:

```bash
cd /home/ubuntu/evidence-aware-rag
python3 -m pip install -r requirements.txt
```

Terminal 1 — backend:

```bash
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Terminal 2 — frontend:

```bash
pnpm install
pnpm dev
```

Open `http://localhost:5173`. The Vite development server proxies `/api` requests to FastAPI on port 8000.

## Completed application features

The **Knowledge Base** page accepts PDF, DOCX, or TXT uploads, extracts text, preserves page metadata when available, chunks content, writes documents and chunks to `care_rag.db`, rebuilds embeddings, prevents duplicate filenames, and supports deletion. Data survives backend restarts.

The **Ask Question** page makes a live `POST /api/ask` call. CARE-RAG analyzes query complexity, retrieves a candidate pool, calculates semantic relevance, redundancy, information gain, evidence contribution, and a transparent adaptive score, then stops when the configured sufficiency rule is met. Rejected chunks are not passed to generation.

The **Retrieval Analysis** page exposes the full persisted trace: every candidate, selected/rejected state, source chunk, page, decision reason, coverage, redundancy, information gain, and configured weights.

The **Evaluation** page runs a live comparison between Fixed Top-K, Similarity Adaptive, and CARE-RAG. It measures Recall@K, MRR, nDCG, faithfulness, answer correctness, context tokens, redundancy, evidence coverage, and latency against the built-in relevance labels. For a custom uploaded corpus, evaluation metrics should be interpreted as retrieval measurements until a labeled test set is supplied.

## Optional LLM generation

CARE-RAG works without a remote model using a safe extractive fallback. To enable grounded OpenAI-compatible generation, copy `.env.example` to `.env` and provide `OPENAI_API_KEY`, optionally changing `OPENAI_API_BASE` and `CARE_RAG_LLM_MODEL`. The LLM receives only selected evidence and is instructed to cite chunk IDs. If the provider fails, CARE-RAG falls back to extractive grounded output.

## Architecture

```text
backend/
├── app.py
├── storage/
│   └── sqlite_store.py
├── adaptive_retrieval/
│   ├── complexity.py
│   ├── relevance.py
│   ├── redundancy.py
│   ├── information_gain.py
│   ├── evidence_coverage.py
│   ├── sufficiency.py
│   └── selector.py
├── generation/
│   └── generator.py
└── evaluation/
    └── metrics.py
```

## Adaptive scoring

```text
adaptive_score =
  0.40 * relevance +
  0.20 * information_gain +
  0.25 * evidence_coverage_contribution +
  0.15 * low_redundancy
```

`max_rounds` and `sufficiency_threshold` are configurable through the API. The default maximum is three rounds and the default threshold is 0.85. The loop is bounded and cannot run forever.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Backend health and persistence status |
| `GET /api/stats` | Persistent document, chunk, and query metrics |
| `GET /api/documents` | Current knowledge base |
| `POST /api/documents/upload` | Upload and process PDF, DOCX, or TXT files |
| `DELETE /api/documents/{name}` | Delete a document and rebuild the index |
| `POST /api/ask` | Run CARE-RAG retrieval and grounded generation |
| `GET /api/trace` | Retrieve the most recent persisted selection trace |
| `POST /api/evaluate` | Compare three retrieval methods with measured metrics |

## Research positioning

CARE-RAG does not claim adaptive RAG itself is new. It is an evidence-aware combination and implementation of existing signals for context selection, with an explicit decision trace and evaluation surface. The research question is: **how can a retrieval system determine whether it has enough diverse, relevant evidence to answer?**
