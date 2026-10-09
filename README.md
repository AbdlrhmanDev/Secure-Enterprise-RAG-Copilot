# Secure Enterprise RAG Copilot

Hybrid retrieval (BM25 + dense), cross-encoder reranking, grounded answers with citations,
document-level ACLs and an evaluation suite. Requirements are in `docs/PRD.md`.

## Run locally (no Docker)

    uv sync
    uv run python -m scripts.generate_corpus      # demo corpus + evals/dataset.jsonl
    uv run python -m scripts.ingest_corpus        # 125 documents, ~630 pages
    uv run uvicorn app.main:app                   # UI at http://localhost:8000, API docs at /docs

Local mode uses SQLite and embedded Qdrant (one process at a time: stop the API before
running the ingest or eval scripts). Set `OPENAI_API_KEY` to answer with OpenAI;
without it an offline extractive generator is used.

## Run with Docker

    docker compose up --build
    docker compose exec api python -m scripts.generate_corpus
    docker compose exec api python -m scripts.ingest_corpus

## Tests and evaluation

    uv run pytest
    uv run python -m scripts.run_eval --all                # results in evals/experiments/results/
    uv run python -m scripts.run_eval --all --no-generate  # retrieval metrics only

Demo users (`POST /auth/token`): `u_employee`, `u_finance`, `u_engineer`, `u_admin`.

## What it does

- **Ingestion**: PDF, DOCX, Markdown, HTML, TXT. Header/footer cleaning, section-aware chunking
  (`CHUNK_SIZE`, `CHUNK_OVERLAP`), idempotent re-ingestion by checksum and version, retryable jobs.
- **Retrieval**: BM25, dense and hybrid (RRF or weighted fusion) plus a cross-encoder reranker.
  `top_k`, fusion weights, reranker candidate count and final context count are configurable
  per request and per experiment.
- **Generation**: answers only from retrieved context, inline `[n]` citations mapped to chunk
  ids, explicit refusal when evidence is missing.
- **Security**: JWT auth for simulated users, role-based document ACLs enforced inside each
  retriever and re-checked before reranking. See `docs/architecture.md`.
- **Evaluation**: 136 labeled cases (98 answerable of which 54 are paraphrased, 12 unanswerable,
  26 ACL), retrieval and answer metrics, latency and cost.
- **UI**: chat, citations drawer, upload, collection selector, evaluation dashboard.

## How it works

### 1. A document is uploaded

1. A signed-in user sends a file to `POST /documents` with a collection name and the roles
   allowed to read it. Users can only restrict a document to roles they hold themselves.
2. The API checks the file type and size, computes a SHA-256 checksum and looks up the document
   by collection and filename. Identical content is skipped; changed content becomes a new version.
3. The file is saved and an ingestion job is queued (a Redis worker in Docker, a background task
   locally). The API returns immediately with status `pending`.

### 2. The document is ingested

1. **Parse**: a parser per format (PDF, DOCX, Markdown, HTML, TXT) produces text blocks, each
   tagged with its page and section heading.
2. **Clean**: running headers, footers and page numbers that repeat across pages are removed.
3. **Chunk**: sentences are packed into chunks of about 220 tokens with 40 tokens of overlap.
   A chunk never crosses a section boundary.
4. **Embed**: each chunk (prefixed with its document title and section) is turned into a
   384-dimension vector.
5. **Store**: chunk text and metadata go to PostgreSQL; vectors go to Qdrant together with the
   collection and the allowed roles. Chunks of the previous version are deleted first.
6. The document becomes `ready`, or `failed` with an error message. Failures other than parse
   errors are retried.

### 3. A question is asked

1. **Authenticate**: `POST /query` carries a JWT. The user's roles are looked up from the user
   directory, never taken from the request body.
2. **Scope**: the system lists the documents in the collection that this user may read.
   Everything after this step is limited to that list.
3. **Retrieve**: two searches run over the authorized chunks only. BM25 finds exact word
   matches; dense search in Qdrant finds passages with similar meaning, filtered by role.
4. **Fuse**: the two ranked lists are merged with reciprocal rank fusion (or a weighted sum).
5. **Re-check**: any candidate whose document is not on the authorized list is dropped. This
   protects against a stale or wrong ACL in the vector index.
6. **Rerank**: a cross-encoder reads the question together with each of the top 20 candidates
   and rescores them. The best 5 become the context.
7. **Generate**: the model receives the 5 passages as numbered sources and must answer only
   from them, marking each claim with `[n]`, or say that the evidence is insufficient.
8. **Cite**: each `[n]` is mapped back to its chunk. Markers pointing at anything outside the
   context are removed. The response contains the answer, the citations with document, page,
   section and passage text, timings, token usage and estimated cost.
9. **Cache and log**: the answer is cached under a key that includes the user's roles and the
   versions of the documents they can see. The log line holds ids and timings, not text.

### 4. Quality is measured

1. Each of the 136 evaluation cases names a question, the user asking it, the expected answer
   and a quote from the source document. Quotes are matched to chunk ids at run time.
2. The runner sends every case through the same query path as the API, for each retrieval
   configuration in `evals/experiments/`.
3. It reports Recall@K, Precision@K, MRR and nDCG@K for answerable questions, answer
   correctness, groundedness and citation correctness for the answers, refusal accuracy for
   questions that must be declined, the number of leaked chunks, latency percentiles and cost.

Diagrams and the security model are in `docs/architecture.md`.

## Measured results

Corpus: 125 synthetic documents, 630 pages, 1,900 chunks (chunk size 220, overlap 40).
K = 10, 98 answerable cases, 38 cases that must be refused. Embedder `bge-small-en-v1.5`,
reranker `ms-marco-MiniLM-L-6-v2`, CPU only. Reproduce with `python -m scripts.run_eval --all`.

| Experiment | Recall@10 | MRR | nDCG@10 | ACL leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.663 | 0.576 | 0.597 | 0 | 8.5 | 16.9 |
| Dense | 0.949 | 0.889 | 0.904 | 0 | 142.1 | 221.7 |
| Hybrid (RRF) | 0.949 | 0.775 | 0.816 | 0 | 71.2 | 168.4 |
| Hybrid (weighted 0.6/0.4) | 0.959 | 0.860 | 0.885 | 0 | 76.0 | 183.6 |
| Dense + rerank | 0.959 | 0.897 | 0.912 | 0 | 771.0 | 1304.9 |
| Hybrid (RRF) + rerank | 0.959 | 0.906 | 0.919 | 0 | 827.4 | 1156.1 |

How to read this:

- The corpus is synthetic and small, and the filler documents are templated. Treat the numbers
  as a working benchmark harness, not as evidence about real company documents.
- Latency was measured on a laptop while other containers were running, with the offline
  extractive generator, so it covers retrieval and reranking only. Reranking 20 candidates on
  CPU is nearly all of it.
- Answer metrics in this run come from the extractive fallback generator and lexical
  heuristics: answer correctness 0.43-0.44, citation correctness 0.97-0.99, refusal accuracy
  0.76-0.82. The extractive generator cannot handle paraphrased questions, which is most of the
  correctness gap. The OpenAI generator and the LLM judge have not been run (no API key was
  available), so there are no token or cost figures yet.

Full per-case output is in `evals/experiments/results/`.

## Configuration

All settings are environment variables (see `app/config.py`). The ones you are most likely to change:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | unset | Enables OpenAI for generation and the LLM judge |
| `OPENAI_MODEL` | `gpt-5-mini` | Generation model |
| `LLM_INPUT_PRICE_PER_MTOK`, `LLM_OUTPUT_PRICE_PER_MTOK` | `0.25`, `2.0` | Prices used for the cost estimate; set to your model's current list price |
| `RETRIEVAL_MODE` | `hybrid` | `bm25`, `dense` or `hybrid` |
| `RERANK` | `true` | Cross-encoder reranking |
| `MIN_RELEVANCE_SCORE` | `0` | Refuse without calling the LLM when the best reranked chunk scores below this |
| `INGEST_BACKEND` | `background` | `arq` (Redis worker), `background` or `inline` |
| `API_PORT` | `8000` | Host port for the API in Docker Compose |
| `JWT_SECRET` | dev value | Must be set when `ENV=prod` |

## Layout

    app/api          HTTP routes          app/retrieval    BM25, Qdrant, fusion, pipeline
    app/auth         users, JWT, ACL      app/reranking    cross-encoder
    app/ingestion    parse, chunk, embed  app/generation   prompts, LLM, citations
    app/evaluation   dataset, metrics     app/web          static UI
    scripts/         corpus, ingest, eval evals/           dataset.jsonl, experiments/
    docs/            architecture.md, deployment-aws.md

## Not done

Query rewriting, parent-child and multi-query retrieval, semantic caching, OpenTelemetry and the
Bedrock comparison (PRD stretch goals). There are no database migrations.
