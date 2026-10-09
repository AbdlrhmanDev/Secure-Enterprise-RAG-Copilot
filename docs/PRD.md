# PRD — Secure Enterprise RAG Copilot

## 1. Product Summary

Build a production-style knowledge assistant that answers questions over private company documents using hybrid retrieval, reranking, grounded generation, citations, access control, and measurable retrieval/generation quality.

This is not a "chat with PDF" demo. The project must prove that the system can ingest heterogeneous documents, retrieve relevant evidence, respect document-level permissions, answer with traceable citations, and be evaluated quantitatively.

## 2. Problem

Organizations have policies, manuals, reports, and internal documentation spread across files. Keyword search misses semantic matches, while naive LLM chat can hallucinate or expose content a user should not see.

The product should return concise, source-grounded answers only from documents the requesting user is authorized to access.

## 3. Target Users

- Internal employees searching company knowledge.
- Operations/support teams looking up procedures.
- Engineering teams querying technical documentation.
- Administrators managing document collections and permissions.

## 4. Goals

- Ingest PDF, DOCX, Markdown, HTML, and TXT documents.
- Build hybrid retrieval using dense + sparse/keyword retrieval.
- Add reranking before generation.
- Return answer-level citations to exact source chunks.
- Enforce metadata/document-level ACL filtering before generation.
- Build an evaluation set and measure retrieval and answer quality.
- Expose the system through a FastAPI backend and a lightweight web UI.
- Package and deploy the service using Docker and cloud infrastructure.

## 5. Non-Goals

- General internet search.
- Fully autonomous agents.
- Training a foundation model from scratch.
- Enterprise SSO in the MVP; simulated users/roles are acceptable.

## 6. Core User Stories

1. As a user, I can upload or sync documents into a knowledge base.
2. As a user, I can ask a natural-language question and receive an answer with citations.
3. As a user, I can open a citation and see the supporting passage and document metadata.
4. As an authorized user, I only retrieve chunks from documents I am permitted to access.
5. As an admin, I can inspect ingestion status and failed documents.
6. As an engineer, I can run an evaluation suite and compare retrieval configurations.

## 7. Functional Requirements

### 7.1 Ingestion

- Accept PDF, DOCX, Markdown, HTML, TXT.
- Extract text and basic metadata.
- Clean repeated headers/footers where possible.
- Chunk using a configurable strategy.
- Store:
  - document_id
  - chunk_id
  - source
  - page/section
  - text
  - role/ACL metadata
  - embedding
- Support idempotent re-ingestion by document hash/version.

### 7.2 Retrieval

Implement at least three retrievers for comparison:

- BM25 / lexical.
- Dense vector retrieval.
- Hybrid retrieval.

Add a reranker to the final candidate set.

Configuration must expose:
- top_k
- chunk size
- overlap
- fusion weights or RRF
- reranker candidate count
- final context count

### 7.3 Generation

The answer pipeline must:
- use only retrieved context;
- state when evidence is insufficient;
- include inline citations;
- expose cited chunk IDs;
- avoid leaking unauthorized content.

### 7.4 Security

- Simulated JWT authentication.
- Roles such as employee, finance, engineering, admin.
- Retrieval filters applied before results are passed to the LLM.
- Test cases proving unauthorized chunks cannot enter the context.

### 7.5 Evaluation

Create a labeled dataset with at least 75–150 queries.

Retrieval metrics:
- Recall@K
- Precision@K
- MRR
- nDCG@K (recommended)

Generation metrics:
- answer correctness
- groundedness / faithfulness
- citation correctness
- refusal when evidence is absent

Track latency and cost:
- retrieval latency
- reranking latency
- generation latency
- end-to-end p50/p95
- token usage and estimated cost/query

### 7.6 UI

Minimum interface:
- chat
- citations drawer
- document upload
- collection selector
- evaluation dashboard

## 8. Suggested Architecture

Client
→ FastAPI API
→ Auth/ACL middleware
→ Query pipeline
→ Dense retriever + BM25 retriever
→ Fusion
→ Reranker
→ Context builder
→ LLM
→ Citation formatter

Ingestion:
Upload
→ Parser
→ Cleaner
→ Chunker
→ Embedding worker
→ Vector DB + metadata store

## 9. Suggested Tech Stack

- Python 3.12+
- FastAPI
- Pydantic
- PostgreSQL
- Qdrant or pgvector
- BM25 via Elasticsearch/OpenSearch or local rank_bm25 for MVP
- Sentence Transformers or provider embeddings
- Cross-encoder / ColBERT-style reranker
- OpenAI / Anthropic / open-weight model for generation
- Redis for caching
- Celery/RQ/Arq for background ingestion
- Streamlit or Next.js for UI
- Docker + Docker Compose
- AWS: ECS/Fargate or EC2, S3, RDS, CloudWatch

## 10. API Surface

### POST /documents
Upload a document.

### GET /documents/{id}
Return document metadata and ingestion status.

### POST /query
Input:
```json
{
  "query": "What is the refund approval policy?",
  "user_id": "u_123",
  "collection_id": "policies"
}
```

Output:
```json
{
  "answer": "...",
  "citations": [
    {
      "document_id": "doc_1",
      "chunk_id": "chunk_28",
      "page": 7,
      "score": 0.91
    }
  ],
  "latency_ms": 842
}
```

### POST /eval/run
Run an evaluation configuration.

### GET /eval/runs/{id}
Return metrics and experiment configuration.

## 11. Data Model

### documents
- id
- filename
- source_uri
- checksum
- version
- status
- allowed_roles
- created_at

### chunks
- id
- document_id
- text
- page
- section
- token_count
- metadata

### eval_cases
- id
- query
- relevant_chunk_ids
- expected_answer
- tags

### eval_runs
- id
- config_json
- recall_at_k
- mrr
- groundedness
- p95_latency_ms
- cost_per_query

## 12. Non-Functional Requirements

- API errors must be structured.
- Ingestion jobs must be retryable.
- Query path must log trace IDs.
- Secrets must use environment variables/secret manager.
- Unit + integration tests required.
- p95 latency target should be documented and measured.
- No document text in production logs by default.

## 13. MVP Acceptance Criteria

The MVP is done when:

- 500+ pages can be indexed.
- Hybrid retrieval and reranking are implemented.
- At least 75 labeled queries exist.
- Evaluation script outputs retrieval metrics.
- Answers include working citations.
- ACL tests demonstrate no unauthorized retrieval.
- Docker Compose starts the complete local system.
- FastAPI docs are available.
- The system is deployed or has a reproducible cloud deployment guide.

## 14. Stretch Goals

- Query rewriting / decomposition.
- Parent-child retrieval.
- Multi-query retrieval.
- Semantic caching.
- Feedback collection and active evaluation set growth.
- Multi-tenant collections.
- AWS Bedrock Knowledge Bases comparison benchmark.
- OpenTelemetry tracing.

## 15. Repository Structure

```text
secure-rag-copilot/
├── app/
│   ├── api/
│   ├── auth/
│   ├── ingestion/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   └── evaluation/
├── tests/
├── evals/
│   ├── dataset.jsonl
│   └── experiments/
├── scripts/
├── docker/
├── docs/
│   └── architecture.md
├── docker-compose.yml
├── pyproject.toml
└── README.md
```
