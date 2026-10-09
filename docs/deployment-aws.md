# Deploying to AWS (ECS Fargate)

This guide has not been executed end to end; the container image and Compose stack it is based
on have been built and run locally. Replace `<...>` placeholders.

## Target layout

| Piece | AWS service |
|---|---|
| `api` and `worker` containers (same image) | ECS Fargate, two services |
| PostgreSQL | RDS for PostgreSQL 16 |
| Redis | ElastiCache |
| Qdrant | ECS service with an EFS volume, or Qdrant Cloud |
| Uploaded files and model cache (`/data`) | EFS mounted into `api` and `worker` |
| Secrets | Secrets Manager |
| Logs | CloudWatch Logs (the app writes JSON to stdout) |
| Ingress | Application Load Balancer with an ACM certificate |

Uploads are written to `STORAGE_DIR` on a shared filesystem, which is why EFS is used. Storing
them in S3 instead needs a small change in `app/ingestion/pipeline.py` (where the file is written
and read back).

## 1. Image

```bash
aws ecr create-repository --repository-name secure-rag-copilot
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -f docker/Dockerfile -t <account>.dkr.ecr.<region>.amazonaws.com/secure-rag-copilot:v1 .
docker push <account>.dkr.ecr.<region>.amazonaws.com/secure-rag-copilot:v1
```

## 2. Data services

- RDS PostgreSQL 16 in private subnets; create database `rag`.
- ElastiCache Redis, single node is enough to start.
- Qdrant: run `qdrant/qdrant` as an ECS service with `/qdrant/storage` on EFS, registered in
  Cloud Map as `qdrant.<namespace>` so the app can reach `http://qdrant.<namespace>:6333`.
- EFS file system with an access point for `/data`.

Security groups: only the ECS tasks may reach RDS (5432), Redis (6379), Qdrant (6333) and EFS (2049).

## 3. Secrets

```bash
aws secretsmanager create-secret --name rag/jwt-secret --secret-string "$(openssl rand -hex 32)"
aws secretsmanager create-secret --name rag/openai-api-key --secret-string "<key>"
aws secretsmanager create-secret --name rag/database-url \
  --secret-string "postgresql+psycopg://<user>:<password>@<rds-endpoint>:5432/rag"
```

## 4. Task definitions

Both tasks use the same image, 2 vCPU / 4 GB (the embedding and reranking models run on CPU),
the EFS volume mounted at `/data`, and the `awslogs` driver.

Environment:

```
ENV=prod
QDRANT_URL=http://qdrant.<namespace>:6333
REDIS_URL=redis://<elasticache-endpoint>:6379
INGEST_BACKEND=arq
STORAGE_DIR=/data/uploads
MODEL_CACHE_DIR=/data/models
```

Secrets (from Secrets Manager): `DATABASE_URL`, `JWT_SECRET`, `OPENAI_API_KEY`.
With `ENV=prod` the app refuses to start on the default JWT secret.

- `api`: default command, container port 8000, health check `GET /health`.
- `worker`: command `arq app.ingestion.worker.WorkerSettings`, no port.

The task execution role needs `secretsmanager:GetSecretValue` on the three secrets; the task role
needs EFS client mount/write.

## 5. Services and load balancer

- `api` service behind an ALB target group (port 8000, health check path `/health`), HTTPS listener.
- `worker` service with desired count 1; scale on queue depth later.
- Start one `api` task first: tables are created at start-up.

## 6. Verify

```bash
curl https://<host>/health
TOKEN=$(curl -s -X POST https://<host>/auth/token -H 'content-type: application/json' \
  -d '{"user_id":"u_admin"}' | jq -r .access_token)
curl -s -X POST https://<host>/documents -H "authorization: Bearer $TOKEN" \
  -F file=@handbook.pdf -F collection_id=policies -F allowed_roles=employee
curl -s https://<host>/admin/ingestion -H "authorization: Bearer $TOKEN"
```

To load the demo corpus, run a one-off ECS task from the `api` task definition with the command
`sh -c "python -m scripts.generate_corpus && python -m scripts.ingest_corpus"`.

## Before real users

- The login endpoint issues a token for any listed demo user without a password. Put the service
  behind your identity provider (or restrict the ALB) before exposing it.
- Add database migrations; the app currently only creates missing tables.
- Set a CloudWatch alarm on 5xx rate and on documents stuck in `failed`.
