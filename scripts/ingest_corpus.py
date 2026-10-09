"""Ingest the generated corpus (data/corpus/manifest.json) straight through the ingestion pipeline.

    python -m scripts.ingest_corpus                # everything in the manifest
    python -m scripts.ingest_corpus --kind core    # only the documents the evaluation asks about

Idempotent: unchanged files are skipped. In the default local setup (embedded Qdrant) stop the
API first, because embedded Qdrant allows a single process at a time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sqlalchemy import func, select

from app.auth.users import USERS
from app.config import get_settings
from app.db import Document, init_db, session_scope
from app.ingestion.pipeline import ingest_document, register_upload
from app.observability import configure_logging
from app.services import get_services


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/corpus/manifest.json")
    parser.add_argument("--kind", choices=["core", "filler"], help="ingest only this kind of document")
    args = parser.parse_args()

    configure_logging("WARNING")
    init_db()
    services = get_services()
    settings = get_settings()
    entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.kind:
        entries = [e for e in entries if e["kind"] == args.kind]

    started = time.perf_counter()
    ingested = skipped = failed = 0
    for number, entry in enumerate(entries, start=1):
        path = Path(entry["path"])
        with session_scope() as db:
            document, needs_ingestion = register_upload(
                db,
                filename=path.name,
                data=path.read_bytes(),
                collection_id=entry["collection_id"],
                allowed_roles=entry["allowed_roles"],
                user=USERS["u_admin"],
            )
            document_id = document.id
        if not needs_ingestion:
            skipped += 1
            continue
        try:
            ingest_document(document_id, services)
            ingested += 1
        except Exception as exc:
            failed += 1
            print(f"  FAILED {path.name}: {type(exc).__name__}: {exc}")
        if number % 20 == 0:
            print(f"  {number}/{len(entries)} documents processed")

    with session_scope() as db:
        documents, pages, chunks = db.execute(
            select(func.count(), func.sum(Document.page_count), func.sum(Document.chunk_count)).where(
                Document.status == "ready"
            )
        ).one()
    elapsed = time.perf_counter() - started
    print(f"ingested {ingested}, skipped {skipped} unchanged, failed {failed} in {elapsed:.1f}s")
    print(f"index now holds {documents} documents, {pages or 0} pages, {chunks or 0} chunks")
    print(f"embedder: {services.embedder.name}  chunk_size={settings.chunk_size} overlap={settings.chunk_overlap}")


if __name__ == "__main__":
    main()
