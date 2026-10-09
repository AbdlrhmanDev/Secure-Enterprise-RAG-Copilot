"""Run evaluation experiments and compare retrieval configurations.

    python -m scripts.run_eval --all                              # every config in evals/experiments/
    python -m scripts.run_eval evals/experiments/hybrid_rrf_rerank.json
    python -m scripts.run_eval --all --no-generate                # retrieval metrics only (faster)
    python -m scripts.run_eval --all --judge llm                  # LLM-judged correctness/faithfulness

Each run is stored in the eval_runs table (visible in the UI dashboard) and written to
evals/experiments/results/<name>.json; a comparison table goes to results/comparison.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.db import EvalRun, init_db, session_scope
from app.evaluation.runner import EvalConfig, create_run, execute_run
from app.observability import configure_logging
from app.services import get_services

EXPERIMENTS_DIR = Path("evals/experiments")
RESULTS_DIR = EXPERIMENTS_DIR / "results"

COLUMNS = [
    ("recall_at_k", "Recall@K"),
    ("precision_at_k", "Precision@K"),
    ("mrr", "MRR"),
    ("ndcg_at_k", "nDCG@K"),
    ("acl_leaked_chunks", "ACL leaks"),
    ("answer_correctness", "Correct"),
    ("groundedness", "Grounded"),
    ("citation_correctness", "Citation"),
    ("refusal_accuracy", "Refusal"),
    ("retrieval_p95_ms", "Retr p95 ms"),
    ("rerank_p95_ms", "Rerank p95 ms"),
    ("latency_p50_ms", "p50 ms"),
    ("latency_p95_ms", "p95 ms"),
    ("cost_per_query_usd", "USD/query"),
]


def format_table(rows: list[tuple[str, dict]]) -> str:
    header = "| Experiment | " + " | ".join(label for _, label in COLUMNS) + " |"
    divider = "|---|" + "---:|" * len(COLUMNS)
    lines = [header, divider]
    for name, summary in rows:
        cells = ["-" if summary.get(key) is None else str(summary[key]) for key, _ in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", type=Path, help="experiment config files")
    parser.add_argument("--all", action="store_true", help="run every config in evals/experiments/")
    parser.add_argument("--no-generate", action="store_true", help="skip answer generation")
    parser.add_argument("--judge", choices=["heuristic", "llm"])
    parser.add_argument("--k", type=int)
    parser.add_argument("--limit", type=int, help="evaluate only the first N cases")
    args = parser.parse_args()

    paths = sorted(EXPERIMENTS_DIR.glob("*.json")) if args.all else args.configs
    if not paths:
        parser.error("give at least one config file or --all")

    configure_logging("WARNING")
    init_db()
    services = get_services()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in paths:
        config = EvalConfig(**json.loads(path.read_text(encoding="utf-8")))
        if args.no_generate:
            config.generate = False
        if args.judge:
            config.judge = args.judge
        if args.k:
            config.k = args.k
        if args.limit:
            config.limit = args.limit

        print(f"running {config.name} ...", flush=True)
        with session_scope() as db:
            run_id = create_run(db, config).id
        execute_run(run_id, services)
        with session_scope() as db:
            run = db.get(EvalRun, run_id)
            if run.status != "completed":
                print(f"  FAILED: {run.error}")
                continue
            results = run.metrics_json
        output = {"run_id": run_id, "config": config.model_dump(), **results}
        (RESULTS_DIR / f"{config.name}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        rows.append((config.name, results["summary"]))

    if not rows:
        raise SystemExit(1)
    summary = rows[0][1]
    table = format_table(rows)
    notes = (
        f"\n\nK = {summary['k']}. {summary['cases']} cases ({summary['answerable_cases']} answerable, "
        f"{summary['refusal_cases']} that must be refused). Embedder: {summary['embedder']}. "
        f"Generator: {summary['generator']}. Judge: {summary['judge']}. "
        f"Latency scope: {summary['latency_scope']}.\n"
    )
    print("\n" + table + notes)
    (RESULTS_DIR / "comparison.md").write_text("# Experiment comparison\n\n" + table + notes, encoding="utf-8")


if __name__ == "__main__":
    main()
