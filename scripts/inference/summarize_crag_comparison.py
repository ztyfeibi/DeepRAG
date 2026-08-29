"""Create an offline, ID-aligned BM25 versus HyperGraph CRAG comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "EM",
    "F1",
    "avg_retrieve_count",
    "avg_retrieval_latency_sec",
    "p50_retrieval_latency_sec",
    "p95_retrieval_latency_sec",
    "avg_latency_sec",
    "p50_latency_sec",
    "p95_latency_sec",
    "success_count",
    "failure_count",
    "missing_output_count",
    "empty_context_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_ids_file", required=True, type=Path)
    parser.add_argument("--bm25_dir", required=True, type=Path)
    parser.add_argument("--hypergraph_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Manifest has no non-empty items list: {path}")
    if manifest.get("count") != len(items):
        raise ValueError(f"Manifest count does not match items length: {path}")

    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {position} is not an object: {path}")
        qid = item.get("id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"Manifest item {position} has an invalid ID: {path}")
        if qid in seen_ids:
            raise ValueError(f"Manifest contains duplicate ID {qid}: {path}")
        seen_ids.add(qid)
        normalized.append(item)
    return normalized


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Required output file is missing: {path}")
    records: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                qid = record.get("qid")
                if not isinstance(qid, str) or not qid:
                    raise ValueError(f"{path}:{line_number} has an invalid qid")
                if qid in records:
                    raise ValueError(f"{path} contains duplicate qid {qid}")
                records[qid] = record
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSONL file {path}: {exc}") from exc
    return records


def assert_exact_ids(label: str, records: dict[str, Any], expected_ids: set[str]) -> None:
    actual_ids = set(records)
    if actual_ids == expected_ids:
        return
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    raise ValueError(
        f"{label} IDs differ from the fixed manifest; "
        f"missing={missing}, unexpected={unexpected}"
    )


def load_run(run_dir: Path, expected_ids: set[str], label: str) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise ValueError(f"{label} run directory does not exist: {run_dir}")
    outputs = load_jsonl_by_id(run_dir / "output.jsonl")
    details = load_jsonl_by_id(run_dir / "details.txt")
    assert_exact_ids(f"{label} output.jsonl", outputs, expected_ids)
    assert_exact_ids(f"{label} details.txt", details, expected_ids)

    metrics = read_json(run_dir / "metrics.json")
    missing_metrics = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing_metrics:
        raise ValueError(f"{label} metrics.json is missing fields: {missing_metrics}")
    return {"directory": str(run_dir.resolve()), "outputs": outputs, "details": details, "metrics": metrics}


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def difference(left: Any, right: Any) -> float | None:
    left_number, right_number = finite_number(left), finite_number(right)
    if left_number is None or right_number is None:
        return None
    return right_number - left_number


def classify_f1_delta(delta: float | None) -> str:
    if delta is None or abs(delta) < 1e-12:
        return "tie"
    return "improved" if delta > 0 else "regressed"


def mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def percentile(values: list[float | None], percent: float) -> float | None:
    valid = sorted(value for value in values if value is not None)
    if not valid:
        return None
    position = (len(valid) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return valid[lower]
    return valid[lower] + (valid[upper] - valid[lower]) * (position - lower)


def per_question_rows(manifest: list[dict[str, Any]], bm25: dict[str, Any], hypergraph: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for display_order, item in enumerate(manifest, start=1):
        qid = item["id"]
        bm25_output, hypergraph_output = bm25["outputs"][qid], hypergraph["outputs"][qid]
        bm25_detail, hypergraph_detail = bm25["details"][qid], hypergraph["details"][qid]
        f1_delta = difference(bm25_detail.get("F1"), hypergraph_detail.get("F1"))
        row = {
            "display_order": display_order,
            "id": qid,
            "original_position": item.get("original_position"),
            "question": item.get("question"),
            "comparison": classify_f1_delta(f1_delta),
            "bm25_em": finite_number(bm25_detail.get("EM")),
            "hypergraph_em": finite_number(hypergraph_detail.get("EM")),
            "em_delta_hypergraph_minus_bm25": difference(bm25_detail.get("EM"), hypergraph_detail.get("EM")),
            "bm25_f1": finite_number(bm25_detail.get("F1")),
            "hypergraph_f1": finite_number(hypergraph_detail.get("F1")),
            "f1_delta_hypergraph_minus_bm25": f1_delta,
            "bm25_retrieve_count": finite_number(bm25_output.get("retrieve_count")),
            "hypergraph_retrieve_count": finite_number(hypergraph_output.get("retrieve_count")),
            "retrieve_count_delta_hypergraph_minus_bm25": difference(bm25_output.get("retrieve_count"), hypergraph_output.get("retrieve_count")),
            "bm25_retrieval_latency_sec": finite_number(bm25_output.get("retrieval_latency_sec")),
            "hypergraph_retrieval_latency_sec": finite_number(hypergraph_output.get("retrieval_latency_sec")),
            "retrieval_latency_delta_sec_hypergraph_minus_bm25": difference(bm25_output.get("retrieval_latency_sec"), hypergraph_output.get("retrieval_latency_sec")),
            "bm25_latency_sec": finite_number(bm25_output.get("latency_sec")),
            "hypergraph_latency_sec": finite_number(hypergraph_output.get("latency_sec")),
            "latency_delta_sec_hypergraph_minus_bm25": difference(bm25_output.get("latency_sec"), hypergraph_output.get("latency_sec")),
            "bm25_final_prediction": bm25_detail.get("final_pred"),
            "hypergraph_final_prediction": hypergraph_detail.get("final_pred"),
        }
        rows.append(row)
    return rows


def metric_rows(bm25_metrics: dict[str, Any], hypergraph_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "metric": key,
            "bm25": finite_number(bm25_metrics.get(key)),
            "hypergraph": finite_number(hypergraph_metrics.get(key)),
            "delta_hypergraph_minus_bm25": difference(bm25_metrics.get(key), hypergraph_metrics.get(key)),
        }
        for key in REQUIRED_METRICS
    ]


def calculated_metrics(rows: list[dict[str, Any]], backend: str, recorded_metrics: dict[str, Any]) -> dict[str, Any]:
    prefix = f"{backend}_"
    ems = [row[f"{prefix}em"] for row in rows]
    f1s = [row[f"{prefix}f1"] for row in rows]
    retrieve_counts = [row[f"{prefix}retrieve_count"] for row in rows]
    retrieval_latencies = [row[f"{prefix}retrieval_latency_sec"] for row in rows]
    total_latencies = [row[f"{prefix}latency_sec"] for row in rows]
    calculated = {
        "EM": mean(ems),
        "F1": mean(f1s),
        "avg_retrieve_count": mean(retrieve_counts),
        "avg_retrieval_latency_sec": mean(retrieval_latencies),
        "p50_retrieval_latency_sec": percentile(retrieval_latencies, 50),
        "p95_retrieval_latency_sec": percentile(retrieval_latencies, 95),
        "avg_latency_sec": mean(total_latencies),
        "p50_latency_sec": percentile(total_latencies, 50),
        "p95_latency_sec": percentile(total_latencies, 95),
    }
    for key in ("success_count", "failure_count", "missing_output_count", "empty_context_count"):
        calculated[key] = finite_number(recorded_metrics.get(key))
    return calculated


def markdown_escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def format_number(value: Any) -> str:
    number = finite_number(value)
    return "" if number is None else f"{number:.6f}"


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# CRAG Smoke Comparison",
        "",
        f"- Fixed manifest: `{comparison['manifest']['path']}`",
        f"- Questions: {comparison['manifest']['count']}",
        f"- BM25 run: `{comparison['runs']['bm25']['directory']}`",
        f"- HyperGraph run: `{comparison['runs']['hypergraph']['directory']}`",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | BM25 | HyperGraph | HyperGraph − BM25 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["aggregate_metrics"]:
        lines.append(
            f"| {row['metric']} | {format_number(row['bm25'])} | "
            f"{format_number(row['hypergraph'])} | {format_number(row['delta_hypergraph_minus_bm25'])} |"
        )
    lines.extend(["", "## Per-question changes", "", "| # | ID | Change | ΔF1 | Δ retrieval sec | Δ total sec | Question |", "|---:|---|---|---:|---:|---:|---|"])
    for row in comparison["per_question"]:
        lines.append(
            f"| {row['display_order']} | {row['id']} | {row['comparison']} | "
            f"{format_number(row['f1_delta_hypergraph_minus_bm25'])} | "
            f"{format_number(row['retrieval_latency_delta_sec_hypergraph_minus_bm25'])} | "
            f"{format_number(row['latency_delta_sec_hypergraph_minus_bm25'])} | "
            f"{markdown_escape(row['question'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_question_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.sample_ids_file)
    expected_ids = {item["id"] for item in manifest}
    bm25 = load_run(args.bm25_dir, expected_ids, "BM25")
    hypergraph = load_run(args.hypergraph_dir, expected_ids, "HyperGraph")
    rows = per_question_rows(manifest, bm25, hypergraph)
    bm25_calculated_metrics = calculated_metrics(rows, "bm25", bm25["metrics"])
    hypergraph_calculated_metrics = calculated_metrics(rows, "hypergraph", hypergraph["metrics"])
    comparison = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(args.sample_ids_file.resolve()), "count": len(manifest)},
        "runs": {
            "bm25": {"directory": bm25["directory"], "metrics": bm25["metrics"]},
            "hypergraph": {"directory": hypergraph["directory"], "metrics": hypergraph["metrics"]},
        },
        "aggregate_metrics": metric_rows(bm25_calculated_metrics, hypergraph_calculated_metrics),
        "per_question": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "comparison.json"
    markdown_path = args.output_dir / "comparison.md"
    csv_path = args.output_dir / "per_question.csv"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(markdown_path, comparison)
    write_per_question_csv(csv_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
