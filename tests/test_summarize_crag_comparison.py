import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "inference" / "summarize_crag_comparison.py"
SPEC = importlib.util.spec_from_file_location("summarize_crag_comparison", SCRIPT_PATH)
comparison_tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison_tool)


class SummarizeCragComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "crag_smoke_20.json"
        self.items = [
            {"id": "q1", "original_position": 0, "question": "Question one"},
            {"id": "q2", "original_position": 1, "question": "Question two"},
            {"id": "q3", "original_position": 2, "question": "Question three"},
        ]
        self.manifest_path.write_text(json.dumps({"count": 3, "items": self.items}), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_run(self, name, output_rows, detail_rows, *, failure_count=0):
        run_dir = self.root / name
        run_dir.mkdir()
        (run_dir / "output.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8"
        )
        (run_dir / "details.txt").write_text(
            "".join(json.dumps(row) + "\n" for row in detail_rows), encoding="utf-8"
        )
        metrics = {key: 0.0 for key in comparison_tool.REQUIRED_METRICS}
        metrics.update({"success_count": 3 - failure_count, "failure_count": failure_count})
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return run_dir

    @staticmethod
    def output_row(qid, prediction, retrieve_count, retrieval_latency, latency):
        return {
            "qid": qid,
            "prediction": prediction,
            "retrieve_count": retrieve_count,
            "retrieval_latency_sec": retrieval_latency,
            "latency_sec": latency,
        }

    @staticmethod
    def detail_row(qid, em, f1):
        return {"qid": qid, "final_pred": qid, "EM": em, "F1": f1}

    def run_tool(self, bm25_dir, hypergraph_dir):
        output_dir = self.root / "comparison"
        argv = [
            str(SCRIPT_PATH),
            "--sample_ids_file", str(self.manifest_path),
            "--bm25_dir", str(bm25_dir),
            "--hypergraph_dir", str(hypergraph_dir),
            "--output_dir", str(output_dir),
        ]
        with patch.object(sys, "argv", argv):
            self.assertEqual(comparison_tool.main(), 0)
        return output_dir

    def test_aligned_results_calculate_em_f1_p95_and_preserve_zero_retrieval(self):
        bm25_outputs = [
            self.output_row("q1", "answer", 1, 0.1, 1.0),
            self.output_row("q2", "answer", 0, 0.2, 2.0),
            self.output_row("q3", None, 2, None, None),
        ]
        hypergraph_outputs = [
            self.output_row("q1", "answer", 1, 0.3, 3.0),
            self.output_row("q2", "answer", 0, 0.4, 4.0),
            self.output_row("q3", "answer", 1, 0.5, 5.0),
        ]
        bm25_details = [self.detail_row("q1", 1, 0.2), self.detail_row("q2", 0, 0.4), self.detail_row("q3", 0, 0.0)]
        hypergraph_details = [self.detail_row("q1", 1, 0.6), self.detail_row("q2", 1, 0.3), self.detail_row("q3", 1, 0.9)]
        bm25_dir = self.write_run("bm25", bm25_outputs, bm25_details, failure_count=1)
        hypergraph_dir = self.write_run("hypergraph", hypergraph_outputs, hypergraph_details)

        output_dir = self.run_tool(bm25_dir, hypergraph_dir)
        comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
        metrics = {row["metric"]: row for row in comparison["aggregate_metrics"]}

        self.assertAlmostEqual(metrics["EM"]["bm25"], 1 / 3)
        self.assertAlmostEqual(metrics["F1"]["hypergraph"], 0.6)
        self.assertAlmostEqual(metrics["p95_latency_sec"]["bm25"], 1.95)
        self.assertAlmostEqual(metrics["p95_retrieval_latency_sec"]["bm25"], 0.195)
        self.assertEqual(metrics["avg_retrieve_count"]["bm25"], 1.0)
        self.assertEqual(metrics["failure_count"]["bm25"], 1.0)
        self.assertIsNone(comparison["per_question"][2]["bm25_latency_sec"])
        self.assertEqual(comparison["per_question"][1]["bm25_retrieve_count"], 0.0)
        self.assertTrue((output_dir / "comparison.md").is_file())
        self.assertTrue((output_dir / "per_question.csv").is_file())

    def test_missing_question_is_rejected(self):
        outputs = [self.output_row("q1", "answer", 0, 0.1, 1), self.output_row("q2", "answer", 0, 0.1, 1)]
        details = [self.detail_row("q1", 0, 0), self.detail_row("q2", 0, 0), self.detail_row("q3", 0, 0)]
        bm25_dir = self.write_run("bm25", outputs, details)
        hypergraph_dir = self.write_run("hypergraph", outputs, details)

        with self.assertRaisesRegex(ValueError, "IDs differ"):
            self.run_tool(bm25_dir, hypergraph_dir)

    def test_duplicate_question_is_rejected(self):
        outputs = [
            self.output_row("q1", "answer", 0, 0.1, 1),
            self.output_row("q2", "answer", 0, 0.1, 1),
            self.output_row("q3", "answer", 0, 0.1, 1),
            self.output_row("q3", "answer", 0, 0.1, 1),
        ]
        details = [self.detail_row("q1", 0, 0), self.detail_row("q2", 0, 0), self.detail_row("q3", 0, 0)]
        bm25_dir = self.write_run("bm25", outputs, details)
        hypergraph_dir = self.write_run("hypergraph", outputs[:3], details)

        with self.assertRaisesRegex(ValueError, "duplicate qid q3"):
            self.run_tool(bm25_dir, hypergraph_dir)


if __name__ == "__main__":
    unittest.main()
