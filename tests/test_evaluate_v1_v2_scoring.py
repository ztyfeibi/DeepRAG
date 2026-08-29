import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "src" / "evaluate_ans.py"


class EvaluateV1V2ScoringTests(unittest.TestCase):
    def write_run(self, root, name, output_record):
        run_dir = root / name
        data_dir = root / "data"
        run_dir.mkdir()
        data_dir.mkdir(exist_ok=True)
        (data_dir / "test_with_id.jsonl").write_text(
            json.dumps({"_id": "q1", "question": "What is the answer?", "answer": "alpha"}) + "\n",
            encoding="utf-8",
        )
        config = {
            "dataset": "hotpotqa",
            "data_path": str(data_dir),
            "split": "test_with_id",
            "fewshot": 0,
            "method": "srag-sftv2",
            "model_name_or_path": "unused-by-srag-sftv2-evaluation",
            "remote_url": None,
        }
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run_dir / "output.jsonl").write_text(json.dumps(output_record) + "\n", encoding="utf-8")
        return run_dir

    def evaluate(self, run_dir):
        completed = subprocess.run(
            [sys.executable, str(EVALUATOR), "--dir", str(run_dir)],
            cwd=PROJECT_ROOT / "src",
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        details = json.loads((run_dir / "details.txt").read_text(encoding="utf-8").strip())
        return metrics, details

    def test_v1_and_v2_preserve_answer_metrics_and_final_prediction(self):
        prediction = "<answer short>alpha</answer short>"
        v1_record = {
            "qid": "q1",
            "question": "What is the answer?",
            "prediction": prediction,
            "answer": "alpha",
            "retrieve_count": 1,
            "latency_sec": 1.0,
        }
        v2_record = {
            **v1_record,
            "schema_version": 2,
            "status": "success",
            "retriever": "BM25",
            "retrieval_events": [{
                "step": 1,
                "query": "What is the answer?",
                "backend": "BM25",
                "status": "success",
                "latency_sec": 0.1,
                "context_recorded": False,
                "context_block_count": 1,
                "context_char_count": 5,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1_metrics, v1_details = self.evaluate(self.write_run(root, "v1", v1_record))
            v2_metrics, v2_details = self.evaluate(self.write_run(root, "v2", v2_record))

        self.assertEqual(v1_metrics["EM"], v2_metrics["EM"])
        self.assertEqual(v1_metrics["F1"], v2_metrics["F1"])
        self.assertEqual(v1_details["final_pred"], v2_details["final_pred"])
        self.assertEqual(v1_details["EM"], v2_details["EM"])
        self.assertEqual(v1_details["F1"], v2_details["F1"])


if __name__ == "__main__":
    unittest.main()
