import ast
import unittest
from pathlib import Path


def load_evaluation_helpers():
    source_path = Path(__file__).resolve().parents[1] / "src" / "evaluate_ans.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_as_int",
        "_has_empty_retrieved_context",
        "_retrieval_events",
        "_event_has_empty_context",
    }
    helpers = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=helpers, type_ignores=[])), str(source_path), "exec"), namespace)
    return namespace


class EvaluateEventCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_evaluation_helpers()

    def test_legacy_record_without_events_keeps_empty_context_fallback(self):
        legacy_record = {"prediction": [{"docs": []}]}

        self.assertEqual(self.helpers["_retrieval_events"](legacy_record), [])
        self.assertTrue(self.helpers["_has_empty_retrieved_context"](legacy_record["prediction"]))

    def test_v2_events_are_read_without_inspecting_prediction(self):
        v2_record = {
            "schema_version": 2,
            "prediction": [{"docs": ["legacy-shaped trace text"]}],
            "retrieval_events": [
                {"status": "success", "context_block_count": 1, "latency_sec": 0.2},
                {"status": "failure", "context_block_count": 0, "latency_sec": 0.3},
            ],
        }

        events = self.helpers["_retrieval_events"](v2_record)
        self.assertEqual(len(events), 2)
        self.assertFalse(self.helpers["_event_has_empty_context"](events[0]))
        self.assertFalse(self.helpers["_event_has_empty_context"](events[1]))

    def test_v2_success_event_with_zero_blocks_is_empty_context(self):
        event = {"status": "success", "context_block_count": 0, "latency_sec": 0.1}

        self.assertTrue(self.helpers["_event_has_empty_context"](event))


if __name__ == "__main__":
    unittest.main()
