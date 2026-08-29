import ast
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_event_harness():
    source_path = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "basic.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    basic_rag = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "BasicRAG")
    names = {"reset_retrieval_events", "_context_blocks_for_event", "_new_retrieval_event", "retrieve"}
    methods = [node for node in basic_rag.body if isinstance(node, ast.FunctionDef) and node.name in names]
    harness = ast.Module(
        body=[ast.ClassDef(name="EventHarness", bases=[], keywords=[], decorator_list=[], body=methods)],
        type_ignores=[],
    )
    namespace = {"time": time}
    exec(compile(ast.fix_missing_locations(harness), str(source_path), "exec"), namespace)
    return namespace["EventHarness"]


class FakeBM25:
    def retrieve(self, queries, topk, max_query_length):
        return [], [["first document", "second document"]]


class FakeHyperGraph:
    def retrieve(self, queries):
        return [], [["combined hypergraph context"]]


class FailingRetriever:
    def retrieve(self, **kwargs):
        raise RuntimeError("secret endpoint detail")


class RetrievalEventTests(unittest.TestCase):
    def make_rag(self, retriever_type, retriever, record_retrieval_context=True):
        rag = load_event_harness()()
        rag.retriever_type = retriever_type
        rag.retriever = retriever
        rag.record_retrieval_context = record_retrieval_context
        rag.counter = SimpleNamespace(retrieve=0, retrieval_latency_sec=0.0)
        rag.reset_retrieval_events()
        return rag

    def test_bm25_events_are_per_query_and_include_multiple_blocks(self):
        rag = self.make_rag("BM25", FakeBM25())

        result = rag.retrieve("first query", topk=2)
        rag.retrieve("second query", topk=2)

        self.assertEqual(result, ["first document", "second document"])
        self.assertEqual(rag.counter.retrieve, 2)
        self.assertEqual(len(rag.retrieval_events), 2)
        self.assertEqual([event["step"] for event in rag.retrieval_events], [1, 2])
        self.assertEqual(rag.retrieval_events[0]["context_block_count"], 2)
        self.assertEqual(rag.retrieval_events[0]["context_blocks"][1]["rank"], 2)
        self.assertEqual(rag.retrieval_events[0]["context_blocks"][0]["source"], "bm25")
        self.assertGreaterEqual(rag.counter.retrieval_latency_sec, 0.0)

    def test_context_text_is_omitted_when_recording_is_disabled(self):
        rag = self.make_rag("BM25", FakeBM25(), record_retrieval_context=False)

        rag.retrieve("query", topk=2)

        event = rag.retrieval_events[0]
        self.assertFalse(event["context_recorded"])
        self.assertNotIn("context_blocks", event)
        self.assertEqual(event["context_block_count"], 2)
        self.assertEqual(event["context_char_count"], len("first documentsecond document"))

    def test_hypergraph_is_one_combined_context_block(self):
        rag = self.make_rag("HyperGraph", FakeHyperGraph())

        rag.retrieve("query")

        event = rag.retrieval_events[0]
        self.assertEqual(event["backend"], "HyperGraph")
        self.assertEqual(event["context_block_count"], 1)
        self.assertEqual(event["context_blocks"][0]["source"], "hypergraph")
        self.assertEqual(event["context_blocks"][0]["rank"], 1)

    def test_reset_produces_a_zero_retrieval_question(self):
        rag = self.make_rag("BM25", FakeBM25())
        rag.retrieve("query")

        rag.reset_retrieval_events()

        self.assertEqual(rag.retrieval_events, [])

    def test_failure_event_is_recorded_before_original_error_is_reraised(self):
        rag = self.make_rag("HyperGraph", FailingRetriever())

        with self.assertRaisesRegex(RuntimeError, "secret endpoint detail"):
            rag.retrieve("query")

        self.assertEqual(rag.counter.retrieve, 1)
        self.assertEqual(len(rag.retrieval_events), 1)
        event = rag.retrieval_events[0]
        self.assertEqual(event["status"], "failure")
        self.assertEqual(event["context_block_count"], 0)
        self.assertEqual(event["error"]["type"], "RuntimeError")
        self.assertEqual(event["error"]["message"], "retrieval failed")
        self.assertNotIn("secret", event["error"]["message"])


if __name__ == "__main__":
    unittest.main()
