import importlib.util
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop


def _load_service_module():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "hypergraph", "serve_retriever.py")
    spec = importlib.util.spec_from_file_location("hypergraph_service_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


service = _load_service_module()


class _Param:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeRAG:
    def __init__(self):
        self.context = "fake context"
        self.failure = None
        self.calls = []

    async def aquery(self, query, param):
        self.calls.append((query, param))
        if self.failure:
            raise self.failure
        return self.context


class HyperGraphServiceTests(AioHTTPTestCase):
    async def get_application(self):
        self.rag = _FakeRAG()
        return service.create_app(self.rag, _Param)

    @unittest_run_loop
    async def test_health(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["backend"], "hypergraph")

    @unittest_run_loop
    async def test_retrieve_returns_context_not_answer(self):
        response = await self.client.post("/retrieve", json={"query": "question"})
        body = await response.json()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["context"], "fake context")
        self.assertNotIn("answer", body)
        self.assertTrue(self.rag.calls[0][1].only_need_context)

    @unittest_run_loop
    async def test_empty_query_is_bad_request(self):
        response = await self.client.post("/retrieve", json={"query": " "})
        self.assertEqual(response.status, 400)

    @unittest_run_loop
    async def test_fake_exception_is_server_error(self):
        self.rag.failure = RuntimeError("fake failure")
        response = await self.client.post("/retrieve", json={"query": "question"})
        self.assertEqual(response.status, 500)

    @unittest_run_loop
    async def test_empty_context_is_unavailable(self):
        self.rag.context = ""
        response = await self.client.post("/retrieve", json={"query": "question"})
        self.assertEqual(response.status, 503)

    @unittest_run_loop
    async def test_invalid_numeric_options_are_bad_requests(self):
        for field, value in (("top_k", 0), ("max_token_for_text_unit", -1),
                             ("max_token_for_local_context", True),
                             ("max_token_for_global_context", "1000")):
            response = await self.client.post("/retrieve", json={"query": "question", field: value})
            self.assertEqual(response.status, 400)
        self.assertEqual(self.rag.calls, [])


class HyperGraphRealConstructionTests(unittest.TestCase):
    def _args(self, project_dir, working_dir):
        return SimpleNamespace(
            hypergraph_project_dir=project_dir,
            working_dir=working_dir,
            llm_base_url="http://example.invalid/llm",
            llm_model="qwen-27b-3.8",
            llm_api_key="EMPTY",
            embedding_base_url="http://example.invalid/embedding",
            embedding_model="qwen3-embedding-4b",
            embedding_dim=1536,
        )

    def test_builds_hypergraphrag_with_explicit_configuration(self):
        captured = {}

        class FakeHyperGraphRAG:
            def __init__(self, working_dir, llm_model_name, llm_model_kwargs,
                         enable_llm_cache):
                captured.update({
                    "working_dir": working_dir,
                    "llm_model_name": llm_model_name,
                    "llm_model_kwargs": llm_model_kwargs,
                    "enable_llm_cache": enable_llm_cache,
                    "embedding_base_url": os.environ["HYPERGRAPHRAG_EMBEDDING_BASE_URL"],
                    "embedding_model": os.environ["HYPERGRAPHRAG_EMBEDDING_MODEL"],
                    "embedding_dim": os.environ["HYPERGRAPHRAG_EMBEDDING_DIM"],
                })

        fake_module = types.ModuleType("hypergraphrag")
        fake_module.HyperGraphRAG = FakeHyperGraphRAG
        fake_module.QueryParam = _Param
        original = sys.modules.get("hypergraphrag")
        embedding_keys = (
            "HYPERGRAPHRAG_EMBEDDING_BASE_URL",
            "HYPERGRAPHRAG_EMBEDDING_MODEL",
            "HYPERGRAPHRAG_EMBEDDING_DIM",
        )
        original_embedding = {key: os.environ.get(key) for key in embedding_keys}
        with tempfile.TemporaryDirectory() as directory:
            project_dir = os.path.join(directory, "project")
            working_dir = os.path.join(directory, "index")
            os.makedirs(project_dir)
            os.makedirs(working_dir)
            sys.modules["hypergraphrag"] = fake_module
            try:
                rag, query_param = service.build_real_rag(self._args(project_dir, working_dir))
            finally:
                if original is None:
                    del sys.modules["hypergraphrag"]
                else:
                    sys.modules["hypergraphrag"] = original
                for key, value in original_embedding.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                if project_dir in sys.path:
                    sys.path.remove(project_dir)
        self.assertIsInstance(rag, FakeHyperGraphRAG)
        self.assertIs(query_param, _Param)
        self.assertEqual(captured["working_dir"], os.path.abspath(working_dir))
        self.assertEqual(captured["llm_model_name"], "qwen-27b-3.8")
        self.assertEqual(captured["llm_model_kwargs"], {
            "base_url": "http://example.invalid/llm", "api_key": "EMPTY"})
        self.assertEqual(captured["embedding_base_url"], "http://example.invalid/embedding")
        self.assertEqual(captured["embedding_model"], "qwen3-embedding-4b")
        self.assertEqual(captured["embedding_dim"], "1536")
        self.assertFalse(captured["enable_llm_cache"])
        self.assertEqual(os.environ.get("HYPERGRAPHRAG_EMBEDDING_BASE_URL"), original_embedding["HYPERGRAPHRAG_EMBEDDING_BASE_URL"])
        self.assertEqual(os.environ.get("HYPERGRAPHRAG_EMBEDDING_MODEL"), original_embedding["HYPERGRAPHRAG_EMBEDDING_MODEL"])
        self.assertEqual(os.environ.get("HYPERGRAPHRAG_EMBEDDING_DIM"), original_embedding["HYPERGRAPHRAG_EMBEDDING_DIM"])

    def test_relative_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            project_dir = os.path.join(directory, "project")
            working_dir = os.path.join(directory, "index")
            os.makedirs(project_dir)
            os.makedirs(working_dir)
            old_cwd = os.getcwd()
            os.chdir(directory)
            try:
                self.assertEqual(service._existing_absolute_dir("project", "project"), project_dir)
                self.assertEqual(service._existing_absolute_dir("index", "index"), working_dir)
            finally:
                os.chdir(old_cwd)

    def test_missing_directory_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            service._existing_absolute_dir("does-not-exist", "working directory")


if __name__ == "__main__":
    unittest.main()
