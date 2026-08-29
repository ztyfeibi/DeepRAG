"""Run a context-only HyperGraphRAG HTTP service with the gepa interpreter."""
import argparse
import asyncio
import logging
import os
import sys
import time

from aiohttp import web


logger = logging.getLogger(__name__)


def _error(status, message):
    return web.json_response({"status": "error", "error": message}, status=status)


def _positive_int(payload, name, default):
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _query_options(payload):
    return {
        "mode": "hybrid",
        "only_need_context": True,
        "top_k": _positive_int(payload, "top_k", 10),
        "max_token_for_text_unit": _positive_int(payload, "max_token_for_text_unit", 2000),
        "max_token_for_local_context": _positive_int(payload, "max_token_for_local_context", 1000),
        "max_token_for_global_context": _positive_int(payload, "max_token_for_global_context", 1000),
    }


def create_app(rag, query_param_factory, max_concurrency=1, embedding_dimension=1536):
    """Create the service with an injected RAG, enabling fully offline tests."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    app = web.Application()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def health(_request):
        return web.json_response({
            "status": "ok", "ready": True, "backend": "hypergraph",
            "mode": "hybrid", "embedding_dimension": embedding_dimension,
        })

    async def retrieve(request):
        try:
            payload = await request.json()
        except Exception:
            return _error(400, "request body must be JSON")
        if not isinstance(payload, dict) or not isinstance(payload.get("query"), str) or not payload["query"].strip():
            return _error(400, "query must be a non-empty string")
        try:
            params = query_param_factory(**_query_options(payload))
        except (TypeError, ValueError) as exc:
            return _error(400, "invalid retrieval parameters: {}".format(exc))
        started = time.perf_counter()
        try:
            async with semaphore:
                context = await rag.aquery(payload["query"], param=params)
        except Exception as exc:
            logger.error("HyperGraph retrieval failed: %s", type(exc).__name__)
            return _error(500, "HyperGraph retrieval failed")
        if not isinstance(context, str) or not context.strip():
            return _error(503, "HyperGraph returned empty context")
        return web.json_response({
            "status": "ok", "backend": "hypergraph", "mode": "hybrid",
            "context": context, "latency_sec": time.perf_counter() - started,
            "top_k": getattr(params, "top_k", payload.get("top_k", 10)),
        })

    app.router.add_get("/health", health)
    app.router.add_post("/retrieve", retrieve)
    return app


def _existing_absolute_dir(value, label):
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.isdir(path):
        raise FileNotFoundError("{} does not exist or is not a directory".format(label))
    return path


def build_real_rag(args):
    """Import and configure the project's HyperGraphRAG implementation once."""
    project_dir = _existing_absolute_dir(args.hypergraph_project_dir, "HyperGraphRAG project directory")
    working_dir = _existing_absolute_dir(args.working_dir, "HyperGraphRAG working directory")
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    try:
        from hypergraphrag import HyperGraphRAG, QueryParam
    except ImportError as exc:
        raise RuntimeError("HyperGraphRAG imports failed; use the gepa interpreter") from exc
    os.environ["HYPERGRAPHRAG_EMBEDDING_BASE_URL"] = args.embedding_base_url
    os.environ["HYPERGRAPHRAG_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["HYPERGRAPHRAG_EMBEDDING_DIM"] = str(args.embedding_dim)
    rag = HyperGraphRAG(
        working_dir=working_dir,
        llm_model_name=args.llm_model,
        llm_model_kwargs={
            "base_url": args.llm_base_url,
            "api_key": args.llm_api_key,
        },
        enable_llm_cache=False,
    )
    return rag, QueryParam


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypergraph_project_dir", default=r"D:\projectes\finalRAG\HyperGraphRAG")
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--llm_base_url", default="http://10.65.1.110:8003/v1")
    parser.add_argument("--llm_model", default="qwen-27b-3.8")
    parser.add_argument("--llm_api_key", default="EMPTY")
    parser.add_argument("--embedding_base_url", default="http://10.65.1.110:8006/v1")
    parser.add_argument("--embedding_model", default="qwen3-embedding-4b")
    parser.add_argument("--embedding_dim", type=int, default=1536)
    parser.add_argument("--max_concurrency", type=int, default=1)
    return parser.parse_args()


async def main_async(args):
    try:
        rag, query_param_factory = build_real_rag(args)
    except Exception as exc:
        logger.error("HyperGraph service initialization failed: %s", type(exc).__name__)
        raise RuntimeError("HyperGraph service initialization failed") from exc
    if hasattr(rag, "initialize_storages"):
        await rag.initialize_storages()
    runner = web.AppRunner(create_app(
        rag, query_param_factory, args.max_concurrency, args.embedding_dim))
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    print("HyperGraph context service listening on http://{}:{}".format(args.host, args.port))
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
