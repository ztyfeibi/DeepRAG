# Fixed CRAG smoke experiment matrix

This matrix controls the 20-question CRAG smoke comparison. Its purpose is to validate the retrieval path, result schema, and directional behavior. It is not sufficient evidence for a final quality conclusion.

## Matrix

| ID | DeepRAG method | Retriever | Status | Runner | Result root |
|---|---|---|---|---|---|
| `crag-smoke-bm25` | `srag-sftv2` | `BM25` | Ready | `scripts/inference/run_crag_bm25_smoke.ps1` | `results/bm25_crag_smoke` |
| `crag-smoke-hypergraph` | `srag-sftv2` | `HyperGraph` | Wait for formal index acceptance | `scripts/inference/run_crag_hypergraph_smoke.ps1` | `results/hypergraph_crag_smoke` |
| `crag-smoke-fusion` | `srag-sftv2` | `Fusion` | Reserved; not implemented | No runner yet | A future dedicated root, never either root above |

Fusion is intentionally only a reserved comparison row. It must not be represented as an executable experiment until its retriever, event source labels, tests, and independent runner have been implemented and reviewed.

## Shared controls

Every executable matrix row must use exactly these controls:

| Control | Fixed value |
|---|---|
| Dataset and split | `data/crag`, `test_with_id` |
| Questions | `data/eval_subsets/crag_smoke_20.json`, exactly 20 IDs in manifest order |
| Method | `srag-sftv2` |
| DeepRAG model endpoint | `http://10.65.1.110:8005/v1` for both main and follow-up generation |
| Model path | `hf_models` |
| Temperature | `0.0` |
| Few-shot examples | `8` |
| Generation maximum length | `4096` |
| Processes | `1` |
| Retrieval-context recording | Enabled with `--record_retrieval_context` |
| Resume | Disabled; no `--resume` argument |

The prompts and decision policy are those already selected by `srag-sftv2`; the matrix does not alter SRAGSFTV2 prompts or its retrieve/no-retrieve decision.

## Retriever-specific fixed settings

| Retriever | Fixed settings |
|---|---|
| BM25 | Elasticsearch index `crag`; document `retrieve_topk=3`. |
| HyperGraph | Local HTTP URL `http://127.0.0.1:8765`; client timeout `300s`; candidate `top_k=10`; text token cap `2000`; local/global token caps `1000` each. The server contract fixes `mode=hybrid`, `only_need_context=true`, and initial concurrency `1`. |

The BM25 and HyperGraph `top_k` values are not numerically compared because they represent different retrieval semantics. Their individual values are frozen and recorded in each run's `config.json`.

## Isolation and comparison rules

1. Each row writes only to its own result root. It must not write under `results/paper_reproduction` and must not reuse a completed run directory.
2. Before comparison, each run must be evaluated with `evaluate_ans.py`; its `output.jsonl`, `details.txt`, `metrics.json`, and `config.json` are retained.
3. `summarize_crag_comparison.py` accepts only runs whose output IDs exactly equal the fixed manifest IDs on both sides.
4. Compare EM, F1, DeepRAG retrieval count, retrieval latency, total latency, success/failure/missing-output counts, and empty-context counts. A HyperGraph internal vector/entity/hyperedge operation is not an additional DeepRAG retrieval.
5. HyperGraph execution is prohibited until the formal CRAG index validation succeeds. The smoke run remains a small-scale integration and trend check, not a final benchmark.
