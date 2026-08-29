# DeepRAG result contract (schema version 2)

This document defines the additive result format for DeepRAG retrieval experiments. It applies to each line of `output.jsonl` and keeps `prediction` unchanged so that the existing answer extraction and scoring path remains valid.

## Compatibility

- A result with no `schema_version` is a legacy v1 result. Existing evaluators must continue to score it from `prediction` exactly as they do today.
- A v2 result has `schema_version: 2`. It retains every existing output field and adds retrieval provenance; no consumer may infer new events by parsing `prediction`.
- `prediction` remains the unmodified model trajectory for `srag-sftv2`. It is not a final-answer field and is never rewritten by this contract.

## Top-level v2 record

The following fields are required for a successful v2 record.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | integer | Always `2`. |
| `qid` | non-empty string | Dataset question ID. |
| `question` | string | Original question text. |
| `prediction` | JSON value | Existing full inference trajectory, retained unchanged. |
| `answer` | JSON value | Existing reference-answer field. |
| `status` | string | `success` for a completed inference, `failure` for a serialized per-question failure. |
| `retriever` | string | Configured DeepRAG backend: `BM25`, `HyperGraph`, or a future `Fusion`. |
| `retrieval_events` | array | Chronological DeepRAG retrieval attempts for this question. Empty when no retrieval was triggered. |
| `retrieve_count` | non-negative integer | Existing Counter delta. It must equal `len(retrieval_events)`. |
| `retrieval_latency_sec` | non-negative number | DeepRAG client-side retrieval elapsed time in seconds. |
| `latency_sec` | non-negative number | End-to-end inference elapsed time in seconds, including decomposition, retrieval, and generation. |

Existing fields such as `qa_pairs`, `all_results`, `generate_count`, `token_count`, `sentence_count`, and other Counter values remain allowed and unchanged.

`retrieval_latency_sec` is measured around each call to `BasicRAG.retrieve()`, not around individual vector, entity, or hyperedge operations. `latency_sec` remains the outer timing around the complete `model.inference()` call.

## Retrieval event

Each call to `BasicRAG.retrieve()` appends exactly one event, before returning or raising. `step` is one-based and contiguous in invocation order.

```json
{
  "step": 1,
  "query": "follow-up retrieval query",
  "backend": "BM25",
  "status": "success",
  "latency_sec": 0.12,
  "context_recorded": true,
  "context_block_count": 3,
  "context_char_count": 2841,
  "context_blocks": [
    {"source": "bm25", "rank": 1, "text": "..."},
    {"source": "bm25", "rank": 2, "text": "..."}
  ]
}
```

Required event fields are `step`, `query`, `backend`, `status`, `latency_sec`, `context_recorded`, `context_block_count`, and `context_char_count`.

- `backend` identifies the DeepRAG backend, not its internal subqueries. HyperGraph's vector, entity, and hyperedge work stays inside one `HyperGraph` event.
- For BM25, each returned document is one `context_blocks` item with `source: "bm25"` and a one-based `rank`.
- For HyperGraph, the complete HTTP context is exactly one block with `source: "hypergraph"` and `rank: 1`.
- A future Fusion backend must use one block per retained source/context unit, with an explicit lowercase `source` such as `bm25` or `hypergraph`.
- `context_block_count` is the number of returned blocks. `context_char_count` is the sum of their Unicode character lengths before any prompt truncation.

## Context recording switch

`--record_retrieval_context` defaults to `false` and is written to `config.json` with the other CLI arguments.

- When it is `true`, every successful event includes `context_blocks`, including each block's full `text`.
- When it is `false`, successful events omit `context_blocks`; they still record `query`, `backend`, `latency_sec`, `context_block_count`, and `context_char_count`.
- An omitted `context_blocks` field means content was deliberately not recorded. An empty array means content recording was enabled but the backend returned zero blocks.

The fixed 20-question BM25 and HyperGraph smoke scripts will explicitly set this switch to `true`. Larger experiments may set it to `false` to avoid oversized result files.

## Failure event and failure record

If a retrieval attempt raises, its event is appended with `status: "failure"`, the measured `latency_sec`, zero context counts, and a sanitized error object:

```json
{
  "step": 2,
  "query": "follow-up query",
  "backend": "HyperGraph",
  "status": "failure",
  "latency_sec": 300.01,
  "context_recorded": false,
  "context_block_count": 0,
  "context_char_count": 0,
  "error": {"type": "HyperGraphHTTPRetrieverError", "message": "retrieval request failed"}
}
```

No API key, authorization header, full endpoint credentials, or context text may appear in `error`.

When a runner is able to serialize a per-question failure, it uses the same top-level fields with `status: "failure"`, `prediction: null`, an optional sanitized top-level `error`, and all events completed before failure. `retrieve_count == len(retrieval_events)` still holds.

The current inference path deliberately re-raises exceptions. Until its error-handling policy is explicitly changed, a raised per-question exception can therefore leave no `output.jsonl` line; that case is reported as missing output by evaluation rather than fabricated as a failure record. Recording a failure event before re-raising does not change this exception behavior.

## Required invariants and validation

For a successful v2 record:

1. `retrieve_count == len(retrieval_events)`.
2. `retrieval_latency_sec` is the sum of event `latency_sec` values, allowing a small floating-point tolerance.
3. Each event's `step` equals its one-based array position.
4. `latency_sec >= retrieval_latency_sec` unless an explicitly documented timing instrumentation failure occurs.
5. All latency values are seconds and finite, non-negative JSON numbers.
6. `status: "success"` events do not contain `error`; `status: "failure"` events contain a sanitized `error` object.

The backend-specific retrieval settings are not compared by raw `top_k` values: BM25 document `top_k` and HyperGraph candidate `top_k` have different semantics. Each run's complete settings, including HyperGraph hybrid mode, token limits, timeout, and concurrency, are instead frozen in that run's `config.json`.

## Example: zero retrieval

```json
{
  "schema_version": 2,
  "qid": "example-qid",
  "question": "...",
  "prediction": ["..."],
  "answer": "...",
  "status": "success",
  "retriever": "BM25",
  "retrieval_events": [],
  "retrieve_count": 0,
  "retrieval_latency_sec": 0.0,
  "latency_sec": 0.84
}
```
