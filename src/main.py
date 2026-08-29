import os
import json
import argparse
from textwrap import indent
from tqdm import tqdm
from copy import copy
import logging
import time
import torch
from data import ASQA, WikiMultiHopQA, HotpotQA, IIRC, FreshQA
from generate import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to run simultaneously")
    parser.add_argument("--gpus_per_process", type=float, default=1, help="Number of GPUs each process uses for inference")
    parser.add_argument("--remote_url", type=str, nargs='+', default=None, help="Remote URL for Ray")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--sample", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--sample_ids_file", type=str, default=None,
                        help="JSON subset manifest; selects samples in manifest ID order")
    parser.add_argument("--generate_max_length", type=int, default=100, help="Max length for generation")
    parser.add_argument("--fewshot", type=int, default=8, help="Number of fewshot examples")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--es_index_name", type=str, default="wiki")
    parser.add_argument("--retriever", type=str, default="BM25")
    parser.add_argument("--retrieve_topk", type=int, default=3)
    parser.add_argument("--hypergraph_url", type=str, default="http://127.0.0.1:8765")
    parser.add_argument("--hypergraph_timeout", type=float, default=300)
    parser.add_argument("--hypergraph_topk", type=int, default=10)
    parser.add_argument("--hypergraph_max_text_tokens", type=int, default=2000)
    parser.add_argument("--hypergraph_max_local_tokens", type=int, default=1000)
    parser.add_argument("--hypergraph_max_global_tokens", type=int, default=1000)
    parser.add_argument("--record_retrieval_context", action="store_true",
                        help="Record retrieval context text in per-question retrieval events")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the data")
    parser.add_argument("--vllm", default=False, action="store_true", help="Use vLLM")
    parser.add_argument("--follow_up_remote_url", type=str,  nargs='+', default=None, help="Follow up remote url")
    parser.add_argument("--hallucination_threshold", type=float, default=0.1)
    parser.add_argument("--sentence_solver", type=str, default="avg")
    parser.add_argument("--query_formulation", type=str, default="direct")
    parser.add_argument("--check_real_words", type=bool, default=False)
    args = parser.parse_args()
    # HyperGraph service-side settings are fixed and recorded with each client run.
    args.hypergraph_mode = "hybrid"
    args.hypergraph_only_need_context = True
    args.hypergraph_service_max_concurrency = 1
    args.use_counter = True
    return args

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def load_sample_ids(path):
    """Read ordered sample IDs from a reproducible subset manifest."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Unable to read sample_ids_file {}: {}".format(path, type(exc).__name__)) from exc
    items = manifest.get("items") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("sample_ids_file must contain a non-empty items list")
    sample_ids = []
    seen_ids = set()
    duplicate_ids = set()
    for position, item in enumerate(items):
        sample_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_ids_file item {} has no valid id".format(position))
        if sample_id in seen_ids:
            duplicate_ids.add(sample_id)
        seen_ids.add(sample_id)
        sample_ids.append(sample_id)
    if duplicate_ids:
        raise ValueError("sample_ids_file contains duplicate IDs: {}".format(", ".join(sorted(duplicate_ids))))
    return sample_ids


def select_samples_by_id(data, sample_ids):
    """Select all requested IDs strictly and preserve the manifest's ordering."""
    data_by_id = {}
    duplicate_dataset_ids = set()
    for record in data:
        sample_id = record.get("qid")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Dataset contains a record with no valid qid")
        if sample_id in data_by_id:
            duplicate_dataset_ids.add(sample_id)
        data_by_id[sample_id] = record
    if duplicate_dataset_ids:
        raise ValueError("Dataset contains duplicate qid values: {}".format(
            ", ".join(sorted(str(sample_id) for sample_id in duplicate_dataset_ids))))
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in data_by_id]
    if missing_ids:
        raise ValueError("sample_ids_file IDs missing from dataset: {}".format(", ".join(missing_ids)))
    return [data_by_id[sample_id] for sample_id in sample_ids]

def process_data_ray(args, data_indices, data_dict, process_idx):
    import os
    import torch
    from copy import copy
    from data import WikiMultiHopQA, HotpotQA, IIRC
    from datasets import Dataset

    # Initialize logger for this process
    logger = logging.getLogger(f"process_{process_idx}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [Process %(name)s] %(levelname)s: %(message)s'))
    if not logger.hasHandlers():
        logger.addHandler(handler)

    logger.info(f"Process {process_idx} starting")
    logger.info(f"Remaining: {len(data_indices)} data")
    # Ray assigns GPUs automatically; no need to set CUDA_VISIBLE_DEVICES

    # Reconstruct the dataset subset from indices and data_dict
    # data_subset = Dataset.from_dict({key: [data_dict[key][i] for i in data_indices] for key in data_dict})
    data_subset = [data_dict[i] for i in data_indices]

    # Select model based on method
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "baseline-sft":
        model = BaselineSFTRAG(args)
    elif args.method == "token":
        model = TokenRAG(args)
    elif args.method == "srag-sample":
        model = SRAGSampleV2(args)
    elif args.method == "srag-sftv2":
        model = SRAGSFTV2(args)
    elif args.method == "sample-dpo":
        model = SRAGSampleDPO(args)
    elif args.method == "iter-drag":
        model = IterDRAG(args)
    else:
        raise NotImplementedError

    logger.info("Start inference")

    outputs = []
    output_file_path = os.path.join(args.output_dir, f"output_subprocess_{process_idx}.jsonl")
    all_results = None
    with open(output_file_path, "a+", encoding="utf-8") as output_file:
        for i in tqdm(range(len(data_subset)), desc=f"Process {process_idx}"):
            last_counter = copy(model.counter)
            batch = data_subset[i]
            model.reset_retrieval_events()
            # 统计端到端延迟：从调用 model.inference 开始到返回预测结果为止
            latency_sec = None
            t_start = time.perf_counter()
            try:
                if args.method == "answer-aware":
                    pred = model.inference(batch["question"], batch["demo"], batch["case"], batch["answer"])
                elif args.method == "srag-sample" or args.method == "srag-sample-allretrieve" or args.method == "sample-dpo" or args.method == "srag-sample-all":
                    pred, all_results = model.inference(**batch)
                else:
                    pred = model.inference(batch["question"], batch["demo"], batch["case"])
            except Exception:
                # 推理异常时不生成错误的延迟数据，保持原有异常向上传播的行为
                raise
            latency_sec = time.perf_counter() - t_start
            if isinstance(pred, str):
                pred = pred.strip()

            ret = {
                "schema_version": 2,
                "qid": batch["qid"],
                "question": batch["question"],
                "prediction": pred,
                "answer": batch["answer"],
                "status": "success",
                "retriever": args.retriever,
                "retrieval_events": list(model.retrieval_events),
                "qa_pairs": batch.get("qa_pairs",[]),
                "all_results": all_results,
                "latency_sec": latency_sec,
            }
            if args.use_counter:
                ret.update(model.counter.calc(last_counter))
                if ret["retrieve_count"] != len(ret["retrieval_events"]):
                    raise RuntimeError(
                        "retrieve_count does not match the number of retrieval_events"
                    )
            outputs.append(ret)
            output_file.write(json.dumps(ret, ensure_ascii=False) + "\n")
            output_file.flush()
            
    logger.info(f"Process {process_idx} finished")
    return outputs

def main():
    args = get_args()
    logger = setup_logging()
    logger.info(f"{args}")

    # Output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dir_name = os.listdir(args.output_dir)
    for i in range(10000):
        if str(i) not in dir_name:
            if args.resume:
                args.output_dir = os.path.join(args.output_dir, str(i-1))
            else:
                args.output_dir = os.path.join(args.output_dir, str(i))
            os.makedirs(args.output_dir, exist_ok=True)
            break
    logger.info(f"Output dir: {args.output_dir}")
    if args.remote_url:
        # if args.remote_url=="gpt4":
        if not isinstance(args.remote_url, list):
            args.remote_url = [args.remote_url]
        from openai import OpenAI
        openai_api_base = args.remote_url
        openai_api_key = "EMPTY"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base[0],
        )
        models = client.models.list()
        model = models.data[0].id
        with open(args.output_dir+'/model_id.txt','w', encoding="utf-8") as f:
                f.write(model)
            
    # Save config  m  
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=4)

    # Load data
    if args.dataset == "2wikimultihopqa":
        data_loader = WikiMultiHopQA(args.data_path, args.split)
    elif args.dataset == "hotpotqa":
        data_loader = HotpotQA(args.data_path, args.split)
    elif args.dataset == "iirc":
        data_loader = IIRC(args.data_path)
    elif args.dataset == "asqa":
        data_loader = ASQA(args.data_path, args.split)
    elif args.dataset == "freshqa":
        data_loader = FreshQA(args.data_path, args.split)
    else:
        raise NotImplementedError
    data_loader.format(fewshot=args.fewshot)
    data = data_loader.dataset
    if args.sample_ids_file:
        sample_ids = load_sample_ids(args.sample_ids_file)
        data = select_samples_by_id(data, sample_ids)
        logger.info("Selected %d samples from %s in manifest order", len(data), args.sample_ids_file)
    else:
        if args.shuffle:
            data = data.shuffle(seed=42)
        if args.sample != -1:
            samples = min(len(data), args.sample)
            # data = data.select(range(samples))
            data = data[:samples]

    # Convert dataset to a serializable format
    # data_dict = data.to_dict()
    data_dict = data

    # Prepare for Ray parallel processing
    num_cpus = args.num_processes  # Assuming 1 CPU per process

    total_samples = len(data)
    num_processes = args.num_processes

    if num_processes > 1:
        import ray
        # Initialize Ray
        ray.init()
        if args.remote_url: # no use gpu
            get_answers_func = ray.remote(num_cpus=0.03)(
                process_data_ray
            ).remote
        else: # distribute gpu
            get_answers_func = ray.remote(num_gpus=args.gpus_per_process)(
                process_data_ray
            ).remote
    else:
        get_answers_func = process_data_ray
        
    already_processed_id = set()
    if args.resume:
        for file in os.listdir(args.output_dir):
            if file.startswith("output_subprocess_"):
                with open(os.path.join(args.output_dir, file), "r", encoding="utf-8") as f:
                    for line in f:
                        ret = json.loads(line)
                        already_processed_id.add(ret["qid"])
                        
    # Split data indices into chunks, ensuring even distribution
    indices = [idx for idx in range(total_samples) if data_dict[idx]["qid"] not in already_processed_id]

    total_valid_samples = len(indices)
    chunk_size = total_valid_samples // num_processes
    remainder = total_valid_samples % num_processes
    
    data_indices_splits = []
    start_idx = 0
    for i in range(num_processes):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        data_indices_splits.append(indices[start_idx:end_idx])
        start_idx = end_idx

    # Launch Ray tasks
    futures = []
    for process_idx in range(num_processes):
        data_indices = data_indices_splits[process_idx]
        future = get_answers_func(args, data_indices, data_dict, process_idx)
        futures.append(future)

    # Gather results
    logger.info("Waiting for all processes to finish...")
    if num_processes > 1:
        results = ray.get(futures)
    else:
        results = futures

    # Merge outputs
    logger.info("Merging outputs from all processes")
    merged_outputs = []
    for output in results:
        merged_outputs.extend(output)

    # Sort outputs if order is important (e.g., by 'qid' or original data order)
    # merged_outputs.sort(key=lambda x: x['qid'])

    with open(os.path.join(args.output_dir, "output.jsonl"), "w", encoding="utf-8") as outfile:
        for ret in merged_outputs:
            outfile.write(json.dumps(ret, ensure_ascii=False) + "\n")

    logger.info(f"All outputs have been merged into {os.path.join(args.output_dir, 'output.jsonl')}")


if __name__ == "__main__":
    main()
