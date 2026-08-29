import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import FreshQA, WikiMultiHopQA, HotpotQA, IIRC
from transformers import AutoTokenizer, AutoModelForCausalLM 

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

no_Gen = ["single-retrieval-sft", "adaptive-retrieval-sft", "baseline-sft", "answer-aware", "adaptive-retrieval","srag-sample","srag-sftv2", "srag-allretrieve", "srag-nonretrieve", "iter-drag"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--remote_url", type=str, default=None)
    tmp = parser.parse_args()
    with open(os.path.join(tmp.dir, "config.json"), "r", encoding="utf-8") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.output_dir = tmp.dir
    args.remote_url = tmp.remote_url
    return args

def extract_answer(cot):
    import re
    cot = cot.split("<end>")[0]
    if "<answer short>" in cot:
        return cot.split("<answer short>")[-1].split("</answer short>")[0]
    elif "<answer long>" in cot:
        return cot.split("<answer long>")[-1].split("</answer long>")[0]
    else:
        cot = cot.split("</answer long>")[0].split("</answer short>")[0]
    
    if cot.endswith("<|im_end|>"):
        cot = cot[:-len("<|im_end|>")]
    if cot.endswith("<|eot_id|>"):
        cot = cot[:-len("<|eot_id|>")]
    pattern = r'<answer>([^<]+)</answer>(?!.*<answer>)'
    match = re.findall(pattern, cot)
    # print(cot)
    if len(match)>0:
        last_answer = match[-1]
        # print(last_answer)
        return last_answer
    else:
        return cot

def regenerate_answer(cot, tokenizer, model, case, demo):
    # print("##### origin #####")
    # print(cot)
    split_words = ["Question:", "#10000000", "Note:"]
    # split_words = ["Question:", "#10000000", "\n"]
    for word in split_words:
        pos = cot.find(word)
        if pos != -1 and pos > 0:
            cot = cot[:pos]
    if "the answer is" in cot:
        return cot
    cot = cot.rstrip().removesuffix("<|eot_id|>")
    
    cot += " So the answer is "
    prompt = "".join([d["case"]+"\n" for d in demo])
    prompt += case + " " + cot
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    input_length = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        max_new_tokens = 20)
    generated_tokens = outputs[:, input_length:]
    text = tokenizer.decode(generated_tokens[0])
    text = cot + text.strip()
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos] 
    # print("##### prompt #####")
    # print(prompt)
    # print("##### output #####")
    # print(text)
    # print("##### pred #####")
    return text

model_name = None
def regenerate_remote_answer(cot, tokenizer, model, case, demo):
    global model_name
    # print("##### origin #####")
    # print(cot)
    split_words = ["Question:", "#10000000", "Note:"]
    # split_words = ["Question:", "#10000000", "\n"]
    for word in split_words:
        pos = cot.find(word)
        if pos != -1 and pos > 0:
            cot = cot[:pos]
    
    if "the answer is" in cot:
        return cot 
    cot = cot.rstrip().removesuffix("<|eot_id|>")
    cot += " So the answer is "
    prompt = "".join([d["case"]+"\n" for d in demo])
    prompt += case + " " + cot
    text = model.completions.create(model=model_name,prompt=prompt,max_tokens=20).choices[0].text
    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # input_ids = input_ids.to(model.device)
    # input_length = input_ids.shape[1]
    # attention_mask = torch.ones_like(input_ids)
    # outputs = model.generate(
    #     input_ids = input_ids, 
    #     attention_mask = attention_mask, 
    #     max_new_tokens = 20)
    # generated_tokens = outputs[:, input_length:]
    # text = tokenizer.decode(generated_tokens[0])
    text = cot + text.strip()
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos] 
    # print("##### prompt #####")
    # print(prompt)
    # print("##### output #####")
    # print(text)
    # print("##### pred #####")
    return text


def _as_float(v):
    """安全地把值转换为 float；None 或非法值返回 None。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v):
    """安全地把值转换为 int；None 或非法值返回 0。"""
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _safe_mean(vals):
    """忽略 None 计算均值；全部为空时返回 None（避免除零）。"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _safe_percentile(vals, q):
    """忽略 None 计算分位数；全部为空时返回 None。"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.percentile(vals, q))


def _load_expected_sample_ids(args):
    """Load the optional fixed sample manifest recorded in config.json."""
    sample_ids_file = getattr(args, "sample_ids_file", None)
    if not sample_ids_file:
        return None
    with open(sample_ids_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Invalid sample IDs manifest: {sample_ids_file}")
    sample_ids = [item.get("id") for item in items if isinstance(item, dict)]
    if len(sample_ids) != len(items) or any(not isinstance(qid, str) or not qid for qid in sample_ids):
        raise ValueError(f"Sample IDs manifest contains an invalid ID: {sample_ids_file}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Sample IDs manifest contains duplicate IDs: {sample_ids_file}")
    return sample_ids


def _has_empty_retrieved_context(value):
    """Return whether a serialized inference trace contains an empty docs payload."""
    if isinstance(value, dict):
        if "docs" in value:
            docs = value["docs"]
            if docs is None or (hasattr(docs, "__len__") and len(docs) == 0):
                return True
        return any(_has_empty_retrieved_context(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_empty_retrieved_context(item) for item in value)
    return False


def main():
    args = get_args()
    logger.info(f"{args}")
    
    if args.dataset == '2wikimultihopqa':
        data = WikiMultiHopQA(args.data_path, args.split)
    elif args.dataset == 'hotpotqa':
        data = HotpotQA(args.data_path, args.split)
    elif args.dataset == 'iirc':
        data = IIRC(args.data_path)
    elif args.dataset == 'freshqa':
        data = FreshQA(args.data_path, args.split)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)

    dataset = {}
    for i in range(len(data.dataset)):
        t = data.dataset[i]
        dataset[t["qid"]] = [
            t["answer"], 
            t["answer_id"] if "answer_id" in t else None,
            t["case"] if "case" in t else None
        ]

    with open(os.path.join(args.output_dir, "output.jsonl"), "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    
    # 逐样本收集指标
    ems, f1s, precs, recalls = [], [], [], []
    retrieve_counts, generate_counts = [], []
    hallucinated_counts, token_counts, sentence_counts = [], [], []
    retrieval_latencies, latencies = [], []
    expected_sample_ids = _load_expected_sample_ids(args)
    observed_qids, failed_qids = set(), set()
    empty_context_count = 0
    
    need_generate = args.dataset in ['2wikimultihopqa', "hotpotqa", "iirc", "strategyqa"] 
    if args.method == "baseline-sft":
        need_generate = False
    if need_generate and args.method not in no_Gen:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if args.remote_url is None:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto",
                                                     trust_remote_code = "falcon" in args.model_name_or_path)
        else:
            from openai import OpenAI
            # openai_api_base = args.remote_url
            # openai_api_key = "EMPTY"
            # model = OpenAI(
            #     api_key=openai_api_key,
            #     base_url=openai_api_base,
            # )
            # global model_name
            # model_name = model.models.list().data[0].id

        demo = data.dataset[0]["demo"]
    pred_out = open(f"{args.output_dir}/details.txt", "w", encoding="utf-8")
    
    for line in tqdm(lines):
        rd = json.loads(line)
        qid = rd["qid"]
        pred = rd["prediction"]
        observed_qids.add(qid)
        if pred is None or pred is False:
            failed_qids.add(qid)
        if _has_empty_retrieved_context(pred):
            empty_context_count += 1
        # 修复：仅当记录中不存在 retrieve_count 字段时，才根据包含 docs 的轨迹节点回退计算
        if "retrieve_count" not in rd:
            rd["retrieve_count"] = sum([1 for p in pred if isinstance(p, dict) and "docs" in p]) if pred is not None else 0
        if pred is None:
            pred = ""
        if isinstance(pred, list):
            if pred[-1] == False or pred[-1] is None:
                pred = pred[-2]
            else:
                pred = pred[-1]
        if isinstance(pred, dict):
            pred = pred["answer"]
        
            
        ground_truth, ground_truth_id, case = dataset[qid]

        if pred.endswith("<|im_end|>"):
            pred = pred[:-len("<|im_end|>")]
        if pred.endswith("<|eot_id|>"):
            pred = pred[:-len("<|eot_id|>")]

        if args.method in no_Gen:
            pred = extract_answer(pred).split("<|eot_id|>")[0]
            # pred = pred.split(',')[0]
        elif args.method == "non-retrieval":
            pred = data.get_real_prediction(pred)
        else:
            # if need_generate:
            #     if args.remote_url is None:
            #         pred = regenerate_answer(pred, tokenizer, model, case, demo) 
            #     else:
            #         pred = regenerate_remote_answer(pred, tokenizer, model, case, demo) 
            
            pred = data.get_real_prediction(pred)
            pred = pred.split("So the answer is")[0].split(".")[0]


        em_ret = data.exact_match_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        f1_ret = data.f1_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        ems.append(em_ret["correct"])
        f1s.append(f1_ret["f1"])
        precs.append(f1_ret["precision"])
        recalls.append(f1_ret["recall"])

        retrieve_counts.append(_as_int(rd.get("retrieve_count")))
        generate_counts.append(_as_int(rd.get("generate_count")))
        hallucinated_counts.append(_as_int(rd.get("hallucinated_count")))
        token_counts.append(_as_int(rd.get("token_count")))
        sentence_counts.append(_as_int(rd.get("sentence_count")))
        retrieval_latencies.append(_as_float(rd.get("retrieval_latency_sec")))
        latencies.append(_as_float(rd.get("latency_sec")))
        detail = {
            "qid": qid, 
            "final_pred": pred,
            "EM": str(em_ret["correct"]), 
            "F1": str(f1_ret["f1"]) 
        }
        pred_out.write(json.dumps(detail)+"\n")
    pred_out.close()

    num_samples = len(ems)
    
    # 汇总指标
    em_mean = _safe_mean(ems)
    f1_mean = _safe_mean(f1s)
    prec_mean = _safe_mean(precs)
    recall_mean = _safe_mean(recalls)
    avg_retrieve = _safe_mean(retrieve_counts)
    avg_generate = _safe_mean(generate_counts)
    avg_hallucinated = _safe_mean(hallucinated_counts)
    avg_tokens = _safe_mean(token_counts)
    avg_sentence = _safe_mean(sentence_counts)
    valid_retrieval_latencies = [latency for latency in retrieval_latencies if latency is not None]
    avg_retrieval_latency = _safe_mean(valid_retrieval_latencies)
    p50_retrieval_latency = _safe_percentile(valid_retrieval_latencies, 50)
    p95_retrieval_latency = _safe_percentile(valid_retrieval_latencies, 95)

    if expected_sample_ids is None:
        expected_sample_ids = list(observed_qids)
    expected_qids = set(expected_sample_ids)
    missing_output_count = len(expected_qids - observed_qids)
    failed_count = len(expected_qids & failed_qids)
    success_count = len(expected_qids & observed_qids) - failed_count
    
    # 检索率：retrieve_count > 0 与 == 0 的样本比例
    if num_samples > 0:
        retrieval_rate = float(sum(1 for c in retrieve_counts if c > 0)) / num_samples
        zero_retrieval_rate = float(sum(1 for c in retrieve_counts if c == 0)) / num_samples
    else:
        retrieval_rate = None
        zero_retrieval_rate = None
    
    # 延迟相关：旧文件缺 latency 时写 null，不能报错
    valid_latencies = [l for l in latencies if l is not None]
    avg_latency = _safe_mean(valid_latencies)
    p50_latency = _safe_percentile(valid_latencies, 50)
    p95_latency = _safe_percentile(valid_latencies, 95)
    
    # 逻辑吞吐量 = 有效样本数 / 所有样本 latency 之和
    total_latency = sum(valid_latencies)
    if total_latency and total_latency > 0:
        throughput_qps = float(len(valid_latencies)) / total_latency
    else:
        throughput_qps = None
    
    # result.tsv（保留原有两列格式：指标名 + 均值）
    rows = [
        ["EM", em_mean],
        ["F1", f1_mean],
        ["Precision", prec_mean],
        ["Recall", recall_mean],
        ["retrieve_count", avg_retrieve],
        ["generate_count", avg_generate],
        ["hallucinated_count", avg_hallucinated],
        ["token_count", avg_tokens],
        ["sentence_count", avg_sentence],
        ["avg_retrieve_count", avg_retrieve],
        ["avg_retrieval_latency_sec", avg_retrieval_latency],
        ["p50_retrieval_latency_sec", p50_retrieval_latency],
        ["p95_retrieval_latency_sec", p95_retrieval_latency],
        ["retrieval_rate", retrieval_rate],
        ["zero_retrieval_rate", zero_retrieval_rate],
        ["avg_latency_sec", avg_latency],
        ["p50_latency_sec", p50_latency],
        ["p95_latency_sec", p95_latency],
        ["avg_generate_count", avg_generate],
        ["avg_output_tokens", avg_tokens],
        ["throughput_qps", throughput_qps],
        ["success_count", success_count],
        ["failure_count", failed_count],
        ["missing_output_count", missing_output_count],
        ["empty_context_count", empty_context_count],
    ]
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False, encoding="utf-8")
    
    # 按 em=0 / em=1 观察检索次数（调试信息）
    if num_samples > 0:
        ems_arr = np.array(ems)
        retr_arr = np.array(retrieve_counts)
        em_0 = retr_arr[ems_arr == 0]
        em_1 = retr_arr[ems_arr == 1]
        if len(em_0) > 0 and len(em_1) > 0:
            print(f"em=0: {em_0.mean()}, em=1: {em_1.mean()}")
    
    # metrics.json（使用数值类型，缺失时写 null）
    metrics = {
        "num_samples": num_samples,
        "EM": em_mean,
        "F1": f1_mean,
        "Precision": prec_mean,
        "Recall": recall_mean,
        "avg_retrieve_count": avg_retrieve,
        "avg_retrieval_latency_sec": avg_retrieval_latency,
        "p50_retrieval_latency_sec": p50_retrieval_latency,
        "p95_retrieval_latency_sec": p95_retrieval_latency,
        "retrieval_rate": retrieval_rate,
        "zero_retrieval_rate": zero_retrieval_rate,
        "avg_latency_sec": avg_latency,
        "p50_latency_sec": p50_latency,
        "p95_latency_sec": p95_latency,
        "avg_generate_count": avg_generate,
        "avg_output_tokens": avg_tokens,
        "throughput_qps": throughput_qps,
        "avg_sentence_count": avg_sentence,
        "avg_hallucinated_count": avg_hallucinated,
        "success_count": success_count,
        "failure_count": failed_count,
        "missing_output_count": missing_output_count,
        "empty_context_count": empty_context_count,
    }
    with open(f"{args.output_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
