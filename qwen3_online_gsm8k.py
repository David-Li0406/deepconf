#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warm-up + Online reasoning with dataset-level DeepConf threshold for GSM8K (main test split).

Phase 1 (warm-up):
  - Randomly sample `warmup_n` questions
  - For each, run `N_init` traces WITHOUT early stop
  - Collect the lowest group confidence per trace and compute a single dataset-level threshold s
  - Save all warm-up traces to {save_pred}.warmup.jsonl

Phase 2 (online):
  - For ALL questions, run exactly ONE pseudo-online generation with early stop using threshold s
  - If no extractable final answer in the truncated text, force-finalize via a 'Final Answer:' cue;
    if still none, fall back to the full text answer or a default
  - Save per-question record to {save_pred} and report accuracy

Notes:
  - No vLLM; pure HF Transformers (AutoModelForCausalLM/AutoTokenizer)
  - Multiprocess enabled; set --num_workers and --device_ids
  - 'enable_thinking' is attempted in chat template but gracefully falls back if unsupported
"""

import os
import re
import json
import math
import random
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from fractions import Fraction

# ============================== Args ==============================

def build_parser():
    ap = argparse.ArgumentParser("Qwen3 warm-up+online for GSM8K (dataset-level DeepConf)")
    # model / data
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    # sampling
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=2048)

    # DeepConf core
    ap.add_argument("--N_init", type=int, default=4, help="warm-up traces per sampled question")
    ap.add_argument("--eta", type=float, default=10.0, choices=[10.0, 90.0])
    ap.add_argument("--mode", type=str, default="deepconf-low", choices=["deepconf-low","deepconf-high"])
    ap.add_argument("--group_window", type=int, default=128)
    ap.add_argument("--topk_conf", type=int, default=5)

    # run control
    ap.add_argument("--max_questions", type=int, default=-1)
    ap.add_argument("--save_pred", type=str, default="gsm8k_dc_dataset_thr.jsonl")

    # warm-up sampling
    ap.add_argument("--warmup_n", type=int, default=100)

    # multiprocessing
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--device_ids", type=str, default="")
    return ap

# ======================= Prompting & extraction ======================

def build_messages(question: str) -> list:
    user = (
        f"{question}\n\n"
        "Solve step by step. End with: Final Answer: \\\\boxed{<number>}."
    )
    return [{"role": "user", "content": user}]

def _find_boxed_contents(s: str):
    results = []
    pos = 0
    while True:
        m = re.search(r'\\boxed\s*\{', s[pos:])
        if not m: break
        start_brace = pos + m.end() - 1
        i = start_brace + 1; depth = 1
        while i < len(s) and depth > 0:
            c = s[i]
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            i += 1
        if depth == 0:
            content = s[start_brace+1:i-1]
            results.append(content)
            pos = i
        else:
            break
    return results

def _normalize_latex_to_plain(content: str) -> str:
    content = re.sub(
        r'\\(?:d?frac)\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}',
        r'\1/\2', content
    )
    content = re.sub(r'\\[a-zA-Z]+', '', content)
    content = content.replace('$', '').replace(',', ' ')
    content = content.replace('{', ' ').replace('}', ' ')
    content = re.sub(r'\s+', ' ', content).strip()
    return content

def extract_numbers_from_boxed_after_final_answer(text: str):
    if not text: return []
    finals = list(re.finditer(r'Final Answer[:：]?\s*', text, re.IGNORECASE))
    boxed_content = None
    if finals:
        tail = text[finals[-1].end():]
        tail_boxed = _find_boxed_contents(tail)
        if tail_boxed: boxed_content = tail_boxed[0]
    if boxed_content is None:
        all_boxed = _find_boxed_contents(text)
        if all_boxed: boxed_content = all_boxed[-1]
        else: return []
    content = _normalize_latex_to_plain(boxed_content)
    num_pat = r'-?\d+(?:\.\d+)?(?:/\d+)?'
    raw_nums = re.findall(num_pat, content)
    numbers = []
    for s in raw_nums:
        try:
            if '/' in s:
                f = Fraction(s)
                numbers.append(int(f) if f.denominator == 1 else float(f))
            elif '.' in s:
                numbers.append(float(s))
            else:
                numbers.append(int(s))
        except Exception:
            continue
    return numbers

def extract_gt_from_gsm8k_answer(ans_text: str):
    m = re.search(r"####\s*(-?\d+)", ans_text)
    return int(m.group(1)) if m else None

# ========================= Confidence utils =========================

def token_confidence_from_logits(logits: torch.Tensor, k: int, temperature: float) -> float:
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    topk = min(k, probs.size(-1))
    pvals, _ = torch.topk(probs, k=topk, dim=-1)
    pvals = torch.clamp(pvals, min=1e-12)
    Ci = - torch.log(pvals).mean().item()
    return Ci

def lowest_group_conf(conf_hist, w: int) -> float:
    if not conf_hist:
        return float("inf")
    w = min(w, len(conf_hist))
    ps = [0.0]
    for c in conf_hist:
        ps.append(ps[-1] + c)
    lows = float("inf")
    for end in range(w, len(ps)):
        cg = (ps[end] - ps[end - w]) / w
        if cg < lows:
            lows = cg
    return lows

def percentile(values, q):
    if not values:
        return -float("inf")
    vals = sorted(values)
    rank = (len(vals) - 1) * (q / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (rank - lo)

# ===================== Force-finalize helper (ONLINE) =====================
def _build_force_prompt_from_text(text: str):
    base = (text or "").rstrip()
    if "</think>" not in base:
        base = base + "\n</think>"
    return base + "\nFinal Answer: "

def force_finalize_answer(model, tokenizer, text, device, args, max_new_tokens: int = 24):
    prompt = _build_force_prompt_from_text(text)
    input_ids = tokenizer([prompt], return_tensors="pt").to(device).input_ids
    gen_kwargs = dict(
        do_sample=False, temperature=0.0, top_p=1.0,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True, output_scores=False, use_cache=True,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
    )
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, **gen_kwargs)
        full = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0]
        nums = extract_numbers_from_boxed_after_final_answer(full)
        return nums[0] if nums else None

# ===================== Pseudo-online generation ====================

def generate_one_trace_online(model, tokenizer, prompt_ids, device, args, stop_threshold=None):
    model.eval()
    gen_kwargs = dict(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
    )
    with torch.no_grad():
        out = model.generate(input_ids=prompt_ids, **gen_kwargs)
        full_gen_ids = out.sequences[:, prompt_ids.shape[1]:]
        gen_len = full_gen_ids.shape[1]
        scores = out.scores

        conf_hist = []
        cut_idx = gen_len
        for t in range(gen_len):
            logits_t = scores[t].squeeze(0)
            Ci = token_confidence_from_logits(logits_t, args.topk_conf, args.temperature)
            conf_hist.append(Ci)

            if stop_threshold is not None:
                w = min(args.group_window, len(conf_hist))
                cg_now = sum(conf_hist[-w:]) / w
                if cg_now < stop_threshold:
                    cut_idx = t + 1
                    break

        trunc_ids = full_gen_ids[:, :cut_idx]
        text = tokenizer.batch_decode(trunc_ids, skip_special_tokens=True)[0] if cut_idx > 0 else ""
        
        text_full = tokenizer.batch_decode(full_gen_ids, skip_special_tokens=True)[0]
        if not _find_boxed_contents(text):
            # 在完整输出里找第一个 \boxed{...} 的结束位置，并把截断推进到那里
            m = re.search(r'\\boxed\s*\{', text_full)
            if m:
                # 手动匹配花括号，确保正确找到与 \boxed{ 对应的第一个闭合 }
                start_brace = m.end() - 1  # 指向 '{'
                i = start_brace + 1
                depth = 1
                while i < len(text_full) and depth > 0:
                    c = text_full[i]
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    text = text_full[:i]  # 包含完整的第一个 \boxed{...}

        nums = extract_numbers_from_boxed_after_final_answer(text)
        ans = nums[0] if nums else None
        nums_full = extract_numbers_from_boxed_after_final_answer(text_full)
        ans_full = nums_full[0] if nums_full else None

        lowest_gc = lowest_group_conf(conf_hist[:cut_idx], args.group_window)
        tokens_used = int(cut_idx)
        return text, text_full, ans, ans_full, conf_hist[:cut_idx], lowest_gc, tokens_used

# ============================ Workers ==============================

def worker_warmup(args, warm_indices, part_name, device_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    torch.manual_seed(args.seed + int(part_name))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": device_id} if torch.cuda.is_available() else "auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    device = model.device

    ds = load_dataset("openai/gsm8k", name="main", split="test")

    warm_logs = []
    warm_lowests_all = []

    for g_idx in warm_indices:
        item = ds[int(g_idx)]
        question = item["question"]

        messages = build_messages(question)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompt_ids = tokenizer([prompt_text], return_tensors="pt").to(device).input_ids

        traces = []
        for _ in range(args.N_init):
            text, _, ans, _, conf_hist, lowest_gc, tokens_used = generate_one_trace_online(
                model, tokenizer, prompt_ids, device, args, stop_threshold=None
            )
            traces.append({
                "text": text, "ans": ans, "conf_hist": conf_hist,
                "lowest_gc": lowest_gc, "tokens_used": tokens_used
            })
            warm_lowests_all.append(lowest_gc)

        warm_logs.append({"index": int(g_idx), "warm_traces": traces})

    part_tag = f"W{part_name:02d}"
    with open(f"{args.save_pred}.warm.part{part_tag}.jsonl", "w", encoding="utf-8") as f:
        for r in warm_logs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{args.save_pred}.warm.lowests.part{part_tag}.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"lowests": warm_lowests_all}, ensure_ascii=False) + "\n")


def worker_online(args, indices, part_name, device_id, stop_threshold):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": device_id} if torch.cuda.is_available() else "auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    device = model.device

    ds = load_dataset("openai/gsm8k", name="main", split="test")

    part_logs = []
    for g_idx in tqdm(indices):
        item = ds[int(g_idx)]
        question = item["question"]
        gt = extract_gt_from_gsm8k_answer(item["answer"])

        messages = build_messages(question)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompt_ids = tokenizer([prompt_text], return_tensors="pt").to(device).input_ids

        text, text_full, ans, ans_full, conf_hist, lowest_gc, tokens_used = generate_one_trace_online(
            model, tokenizer, prompt_ids, device, args, stop_threshold=stop_threshold
        )

        forced_used = False
        pred = ans
        if pred is None:
            forced = force_finalize_answer(model, tokenizer, text, device, args)
            if forced is not None:
                pred = forced
                forced_used = True
            else:
                pred = ans_full

        part_logs.append({
            "index": int(g_idx),
            "gt": gt,
            "pred": pred,
            "forced_pred_used": forced_used,
            "threshold": float(stop_threshold),
            "group_window": args.group_window,
            "online_trace": {
                "text": text, "text_full": text_full, "ans": ans, "ans_full": ans_full,
                "conf_hist": conf_hist, "lowest_gc": lowest_gc, "tokens_used": tokens_used
            }
        })

    part_path = f"{args.save_pred}.online.part{part_name}.jsonl"
    with open(part_path, "w", encoding="utf-8") as f:
        for r in part_logs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ONLINE {part_name}] done: {len(indices)} items → {part_path}")

# ====================== Parent: split/launch/merge =====================

def split_indices(n_total, n_workers):
    base = n_total // n_workers
    rem = n_total % n_workers
    parts = []
    start = 0
    for w in range(n_workers):
        cnt = base + (1 if w < rem else 0)
        parts.append(list(range(start, start + cnt)))
        start += cnt
    return parts

def main():
    args = build_parser().parse_args()
    if args.mode == "deepconf-low":
        args.eta = 10.0
    elif args.mode == "deepconf-high":
        args.eta = 90.0

    if args.device_ids.strip():
        devices = [int(x) for x in args.device_ids.split(",") if x.strip()]
    else:
        devices = list(range(max(1, args.num_workers)))
    n_workers = max(1, args.num_workers)
    assert len(devices) >= n_workers

    ds = load_dataset("openai/gsm8k", name="main", split="test")
    n_total_all = len(ds)
    n_total = n_total_all if args.max_questions < 0 else min(n_total_all, args.max_questions)
    all_indices = list(range(n_total))

    random.seed(args.seed)
    warmup_n = min(args.warmup_n, n_total)
    warm_indices = random.sample(all_indices, warmup_n)

    warm_parts = split_indices(len(warm_indices), n_workers)

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    procs = []
    for i, idxs_local in enumerate(warm_parts):
        shard_global = [warm_indices[j] for j in idxs_local]
        p = mp.Process(target=worker_warmup, args=(args, shard_global, i, devices[i]))
        p.start(); procs.append(p)
    for p in procs: p.join()

    warm_merge = 0; warm_lowests_all = []
    with open(f"{args.save_pred}.warmup.jsonl", "w", encoding="utf-8") as fout:
        for i in range(len(warm_parts)):
            pf = f"{args.save_pred}.warm.partW{i:02d}.jsonl"
            if os.path.exists(pf):
                with open(pf, "r", encoding="utf-8") as fin:
                    for line in fin: fout.write(line); warm_merge += 1
        for i in range(len(warm_parts)):
            sf = f"{args.save_pred}.warm.lowests.partW{i:02d}.jsonl"
            if os.path.exists(sf):
                with open(sf, "r", encoding="utf-8") as fin:
                    for line in fin:
                        try:
                            obj = json.loads(line); warm_lowests_all.extend(obj.get("lowests", []))
                        except Exception: pass
    print(f"[WARM MERGE] Saved {warm_merge} warm-up questions to {args.save_pred}.warmup.jsonl")

    s = percentile(warm_lowests_all, 100.0 - args.eta)
    print(f"[DATASET THRESHOLD] eta={args.eta} → s={s:.6f} (from {len(warm_lowests_all)} traces)")

    parts = split_indices(n_total, n_workers)

    procs = []
    for i, idxs_local in enumerate(parts):
        shard_global = [all_indices[j] for j in idxs_local]
        p = mp.Process(target=worker_online, args=(args, shard_global, f"O{i:02d}", devices[i], s))
        p.start(); procs.append(p)
    for p in procs: p.join()

    merged = 0; all_correct = 0; all_total = 0
    with open(args.save_pred, "w", encoding="utf-8") as fout:
        for i in range(len(parts)):
            pf = f"{args.save_pred}.online.partO{i:02d}.jsonl"
            if not os.path.exists(pf): continue
            with open(pf, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line); merged += 1
                    try:
                        row = json.loads(line)
                        gt = row.get("gt"); pred = row.get("pred")
                        if gt is not None and pred is not None:
                            all_total += 1
                            if int(gt) == int(pred):
                                all_correct += 1
                    except Exception:
                        pass
    print(f"[ONLINE MERGE] Wrote {merged} lines to {args.save_pred}")
    if all_total > 0:
        acc = all_correct / all_total * 100.0
        print(f"[RESULT] Accuracy = {all_correct}/{all_total} = {acc:.2f}%")

if __name__ == "__main__":
    main()
