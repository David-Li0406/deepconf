#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warm-up + Online reasoning with dataset-level DeepConf threshold (GPT-OSS-20B, no <think> tokens).

Phase 1 (warm-up):
  - Randomly sample `warmup_n` questions
  - For each, run `N_init` traces WITHOUT early stop
  - Collect the lowest group confidence per trace and compute a single dataset-level threshold s
  - Save all warm-up traces to {save_pred}.warmup.jsonl

Phase 2 (online):
  - For ALL questions, run exactly ONE pseudo-online generation with early stop using threshold s
  - If no extractable final answer in the truncated text, force-finalize via a 'Final Answer:' cue;
    if still none, fall back to the full text answer or a default
  - Save per-question record to {save_pred} and (optionally) report accuracy

Notes:
  - Pure HF Transformers (AutoModelForCausalLM/AutoTokenizer)
  - Multiprocess support omitted for simplicity; set CUDA_VISIBLE_DEVICES to choose GPU.
  - IMPORTANT: This version is for GPT-OSS-20B which does NOT use <think> tokens.
"""

import os
import re
import json
import math
import random
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ------------------------- DeepConf utilities -------------------------

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

def build_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    return tokenizer, model

def force_finalize_answer_text(text: str) -> str:
    """For GPT-OSS (no <think>), just append 'Final Answer:' cue to force output."""
    base = (text or "").rstrip()
    if "<|end|><|start|>assistant<|channel|>final<|messagel>" not in base:
        base = base + "\n<|end|><|start|>assistant<|channel|>final<|messagel>"
    return base + "\nFinal Answer: "

def generate_one_trace_online(model, tokenizer, prompt_ids, temperature: float, top_p: float,
                              max_new_tokens: int, topk_conf: int, group_window: int,
                              stop_threshold: Optional[float] = None):
    model.eval()
    gen_kwargs = dict(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
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
            Ci = token_confidence_from_logits(logits_t, topk_conf, temperature)
            conf_hist.append(Ci)

            if stop_threshold is not None:
                w = min(group_window, len(conf_hist))
                cg_now = sum(conf_hist[-w:]) / w
                if cg_now < stop_threshold:
                    cut_idx = t + 1
                    break

        trunc_ids = full_gen_ids[:, :cut_idx]
        text = tokenizer.batch_decode(trunc_ids, skip_special_tokens=True)[0] if cut_idx > 0 else ""
        text_full = tokenizer.batch_decode(full_gen_ids, skip_special_tokens=True)[0]
        lowest_gc = lowest_group_conf(conf_hist[:cut_idx], group_window)
        tokens_used = int(cut_idx)
        return text, text_full, conf_hist[:cut_idx], lowest_gc, tokens_used

def main_common_before(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, model = build_model(args.model_name)
    return tokenizer, model

def compute_dataset_threshold(lowests_all, eta: float):
    # eta=10 → keep ratio η means we choose percentile 100-η over lowest group confidences
    q = 100.0 - eta
    return percentile(lowests_all, q)

from datasets import load_dataset
from fractions import Fraction

def build_messages_gsm8k(question: str) -> list:
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

def extract_gt(ans_text: str):
    m = re.search(r"####\s*(-?\d+)", ans_text)
    return int(m.group(1)) if m else None

def build_parser():
    ap = argparse.ArgumentParser("GPT-OSS-20B warm-up+online for GSM8K (dataset-level DeepConf)")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--N_init", type=int, default=4)
    ap.add_argument("--eta", type=float, default=10.0, choices=[10.0, 90.0])
    ap.add_argument("--group_window", type=int, default=128)
    ap.add_argument("--topk_conf", type=int, default=5)
    ap.add_argument("--max_questions", type=int, default=-1)
    ap.add_argument("--warmup_n", type=int, default=100)
    ap.add_argument("--save_pred", type=str, default="gsm8k_dc_dataset_thr.gptoss.jsonl")
    return ap

def warmup_phase(args, tokenizer, model):
    ds = load_dataset("openai/gsm8k", name="main", split="test")
    indices = list(range(len(ds)))
    if args.max_questions > 0: indices = indices[:args.max_questions]
    warm_indices = random.sample(indices, min(args.warmup_n, len(indices)))

    lowests_all = []
    warm_logs = []
    for idx in tqdm(warm_indices, desc="Warm-up"):
        item = ds[int(idx)]
        messages = build_messages_gsm8k(item["question"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer([prompt], return_tensors="pt").to(model.device).input_ids

        traces = []
        for _ in range(args.N_init):
            text, _, conf_hist, lowest_gc, tokens_used = generate_one_trace_online(
                model, tokenizer, prompt_ids,
                args.temperature, args.top_p, args.max_new_tokens,
                args.topk_conf, args.group_window,
                stop_threshold=None
            )
            traces.append({"text": text, "conf_hist": conf_hist, "lowest_gc": lowest_gc, "tokens_used": tokens_used})
            lowests_all.append(lowest_gc)

        warm_logs.append({"index": int(idx), "warm_traces": traces})

    with open(f"{args.save_pred}.warmup.jsonl","w",encoding="utf-8") as f:
        for r in warm_logs: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    s = compute_dataset_threshold(lowests_all, args.eta)
    with open(f"{args.save_pred}.warmup.stats.json","w",encoding="utf-8") as f:
        json.dump({"threshold": s, "eta": args.eta, "count": len(lowests_all)}, f, ensure_ascii=False, indent=2)
    return s

def online_phase(args, tokenizer, model, stop_threshold: float):
    ds = load_dataset("openai/gsm8k", name="main", split="test")
    indices = list(range(len(ds)))
    if args.max_questions > 0: indices = indices[:args.max_questions]

    part_logs = []
    correct = total = 0
    for idx in tqdm(indices, desc="Online"):
        item = ds[int(idx)]
        gt = extract_gt(item["answer"])
        messages = build_messages_gsm8k(item["question"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer([prompt], return_tensors="pt").to(model.device).input_ids

        text, text_full, conf_hist, lowest_gc, tokens_used = generate_one_trace_online(
            model, tokenizer, prompt_ids,
            args.temperature, args.top_p, args.max_new_tokens,
            args.topk_conf, args.group_window,
            stop_threshold=stop_threshold
        )
        # parse answers
        nums = extract_numbers_from_boxed_after_final_answer(text)
        ans = nums[0] if nums else None
        if ans is None:
            forced_prompt = force_finalize_answer_text(text)
            forced_ids = tokenizer([forced_prompt], return_tensors="pt").to(model.device).input_ids
            with torch.no_grad():
                out = model.generate(input_ids=forced_ids, do_sample=False, temperature=0.0, top_p=1.0,
                                     max_new_tokens=24, return_dict_in_generate=True)
                forced_text = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0]
            nums_forced = extract_numbers_from_boxed_after_final_answer(forced_text)
            if nums_forced: ans = nums_forced[0]
            else:
                nums_full = extract_numbers_from_boxed_after_final_answer(text_full)
                ans = nums_full[0] if nums_full else None

        if gt is not None and ans is not None and ans == gt: correct += 1
        total += 1

        part_logs.append({
            "index": int(idx),
            "gt": gt,
            "pred": ans,
            "threshold": float(stop_threshold),
            "group_window": args.group_window,
            "online_trace": {
                "text": text, "text_full": text_full,
                "conf_hist": conf_hist, "lowest_gc": lowest_gc, "tokens_used": tokens_used
            }
        })

    with open(f"{args.save_pred}.online.jsonl","w",encoding="utf-8") as f:
        for r in part_logs: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    acc = correct/max(1,total)
    print(f"[GSM8K] Accuracy: {acc:.4f} ({correct}/{total})")
    with open(f"{args.save_pred}.online.stats.json","w",encoding="utf-8") as f:
        json.dump({"accuracy": acc, "correct": correct, "total": total}, f, ensure_ascii=False, indent=2)

def main():
    ap = build_parser(); args = ap.parse_args()
    tokenizer, model = main_common_before(args)
    s = warmup_phase(args, tokenizer, model)
    online_phase(args, tokenizer, model, stop_threshold=s)

if __name__ == "__main__":
    main()
