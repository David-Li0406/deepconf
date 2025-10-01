#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-process CSQA with *dataset-level* DeepConf threshold (no vLLM)
- Phase 1 (warm-up): randomly pick n questions (default 50) and do N_init traces EACH (no early stop)
  * Aggregate ALL lowest group confidences from these n * N_init traces
  * Compute a single *dataset-level* threshold s from their percentile (controlled by eta)
  * Save the warm-up question ids + their N_init traces to {save_pred}.warmup.jsonl
- Phase 2 (online): for ALL questions, run exactly ONE pseudo-online generation
  * Apply early stop using the dataset-level threshold s (no consensus, no multi-trace)
  * Save per-question result to {save_pred}

Run example (4 GPUs):
  python qwen3_online_csqa_mp_dataset_threshold.py \
    --model_name "/scratch/daweili5/cot-valve/saves/Qwen3-8B" \
    --mode deepconf-low --N_init 8 --warmup_n 50 \
    --num_workers 4 --device_ids 0,1,2,3 \
    --max_new_tokens 1024 --save_pred csqa_dc_dataset_thr.jsonl

Notes:
  - This script implements the three requested changes:
    (1) Only n random questions are used for N_init warm-up; threshold s is dataset-level.
    (2) Online stage runs exactly ONE early-stopped trace per question.
    (3) Persist the warm-up n questions and their N_init traces.
"""

import os
import re
import json
import math
import random
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================== Args ==============================

def build_parser():
    ap = argparse.ArgumentParser("CSQA (MP) with dataset-level DeepConf threshold")
    # model/data
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--seed", type=int, default=42)

    # sampling
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=1024)

    # DeepConf core
    ap.add_argument("--N_init", type=int, default=8, help="warm-up traces per sampled question")
    ap.add_argument("--eta", type=float, default=10.0, choices=[10.0, 90.0],
                    help="keep ratio η: s = percentile(100-η) over lowest group confidences")
    ap.add_argument("--mode", type=str, default="deepconf-low",
                    choices=["deepconf-low","deepconf-high"],
                    help="shortcut: deepconf-low sets η=10, deepconf-high sets η=90")
    ap.add_argument("--group_window", type=int, default=256)
    ap.add_argument("--topk_conf", type=int, default=5)

    # run control
    ap.add_argument("--max_questions", type=int, default=-1, help="-1 = all")
    ap.add_argument("--save_pred", type=str, default="deepconf_csqa_dataset_thr.jsonl")

    # warm-up sampling (dataset-level)
    ap.add_argument("--warmup_n", type=int, default=50, help="number of questions for dataset-level warm-up")

    # multiprocessing
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--device_ids", type=str, default="",
                    help="comma-separated GPU ids, e.g. '0,1,2,3'; empty uses range(num_workers)")
    return ap

# ====================== Prompting & extraction =====================

def build_messages(question, choices):
    labels = choices["label"]
    texts = choices["text"]
    lines = [f"{l}. {t}" for l, t in zip(labels, texts)]
    user = (
        f"{question}\n\nChoices:\n" + "\n".join(lines) +
        "\n\nPlease think step by step and answer. End with: Final Answer: \\\\boxed{A/B/C/D/E}."
    )
    return [{"role": "user", "content": user}]

# Robust extractors
EXTRACT_RE_MAIN = re.compile(
    r"(?:final\s*answer|answer)\s*:\s*\\+\s*boxed\s*\{\s*([A-E])\s*\}", re.IGNORECASE)
EXTRACT_RE_BOXED_ONLY = re.compile(r"\\+\s*boxed\s*\{\s*([A-E])\s*\}", re.IGNORECASE)
EXTRACT_RE_TEXTUAL = re.compile(
    r"(?:the\s+answer\s+is|answer\s*is|therefore[,\s]*the\s+answer\s+is)\s*([A-E])\b",
    re.IGNORECASE,
)
EXTRACT_RE_TAIL = re.compile(r"(?:final|answer|ans|option)\D*([A-E])\s*$", re.IGNORECASE | re.MULTILINE)

def extract_choice(text: str):
    if not text:
        return None
    t = text.replace("\u200b", " ").replace("\u00a0", " ")
    for rx in (EXTRACT_RE_MAIN, EXTRACT_RE_BOXED_ONLY, EXTRACT_RE_TEXTUAL, EXTRACT_RE_TAIL):
        m = rx.search(t)
        if m:
            return m.group(1).upper()
    return None

# ========================= Confidence utils ========================

def token_confidence_from_logits(logits: torch.Tensor, k: int, temperature: float) -> float:
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)
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

# ===================== Force-finalize helper (ONLINE ONLY) =====================
def _build_force_prompt_from_text(text: str):
    """Ensure thinking is closed and append 'Final Answer:' to force a choice."""
    base = (text or "").rstrip()
    if "</think>" not in base:
        base = base + "\n</think>"
    return base + "\nFinal Answer: "

def force_finalize_answer(model, tokenizer, text, device, args, max_new_tokens: int = 16):
    """Re-feed current text with a 'Final Answer:' cue to force an explicit choice."""
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
        ans = extract_choice(full)
        return ans

# ===================== Pseudo-online generation ====================

def generate_one_trace_online(model, tokenizer, prompt_ids, device, args, stop_threshold=None):
    """Generate full sequence once, then replay early-stop to truncate.
    Returns: text, final_choice, conf_hist (truncated), lowest_gc, tokens_used
    """
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
        final_choice = extract_choice(text)
        text_full = tokenizer.batch_decode(full_gen_ids, skip_special_tokens=True)[0]
        final_choice_full = extract_choice(text_full)
        lowest_gc = lowest_group_conf(conf_hist[:cut_idx], args.group_window)
        tokens_used = int(cut_idx)
        return text, text_full, final_choice, final_choice_full, conf_hist[:cut_idx], lowest_gc, tokens_used

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

    ds = load_dataset("tau/commonsense_qa", split=args.split)

    warm_logs = []
    warm_lowests_all = []  # collect all lowest_gc across N_init traces for percentile

    for g_idx in warm_indices:
        item = ds[int(g_idx)]
        question = item["question"]
        choices = item["choices"]

        messages = build_messages(question, choices)
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
            # warm-up 阶段不做“强制收尾”，保持原逻辑
            traces.append({
                "text": text, "ans": ans, "conf_hist": conf_hist,
                "lowest_gc": lowest_gc, "tokens_used": tokens_used
            })
            warm_lowests_all.append(lowest_gc)

        warm_logs.append({
            "index": int(g_idx),
            "warm_traces": traces,
        })

    part_tag = f"W{part_name:02d}"
    part_path = f"{args.save_pred}.warm.part{part_tag}.jsonl"
    with open(part_path, "w", encoding="utf-8") as f:
        for r in warm_logs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stat_path = f"{args.save_pred}.warm.lowests.part{part_tag}.jsonl"
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"lowests": warm_lowests_all}, ensure_ascii=False) + "\n")


def worker_online(args, indices, part_name, device_id, stop_threshold):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    m = re.search(r'\d+', str(part_name))
    part_idx = int(m.group()) if m else 0
    torch.manual_seed(args.seed + 1024 + part_idx)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": device_id} if torch.cuda.is_available() else "auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    device = model.device

    ds = load_dataset("tau/commonsense_qa", split=args.split)

    part_logs = []
    for g_idx in tqdm(indices):
        item = ds[int(g_idx)]
        question = item["question"]
        choices = item["choices"]
        gt = str(item["answerKey"]).strip().upper()

        messages = build_messages(question, choices)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompt_ids = tokenizer([prompt_text], return_tensors="pt").to(device).input_ids

        # Exactly ONE pseudo-online trace with early stop using dataset-level s
        text, text_full, ans, ans_full, conf_hist, lowest_gc, tokens_used = generate_one_trace_online(
            model, tokenizer, prompt_ids, device, args, stop_threshold=stop_threshold
        )

        # 当截断文本未抽到答案时，先做“强制收尾”，再回退到 ans_full / "A"
        forced_used = False
        if ans is None:
            forced = force_finalize_answer(model, tokenizer, text, device, args)
            if forced is not None:
                pred = forced
                forced_used = True
            else:
                pred = ans_full if ans_full is not None else "A"
        else:
            pred = ans

        part_logs.append({
            "index": int(g_idx),
            "gt": gt,
            "pred": pred,
            "forced_pred_used": forced_used,
            "threshold": float(stop_threshold),
            "group_window": args.group_window,
            "online_trace": {
                "text": text, "text_full":text_full, "ans": ans, "ans_full": ans_full, "conf_hist": conf_hist,
                "lowest_gc": lowest_gc, "tokens_used": tokens_used
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

    # devices
    if args.device_ids.strip():
        devices = [int(x) for x in args.device_ids.split(",") if x.strip()]
    else:
        devices = list(range(max(1, args.num_workers)))
    n_workers = max(1, args.num_workers)
    assert len(devices) >= n_workers, "device_ids fewer than num_workers"

    # dataset meta
    ds = load_dataset("tau/commonsense_qa", split=args.split)
    n_total_all = len(ds)
    n_total = n_total_all if args.max_questions < 0 else min(n_total_all, args.max_questions)
    all_indices = list(range(n_total))

    # ------------------ PHASE 1: dataset-level warm-up ------------------
    random.seed(args.seed)
    warmup_n = min(args.warmup_n, n_total)
    warm_indices = random.sample(all_indices, warmup_n)

    # split warm_indices
    warm_parts = split_indices(len(warm_indices), n_workers)

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # launch warm-up workers
    procs = []
    for i, idxs_local in enumerate(warm_parts):
        # map local positions back to global ids
        shard_global = [warm_indices[j] for j in idxs_local]
        p = mp.Process(target=worker_warmup, args=(args, shard_global, i, devices[i]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # merge warm-up logs and collect all lowest_gc
    warm_merge = 0
    warm_lowests_all = []
    with open(f"{args.save_pred}.warmup.jsonl", "w", encoding="utf-8") as fout:
        # merge question-level warm traces
        for i in range(len(warm_parts)):
            pf = f"{args.save_pred}.warm.partW{i:02d}.jsonl"
            if os.path.exists(pf):
                with open(pf, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        warm_merge += 1
        # collect lowests from aux files
        for i in range(len(warm_parts)):
            sf = f"{args.save_pred}.warm.lowests.partW{i:02d}.jsonl"
            if os.path.exists(sf):
                with open(sf, "r", encoding="utf-8") as fin:
                    for line in fin:
                        try:
                            obj = json.loads(line)
                            warm_lowests_all.extend(obj.get("lowests", []))
                        except Exception:
                            pass
    print(f"[WARM MERGE] Saved {warm_merge} warm-up questions to {args.save_pred}.warmup.jsonl")

    # compute dataset-level threshold s
    s = percentile(warm_lowests_all, 100.0 - args.eta)
    print(f"[DATASET THRESHOLD] eta={args.eta} → s={s:.6f} (from {len(warm_lowests_all)} traces)")

    # ------------------ PHASE 2: one-trace online for ALL ------------------
    parts = split_indices(n_total, n_workers)

    procs = []
    for i, idxs_local in enumerate(parts):
        shard_global = [all_indices[j] for j in idxs_local]
        p = mp.Process(target=worker_online, args=(args, shard_global, f"O{i:02d}", devices[i], s))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # merge online outputs and compute accuracy
    merged = 0
    all_correct = 0
    all_total = 0
    with open(args.save_pred, "w", encoding="utf-8") as fout:
        for i in range(len(parts)):
            pf = f"{args.save_pred}.online.partO{i:02d}.jsonl"
            if not os.path.exists(pf):
                continue
            with open(pf, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    merged += 1
                    try:
                        row = json.loads(line)
                        gt = row.get("gt")
                        pred = row.get("pred")
                        if gt is not None and pred is not None:
                            all_total += 1
                            if str(gt).strip().upper() == str(pred).strip().upper():
                                all_correct += 1
                    except Exception:
                        pass
    print(f"[ONLINE MERGE] Wrote {merged} lines to {args.save_pred}")
    if all_total > 0:
        acc = all_correct / all_total * 100
        print(f"[RESULT] Accuracy = {all_correct}/{all_total} = {acc:.2f}%")


if __name__ == "__main__":
    main()
