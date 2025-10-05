#!/usr/bin/env python3
# coding: utf-8

import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer

INPUT_JSONL = "qwen_csqa_dc_dataset_thr.jsonl"
MODEL_NAME = "Qwen/Qwen3-8B"  # 如果不可用请换成本地或其他 tokenizer 名称

# 放宽匹配用的正则（只在没有 boxed 时使用）
EXTRACT_RE_TEXTUAL = re.compile(
    r"(?:the\s+answer\s+is|answer\s*is|therefore[,\s]*the\s+answer\s+is)\s*([A-E])\b",
    re.IGNORECASE,
)
EXTRACT_RE_TAIL = re.compile(r"(?:final|answer|ans|option)\D*([A-E])\s*$", re.IGNORECASE | re.MULTILINE)

# boxed 匹配（兼容 \boxed{A}、\boxed{{A}} 等）
BOXED_RE = re.compile(r"\\boxed\s*\{\s*\{?\s*([A-Ea-e])\s*\}?\s*\}", re.DOTALL)


def extract_boxed_from_text(text):
    """返回所有 boxed 的 match 对象列表（可能为空）。"""
    if text is None:
        return []
    return list(BOXED_RE.finditer(str(text)))


def extract_pred_answer_from_text(text, allow_relaxed=False):
    """
    1. 优先寻找 Final Answer 后面的第一个 boxed（若 allow_relaxed=False）
    2. 然后寻找全文最后一个 boxed。
    3. 如果 allow_relaxed=True（表示在整个样本中找不到 boxed），则尝试 EXTRACT_RE_TEXTUAL 和 EXTRACT_RE_TAIL。
    返回 'A'/'B'/... 或 None。
    """
    if text is None:
        return None
    s = str(text)

    # 优先：Final Answer 后的 boxed
    last_final_answer_pos = s.rfind("Final Answer")
    if last_final_answer_pos != -1:
        substring_after = s[last_final_answer_pos:]
        m = BOXED_RE.search(substring_after)
        if m:
            return m.group(1).upper()

    # 回退：全文中找最后一个 boxed
    all_matches = BOXED_RE.findall(s)
    if all_matches:
        return all_matches[-1].upper()

    # 如果允许放宽匹配，试用给定的两个正则
    if allow_relaxed:
        m1 = EXTRACT_RE_TEXTUAL.search(s)
        if m1:
            return m1.group(1).upper()
        m2 = EXTRACT_RE_TAIL.search(s)
        if m2:
            return m2.group(1).upper()

    return None


def get_gold_label_from_item(item):
    """
    尝试多种字段名读取 gold 答案（A/B/...），找不到返回 None。
    """
    if not isinstance(item, dict):
        return None
    candidates = [
        "final_answer", "finalAnswer", "answer", "gold", "label", "gt", "correct_answer", "gold_answer"
    ]
    for k in candidates:
        if k in item and item[k] is not None:
            return str(item[k]).strip().upper()
    # 尝试 prompt 里提取
    if "prompt" in item and isinstance(item["prompt"], (list, tuple)) and len(item["prompt"]) > 0:
        try:
            p0 = item["prompt"][0]
            if isinstance(p0, dict) and "value" in p0:
                v = str(p0["value"])
                m = re.search(r"Answer\s*[:：]?\s*([A-Ea-e])", v)
                if m:
                    return m.group(1).upper()
        except Exception:
            pass
    return None


def substring_until_boxed(full_text):
    """
    在 full_text 中搜索第一个 boxed match，并返回从文本开头到该 match 结尾的子串（包含 boxed）。
    如果没有找到 boxed，返回 None。
    """
    if full_text is None:
        return None
    s = str(full_text)
    m = BOXED_RE.search(s)
    if not m:
        return None
    # 返回从开始到 match.end()（包含 boxed 本身 以及前面的所有内容）
    return s[: m.end()]


def evaluate(jsonl_path):
    # 加载 tokenizer（仅 tokenizer）
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print("Warning: tokenizer load failed:", e)
        tokenizer = None

    total = 0
    correct = 0
    token_sum = 0
    token_list = []
    token_list_correct = []
    token_list_incorrect = []
    missing_gold_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc="Processing lines"):
        total += 1
        try:
            item = json.loads(line)
        except Exception as e:
            print("JSON parse error on line (truncated):", line[:200])
            continue

        gold = get_gold_label_from_item(item)
        if gold is None:
            missing_gold_count += 1

        online_trace = item.get("online_trace", {}) or {}
        t_text = online_trace.get("text")
        t_text_full = online_trace.get("text_full")

        # 先检查 text 是否包含 boxed（优先）
        text_has_boxed = False
        if t_text:
            if BOXED_RE.search(str(t_text)):
                text_has_boxed = True

        use_text_for_eval = None
        used_relaxed_match = False

        if text_has_boxed:
            # 使用 online_trace.text（因为里面就有 boxed）
            use_text_for_eval = str(t_text)
            # 预测直接按 boxed 规则匹配（不放宽）
            pred = extract_pred_answer_from_text(use_text_for_eval, allow_relaxed=False)
        else:
            # text 没有 boxed，尝试在 text_full 中截取从开头到 boxed 的子串
            substr_from_full = substring_until_boxed(t_text_full)
            if substr_from_full is not None:
                # 使用截取的子串（从开头到 boxed 结尾）
                use_text_for_eval = substr_from_full
                pred = extract_pred_answer_from_text(use_text_for_eval, allow_relaxed=False)
            else:
                # text 和 text_full 都没有 boxed：放宽匹配规则。
                # 优先在 text 中用放宽规则找（如果 text 存在），否则在 text_full 中找（整个 text_full）
                used_relaxed_match = True
                use_text_for_eval = str(t_text_full)
                pred = extract_pred_answer_from_text(use_text_for_eval, allow_relaxed=True)


        # 统计 token（参考之前逻辑：如果包含 <think>，从 <think> 开始截断）
        text_for_tokenize = use_text_for_eval if use_text_for_eval is not None else ""
        think_pos = text_for_tokenize.find("<think>")
        if think_pos != -1:
            text_for_tokenize = text_for_tokenize[think_pos:]
        try:
            if tokenizer is not None:
                tok_ids = tokenizer(text_for_tokenize, add_special_tokens=False)["input_ids"]
                tok_cnt = len(tok_ids)
            else:
                # 回退：使用字符数近似
                tok_cnt = len(text_for_tokenize)
        except Exception:
            tok_cnt = len(text_for_tokenize)

        token_sum += tok_cnt
        token_list.append(tok_cnt)
        # if pred is None:
        #     print(use_text_for_eval)
        #     print(tok_cnt)
        # if tok_cnt < 100:
        #     print(tok_cnt)
        #     print(use_text_for_eval)

        # 判定正确与否（只有 pred 与 gold 都存在时才可能正确）
        is_corr = False
        if pred is not None and gold is not None:
            is_corr = pred.strip().upper() == gold.strip().upper()
        else:
            is_corr = False

        if is_corr:
            correct += 1
            token_list_correct.append(tok_cnt)
        else:
            token_list_incorrect.append(tok_cnt)

    evaluated = total
    acc = correct / evaluated if evaluated > 0 else 0.0
    avg_tokens = token_sum / evaluated if evaluated > 0 else 0.0
    avg_tokens_correct = (sum(token_list_correct) / len(token_list_correct)) if token_list_correct else 0.0
    avg_tokens_incorrect = (sum(token_list_incorrect) / len(token_list_incorrect)) if token_list_incorrect else 0.0

    print("======== Evaluation Summary ========")
    print(f"Total samples processed: {total}")
    # print(f"Missing gold labels: {missing_gold_count} (these were treated as incorrect)")
    # print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f} ({correct}/{evaluated})")
    print("")
    print(f"Total tokens (sum): {token_sum}")
    # print(f"Average tokens per sample: {avg_tokens:.2f}")
    # print(f"Average tokens (correct samples): {avg_tokens_correct:.2f}")
    # print(f"Average tokens (incorrect samples): {avg_tokens_incorrect:.2f}")
    print("====================================")


if __name__ == "__main__":
    evaluate(INPUT_JSONL)
