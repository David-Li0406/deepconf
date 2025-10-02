from transformers import AutoTokenizer
import json

# 加载 Qwen3 tokenizer
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 读取文本文件
with open("csqa_dc_dataset_thr_v2.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line.strip()) for line in f if line.strip()]

token_counts = []
for i, item in enumerate(lines):
    # 如果用了text_full
    if item["online_trace"]["ans"] is None and item["forced_pred_used"] == False and item["online_trace"]["ans_full"] is not None:
        text = item["online_trace"]["text_full"]
    # 如果用了early_stop后的text
    else:
        text = item["online_trace"]["text"]
    # 找到第一个 <think> 的位置（包含它）
    think_pos = text.find("<think>")
    if think_pos != -1:
        trimmed_text = text[think_pos:]  # 包含 <think>
    else:
        trimmed_text = text  # 如果没有 <think>，记为空串

    # Tokenize 截断后的文本
    tokens = tokenizer(trimmed_text, add_special_tokens=False)["input_ids"]
    token_counts.append(len(tokens))
    if i < 5:
        print(f"Line {i+1}: {len(tokens)} tokens after <think>")

# 可选：保存到文件
print(len(token_counts))
print(sum(token_counts))
# with open("token_counts_after_think.txt", "w", encoding="utf-8") as f:
    # for i, count in enumerate(token_counts):
        # f.write(f"Line {i+1}: {count} tokens after <think>\n")
