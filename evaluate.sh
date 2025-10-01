#!/usr/bin/env bash
#SBATCH -t 0-28:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH -c 5
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu

export PYTHONNOUSERSITE=1
export TORCHDYNAMO_DISABLE=1

# CommonsenseQA
python qwen3_online_csqa.py \
  --model_name "/scratch/daweili5/hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/" \
  --mode deepconf-low \
  --N_init 16 --warmup_n 50 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 2560 \
  --num_workers 2 --device_ids 0,1 \
  --save_pred csqa_dc_dataset_thr.jsonl

# GPQA Diamond
python qwen3_online_gpqa.py \
  --model_name "/scratch/daweili5/hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/" \
  --gpqa_path "/path/to/gpqa_diamond.json" \
  --mode deepconf-low \
  --N_init 16 --warmup_n 50 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 2560 \
  --num_workers 2 --device_ids 0,1 \
  --save_pred gpqa_dc_dataset_thr.jsonl

# GSM8K
python qwen3_online_gsm8k.py \
  --model_name "/scratch/daweili5/hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/" \
  --mode deepconf-low \
  --N_init 16 --warmup_n 50 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 2560 \
  --num_workers 2 --device_ids 0,1 \
  --save_pred gsm8k_dc_dataset_thr.jsonl

# MATH-500
python qwen3_online_math.py \
  --model_name "/scratch/daweili5/hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/" \
  --mode deepconf-low \
  --N_init 16 --warmup_n 50 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 4096 \
  --num_workers 2 --device_ids 0,1 \
  --save_pred math500_dc_dataset_thr.jsonl


# CommonsenseQA
python gpt_oss_online_csqa.py \
  --model_name "openai/gpt-oss-20b" \
  --N_init 16 --warmup_n 50 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 2560 \
  --save_pred csqa_dc_dataset_thr.gptoss.jsonl

# GPQA Diamond
python gpt_oss_online_gpqa.py \
  --model_name "openai/gpt-oss-20b" \
  --gpqa_path "/path/to/gpqa_diamond.json" \
  --N_init 16 --warmup_n 50 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 2560 \
  --save_pred gpqa_dc_dataset_thr.gptoss.jsonl

# GSM8K
python gpt_oss_online_gsm8k.py \
  --model_name "openai/gpt-oss-20b" \
  --N_init 16 --warmup_n 50 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 2560 \
  --save_pred gsm8k_dc_dataset_thr.gptoss.jsonl

# MATH-500
python gpt_oss_online_math.py \
  --model_name "openai/gpt-oss-20b" \
  --N_init 16 --warmup_n 50 \
  --group_window 256 --topk_conf 5 \
  --max_new_tokens 4096 \
  --save_pred math500_dc_dataset_thr.gptoss.jsonl
