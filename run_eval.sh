#!/bin/bash
set -e

declare -A BATCH_SIZES=(
  ["openai/gpt-oss-120b"]=32
  ["deepseek-ai/DeepSeek-V4-Pro"]=32
  ["Qwen/Qwen3.5-397B-A17B-FP8"]=4
  ["Qwen/Qwen3.6-35B-A3B"]=32
)

# Per-model optimal inference parameters.
# Models without specific optimal params use existing defaults (temperature=1.0, top_p=1.0).
declare -A INFERENCE_PARAMS=(
  ["openai/gpt-oss-120b"]="--temperature 1.0 --top_p 1.0"
  ["deepseek-ai/DeepSeek-V4-Pro"]="--temperature 1.0 --top_p 1.0"
  ["Qwen/Qwen3.5-397B-A17B-FP8"]="--temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 0.0 --repetition_penalty 1.0"
  ["Qwen/Qwen3.6-35B-A3B"]="--temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 0.0 --repetition_penalty 1.0"
)

MODELS=(
  "openai/gpt-oss-120b"
  "deepseek-ai/DeepSeek-V4-Pro"
  "Qwen/Qwen3.5-397B-A17B-FP8"
  "Qwen/Qwen3.6-35B-A3B"
)

for model in "${MODELS[@]}"; do
  batch_size="${BATCH_SIZES[$model]}"
  params="${INFERENCE_PARAMS[$model]}"
  echo ""
  echo "=== Running $model (batch_size=$batch_size) ==="
  PYTHONPATH="." .venv/bin/python causal_discovery/main.py \
    --backend openai \
    --model "$model" \
    --api-base "https://llm-chat.sk.appliedai.ru/api" \
    --input_file "data/test_dataset.csv" \
    --mode batched \
    --batch_size "$batch_size" \
    --num_experiments 1 \
    $params
  echo "=== Done $model ==="
done

echo ""
echo "All models completed."