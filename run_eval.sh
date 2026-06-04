#!/bin/bash
set -e

MODEL="deepseek-ai/DeepSeek-V4-Pro"
API_BASE="https://llm-chat.sk.appliedai.ru/api"
INPUT_FILE="data/test_dataset.csv"
BATCH_SIZE=32
VENV_PYTHON="/home/d.kornilov/work/causal_discovery/agents/causal-reasoning-in-pieces/.venv/bin/python"

echo ""
echo "============================================================"
echo "TEST: --indexes 2 4 --num_experiments 3"
echo "============================================================"
PYTHONPATH="." $VENV_PYTHON causal_discovery/main.py \
  --backend openai \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --input_file "$INPUT_FILE" \
  --mode batched \
  --batch_size "$BATCH_SIZE" \
  --num_experiments 3 \
  --temperature 0.1 \
  # --thinking \
  # --reasoning_effort high \
  --indexes 2 4
echo "=== Done test ==="

echo ""
echo "All evaluations completed."