#!/bin/bash
set -e

# Evaluate DeepSeek V4 Pro on full test dataset with thinking mode.
# Two runs: reasoning_effort=high and reasoning_effort=max.
# Results saved to causal_discovery/logs/benchmarks.tsv.

MODEL="deepseek-ai/DeepSeek-V4-Pro"
API_BASE="https://llm-chat.sk.appliedai.ru/api"
INPUT_FILE="data/test_dataset.csv"
BATCH_SIZE=32
VENV_PYTHON="/home/d.kornilov/work/causal_discovery/agents/causal-reasoning-in-pieces/.venv/bin/python"

echo ""
echo "============================================================"
echo "RUN 1: thinking=true, reasoning_effort=high, T=0.1"
echo "============================================================"
PYTHONPATH="." $VENV_PYTHON causal_discovery/main.py \
  --backend openai \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --input_file "$INPUT_FILE" \
  --mode batched \
  --batch_size "$BATCH_SIZE" \
  --num_experiments 1200 \
  --temperature 0.1 \
  --thinking \
  --reasoning_effort high
echo "=== Done high ==="

echo ""
echo "============================================================"
echo "RUN 2: thinking=true, reasoning_effort=max, T=0.1"
echo "============================================================"
PYTHONPATH="." $VENV_PYTHON causal_discovery/main.py \
  --backend openai \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --input_file "$INPUT_FILE" \
  --mode batched \
  --batch_size "$BATCH_SIZE" \
  --num_experiments 1200 \
  --temperature 0.1 \
  --thinking \
  --reasoning_effort max
echo "=== Done max ==="

echo ""
echo "All evaluations completed."