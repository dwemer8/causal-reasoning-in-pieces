import argparse
import logging
import time
from pathlib import Path
import datetime

import pandas as pd
from tqdm import tqdm

from experiment_logger import ExperimentLogger
from utils import extract_premise, extract_hypothesis
from pipeline.pipeline import CausalDiscoveryPipeline, BatchCasualDiscoveryPipeline
from pipeline.stages import UndirectedSkeletonStage, VStructuresStage, MeekRulesStage, HypothesisEvaluationStage
from llm_client import OpenAIClient, BaseLLMClient, HuggingFaceClient, DeepSeekClient

LOGS_DIR: Path = Path("causal_discovery/logs")
BENCHMARKS_FILE: Path = LOGS_DIR / "benchmarks.tsv"
DEFAULT_TIMEOUT = 1800

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Causal Discovery Pipeline with configurable backend and mode."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the split csv file",
        default="data_peturbations/test_dataset_variable_refactorization.csv"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "huggingface", "deepseek"],
        default="openai",
        help="Choose the LLM backend.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to use (overrides backend default).",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Custom API base URL (e.g., https://llm-chat.sk.appliedai.ru/api).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "batched"],
        default="batched",
        help="Run pipeline in sequential or batched mode.",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1200,
        help="Number of experiments to run. If greater than dataset length, the whole test set will be used.",
    )
    parser.add_argument(
        "--indexes",
        type=int,
        nargs=2,
        metavar=("MIN_IDX", "MAX_IDX"),
        default=None,
        help="Evaluate only rows in range [MIN_IDX, MAX_IDX] (both inclusive, positional iloc indexing).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for batch processing.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the LLM. If not set, uses backend default.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter for the LLM. If not set, uses backend default.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter for the LLM. If not set, uses backend default.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=None,
        help="Min-p sampling parameter for the LLM. If not set, uses backend default.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=None,
        help="Presence penalty for the LLM. If not set, uses backend default.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty for the LLM. If not set, uses backend default.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["high", "max"],
        help="Reasoning effort level for thinking mode (high or max).",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode via chat_template_kwargs.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens for completion. None uses model default.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds (default: DEFAULT_TIMEOUT = 30 min).",
    )
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    """Load the dataset CSV and optionally filter by positional index range.

    Args:
        args: Parsed command-line arguments. Uses args.input_file and args.indexes.

    Returns:
        pd.DataFrame with the loaded (and optionally filtered) data.
    """
    csv_file = args.input_file
    df = pd.read_csv(csv_file)
    if args.indexes is not None:
        min_idx, max_idx = args.indexes
        df = df.iloc[min_idx:max_idx + 1]
        logging.info(f"Filtered to rows [{min_idx}, {max_idx}] — {len(df)} rows remaining.")
    logging.info(f"Loaded dataset from {csv_file} with {len(df)} rows.")
    return df


def prepare_input_samples(df: pd.DataFrame, num_experiments: int) -> list[dict]:
    num_experiments = min(num_experiments, len(df))
    # sampled_df = df.sample(n=num_experiments, replace=False) #when using min_idx and max_idx we want linear execution
    sampled_df = df.iloc[0:num_experiments]

    input_samples = []
    for idx, row in sampled_df.iterrows():
        input_text = row["input"]
        premise = extract_premise(input_text)
        hypothesis = extract_hypothesis(input_text)
        sample = {
            "sample_id": idx,
            "sample_input": input_text,
            "sample_label": row["label"],
            "sample_num_variables": row["num_variables"],
            "sample_template": row["template"],
            "premise": premise,
            "hypothesis": hypothesis
        }
        input_samples.append(sample)
    logging.info(f"Prepared {len(input_samples)} input samples for the pipeline.")
    return input_samples


def create_client(backend: str, batch_size: int, model: str, api_base: str | None = None,
                  temperature: float | None = None, top_p: float | None = None,
                  top_k: int | None = None, min_p: float | None = None,
                  presence_penalty: float | None = None,
                  repetition_penalty: float | None = None,
                  reasoning_effort: str | None = None,
                  thinking: bool = False,
                  max_tokens: int | None = None,
                  timeout: float = DEFAULT_TIMEOUT) -> BaseLLMClient:
    if backend == "openai":
        client = OpenAIClient(
            model_id=model, 
            concurrency=batch_size, 
            base_url=api_base, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            min_p=min_p, 
            presence_penalty=presence_penalty, 
            repetition_penalty=repetition_penalty, 
            reasoning_effort=reasoning_effort, 
            thinking=thinking, 
            max_tokens=max_tokens, 
        timeout=timeout)
    elif backend == "huggingface":
        client = HuggingFaceClient(max_new_tokens=8192, batch_size=batch_size, model_id=model, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, presence_penalty=presence_penalty, repetition_penalty=repetition_penalty)
    else:
        client = DeepSeekClient(concurrency=batch_size, model_id=model, base_url=api_base or "https://api.deepseek.com", temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, presence_penalty=presence_penalty, repetition_penalty=repetition_penalty)
    logging.info(f"Using {backend} backend for the pipeline.")
    return client


def post_process_logs(
    log_file: str, 
    model: str, 
    temperature: float, 
    top_p: float,
    top_k: int = None, 
    min_p: float = None,
    presence_penalty: float = None,
    repetition_penalty: float = None,
    reasoning_effort: str = None,
    thinking: bool = False,
    max_tokens: int = None,
    timeout: float = DEFAULT_TIMEOUT
) -> None:
    """
    Read the log CSV file, compute confusion matrix and performance metrics,
    then print them out and append a row to the benchmarks TSV file.
    """
    df = pd.read_csv(log_file)
    n_nulls = df["hypothesis_label"].isnull().sum()
    if n_nulls > 0:
        logging.warning(f"Found {n_nulls} null values in the hypothesis label column. Dropping them.")
        df = df.dropna(subset=["hypothesis_label"])
    df["hypothesis_label"] = df["hypothesis_label"].astype(int)
    df["sample_label"] = df["sample_label"].astype(int)

    tp = ((df["hypothesis_label"] == 1) & (df["sample_label"] == 1)).sum()
    tn = ((df["hypothesis_label"] == 0) & (df["sample_label"] == 0)).sum()
    fp = ((df["hypothesis_label"] == 1) & (df["sample_label"] == 0)).sum()
    fn = ((df["hypothesis_label"] == 0) & (df["sample_label"] == 1)).sum()

    # Calculate metrics.
    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Confusion Matrix ---")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("\n--- Performance Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Append benchmark row to TSV.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{
        "model": model,
        "timestamp": timestamp,
        "temperature": f"{temperature}",
        "top_p": f"{top_p}",
        "top_k": f"{top_k}",
        "min_p": f"{min_p}",
        "presence_penalty": f"{presence_penalty}",
        "repetition_penalty": f"{repetition_penalty}",
        "reasoning_effort": f"{reasoning_effort}",
        "thinking": f"{thinking}",
        "max_tokens": f"{max_tokens}",
        "timeout": f"{timeout}",
        "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1": f"{f1:.4f}",
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "n_nulls": n_nulls,
    }])
    BENCHMARKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if BENCHMARKS_FILE.exists():
        existing = pd.read_csv(BENCHMARKS_FILE, sep="\t")
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv(BENCHMARKS_FILE, sep="\t", index=False)
    logging.info(f"Benchmark results appended to {BENCHMARKS_FILE}")


def main() -> None:
    args = parse_arguments()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load dataset and prepare input samples.
    df = load_dataset(args)
    input_samples = prepare_input_samples(df, args.num_experiments)

    # Create the LLM client based on backend choice.
    client = create_client(
        args.backend, 
        args.batch_size, 
        args.model, 
        args.api_base, 
        args.temperature, 
        args.top_p, 
        args.top_k, 
        args.min_p, 
        args.presence_penalty, 
        args.repetition_penalty, 
        args.reasoning_effort, 
        args.thinking, 
        args.max_tokens, 
        args.timeout
    )
    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    # Prepare the pipeline
    skeleton_stage = UndirectedSkeletonStage(client=client)
    v_structures_stage = VStructuresStage(client=client)
    meek_rules_stage = MeekRulesStage(client=client)
    hypothesis_evaluation_stage = HypothesisEvaluationStage(client=client)

    job_id = Path(args.input_file).stem
    logger = ExperimentLogger(LOGS_DIR, job_id)
    pipeline: CausalDiscoveryPipeline = CausalDiscoveryPipeline(
        stages=[skeleton_stage, v_structures_stage, meek_rules_stage, hypothesis_evaluation_stage],
        logger=logger,
    )

    results = []
    failed_ids = []
    start_time = time.time()

    if args.mode == "batched":
        logging.info("Running pipeline in batched mode.")
        batch_pipeline = BatchCasualDiscoveryPipeline(pipeline=pipeline, batch_size=args.batch_size)
        results, failed_ids = batch_pipeline.run_batch(input_samples)
    else:
        logging.info("Running pipeline in sequential mode.")
        for sample in tqdm(input_samples, desc="Processing samples"):
            try:
                result = pipeline.run(sample)
                results.append(result)
            except Exception as e:
                failed_ids.append(sample["sample_id"])
                logging.error(f"Error processing sample {sample['sample_id']}: {e}")

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Run results post-processing.
    post_process_logs(
        str(logger.log_file), 
        args.model, 
        args.temperature, 
        args.top_p, 
        args.top_k, 
        args.min_p, 
        args.presence_penalty, 
        args.repetition_penalty, 
        args.reasoning_effort, 
        args.thinking, 
        args.max_tokens, 
        args.timeout
    )

    if failed_ids:
        logging.info(f"Total failed experiments after max retries: {len(failed_ids)}")
        logging.info(f"Failed sample IDs: {failed_ids}")


if __name__  == "__main__":
    main()