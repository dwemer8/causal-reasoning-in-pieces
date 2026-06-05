import csv
import datetime
from pathlib import Path
from typing import Optional, Any


# Known pipeline stage class names for per-stage token usage flattening.
# These map stage class names to shorter column prefixes.
_STAGE_COLUMN_PREFIXES: dict[str, str] = {
    "UndirectedSkeletonStage": "undirected_skeleton",
    "VStructuresStage": "v_structures",
    "MeekRulesStage": "meek_rules",
    "HypothesisEvaluationStage": "hypothesis_evaluation",
}


def _flatten_token_usage(record: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten the nested ``token_usage`` dict into scalar columns.

    Transforms::

        {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "per_stage": {
                "UndirectedSkeletonStage": {"input_tokens": 60, "output_tokens": 30, "total_tokens": 90},
                ...
            }
        }

    into flat columns like ``input_tokens``, ``output_tokens``, ``total_tokens``,
    ``undirected_skeleton.input_tokens``, ``undirected_skeleton.output_tokens``,
    ``undirected_skeleton.total_tokens``, etc.
    """
    tu = record.pop("token_usage", None)
    if not tu or not isinstance(tu, dict):
        return record

    # Top-level totals
    record["input_tokens"] = tu.get("input_tokens", 0)
    record["output_tokens"] = tu.get("output_tokens", 0)
    record["total_tokens"] = tu.get("total_tokens", 0)

    # Per-stage breakdown
    per_stage = tu.get("per_stage", {})
    if isinstance(per_stage, dict):
        for stage_name, stage_tokens in per_stage.items():
            prefix = _STAGE_COLUMN_PREFIXES.get(stage_name, stage_name.lower())
            if isinstance(stage_tokens, dict):
                record[f"{prefix}.input_tokens"] = stage_tokens.get("input_tokens", 0)
                record[f"{prefix}.output_tokens"] = stage_tokens.get("output_tokens", 0)
                record[f"{prefix}.total_tokens"] = stage_tokens.get("total_tokens", 0)

    return record


_HISTORY_KEY_TO_COLUMN: dict[str, str] = {
    "undirected_skeleton_history": "undirected_skeleton.reasoning",
    "v_structures_history": "v_structures.reasoning",
    "meek_rules_history": "meek_rules.reasoning",
    "hypothesis_evaluation_history": "hypothesis_evaluation.reasoning",
}


def _flatten_reasoning(record: dict[str, Any]) -> dict[str, Any]:
    """
    Extract per-stage reasoning from history dicts into flat CSV columns.

    Each history key is replaced by a ``{prefix}.reasoning`` column containing
    only the ``reasoning`` text from that stage's history dict.
    """
    for history_key, col_name in _HISTORY_KEY_TO_COLUMN.items():
        history = record.pop(history_key, None)
        if isinstance(history, dict):
            record[col_name] = history.get("reasoning")
        else:
            record[col_name] = None
    return record


class ExperimentLogger:
    """
    Create logging CSV file up‑front and append rows to it as each experiment finishes.
    """
    def __init__(self, logs_dir: Path, job_id: Optional[str] = None,
                 log_reasoning: bool = False) -> None:
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        job_suffix = f"_{job_id}" if job_id else ""
        filename = f"experiment_results_{timestamp}{job_suffix}.csv"
        self.log_file = self.logs_dir / filename

        self._fieldnames: Optional[list[str]] = None
        self.log_reasoning = log_reasoning

    def _init_header(self, fieldnames: list[str]) -> None:
        with self.log_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        self._fieldnames = fieldnames

    def _coerce(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return a copy with hypothesis_label formatted to int and token_usage flattened."""
        if "hypothesis_label" in record:
            label = record["hypothesis_label"]
            if label is not None:
                label = int(label)
            record = {**record, "hypothesis_label": label}
        # Flatten nested token_usage into scalar columns before CSV serialization
        record = _flatten_token_usage(record)
        # Flatten per-stage reasoning history when enabled
        if self.log_reasoning:
            record = _flatten_reasoning(record)
        return record

    def append(self, record: dict[str, Any]) -> None:
        record = self._coerce(record)
        if self._fieldnames is None:
            self._init_header(list(record.keys()))

        with self.log_file.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames).writerow(record)

    def append_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        records = [self._coerce(r) for r in records]
        if self._fieldnames is None:
            self._init_header(list(records[0].keys()))

        with self.log_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerows(records)
