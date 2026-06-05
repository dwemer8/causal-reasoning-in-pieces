import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from causal_discovery.llm_client import BaseLLMClient
from causal_discovery.utils import load_prompts, extract_causal_skeleton_json, extract_v_structures_json, \
    extract_directed_edges_literal_format_json, extract_hypothesis_answer, extract_undirected_edges_literal_format_json

# Maps stage class name to sample dict key for storing stage history
_STAGE_HISTORY_KEYS: dict[str, str] = {
    "UndirectedSkeletonStage": "undirected_skeleton_history",
    "VStructuresStage": "v_structures_history",
    "MeekRulesStage": "meek_rules_history",
    "HypothesisEvaluationStage": "hypothesis_evaluation_history",
}

# Order of stages for formatting prior history
_STAGE_ORDER: list[str] = [
    "UndirectedSkeletonStage",
    "VStructuresStage",
    "MeekRulesStage",
    "HypothesisEvaluationStage",
]


class Stage(ABC):
    """
    Base class for all stages in the pipeline.
    Each subclass needs to implement the `prompt_template` attribute.
    """
    prompts: dict[str, str] = load_prompts()
    prompt_template: str = None

    def __init__(self, client: BaseLLMClient):
        self.client = client
        if self.prompt_template is None:
            raise ValueError("Subclasses must define a prompt_template.")

    @abstractmethod
    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process the single sample using the prompt template and stage-specific logic..
        """
        pass

    @abstractmethod
    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process a batch of samples using the prompt template and stage-specific logic.
        """
        pass

    @staticmethod
    def _format_edges(edges: set[tuple] | None) -> str:
        """
        Format a list of edges with line breaks for better readability in prompts.
        """
        if edges is None:
            return "[]"
        formatted = "[\n    "
        formatted += ",\n    ".join([str(edge) for edge in edges])
        formatted += "\n  ]"
        return formatted

    def _update_token_usage(self, sample: dict[str, Any], usage: dict) -> None:
        if usage is None:
            return
        # Accept both CompletionUsage objects and plain dicts
        input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else usage.prompt_tokens
        output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else usage.completion_tokens
        total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else usage.total_tokens

        if "token_usage" not in sample:
            sample["token_usage"] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "per_stage": {}
            }

        sample["token_usage"]["input_tokens"] += input_tokens
        sample["token_usage"]["output_tokens"] += output_tokens
        sample["token_usage"]["total_tokens"] += total_tokens

        stage_dict = sample["token_usage"]["per_stage"].setdefault(
            self.__class__.__name__,
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )
        stage_dict["input_tokens"] += input_tokens
        stage_dict["output_tokens"] += output_tokens
        stage_dict["total_tokens"] += total_tokens

        logging.info(
            f"[{self.__class__.__name__}]   total so far: {sample['token_usage']['per_stage'][self.__class__.__name__]}"
        )
        logging.info(f"Overall token usage: {sample['token_usage']}")

    def _build_stage_history(
        self,
        sample: dict[str, Any],
        reasoning: Optional[str],
        input_keys: dict[str, str],
        output_keys: dict[str, str],
    ) -> dict[str, Any]:
        """
        Build a stage history dict capturing this stage's input, reasoning, and output.

        Stored in the sample under the key from ``_STAGE_HISTORY_KEYS``.

        :param sample: The sample dict after this stage has processed it.
        :param reasoning: The raw reasoning text from the LLM, or ``None``.
        :param input_keys: Mapping of prompt template placeholder → sample dict key
            for the inputs this stage was given.
        :param output_keys: Mapping of output field name → sample dict key
            for the outputs this stage produced.
        :returns: The history dict that was stored.
        """
        if not getattr(self, 'pass_reasoning', False):
            return {}

        stage_name = self.__class__.__name__
        history_key = _STAGE_HISTORY_KEYS.get(stage_name)
        if history_key is None:
            return {}

        history: dict[str, Any] = {
            "stage": stage_name,
            "input": {},
            "reasoning": reasoning,
            "output": {},
        }

        # Capture inputs
        for label, key in input_keys.items():
            history["input"][label] = sample.get(key)

        # Capture outputs
        for label, key in output_keys.items():
            history["output"][label] = sample.get(key)

        sample[history_key] = history
        return history

    def _format_prior_history(self, sample: dict[str, Any]) -> str:
        """
        Collect and format all prior stages' history as a text block.

        Returns an empty string if ``pass_reasoning`` is disabled or no prior
        history exists.

        :param sample: The sample dict with accumulated stage histories.
        :returns: A formatted multi-line string ready to prepend to a prompt,
            or an empty string.
        """
        if not getattr(self, 'pass_reasoning', False):
            return ""

        current_stage = self.__class__.__name__
        sections: list[str] = []

        for stage_name in _STAGE_ORDER:
            if stage_name == current_stage:
                break
            history_key = _STAGE_HISTORY_KEYS.get(stage_name)
            if history_key is None:
                continue
            history = sample.get(history_key)
            if history is None:
                continue

            sections.append(f"### Stage: {stage_name}")

            # Input section
            if history.get("input"):
                sections.append("**Input:**")
                for label, value in history["input"].items():
                    sections.append(f"- {label}: {value}")

            # Reasoning section
            reasoning = history.get("reasoning")
            if reasoning:
                sections.append(f"\n**Reasoning:**\n{reasoning}")
            else:
                sections.append("\n**Reasoning:** No reasoning available.")

            # Output section
            if history.get("output"):
                sections.append("\n**Output:**")
                for label, value in history["output"].items():
                    sections.append(f"- {label}: {value}")

        if not sections:
            return ""

        return "**Previous stages history:**\n\n" + "\n\n".join(sections)


class UndirectedSkeletonStage(Stage):
    """
    Stage for generating the undirected skeleton of the causal graph.
    """
    prompt_template = Stage.prompts["undirected_skeleton"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        if "premise" not in input_data:
            raise ValueError("Input data must contain Premise.")

        # 2. Build prompt
        prompt = self.prompt_template.format(premise=input_data["premise"])

        # 3. Send request to LLM
        logging.info("UndirectedSkeletonStage: Sending prompt to LLM.")
        response, reasoning, usage = self.client.complete(prompt=prompt)

        # 4. Unpack responses and update token usage
        self._update_token_usage(input_data, usage)
        try:
            skeleton = extract_causal_skeleton_json(answer=response)
            input_data["nodes"] = skeleton["nodes"]
            input_data["undirected_edges"] = skeleton["edges"]
        except Exception as e:
            logging.error("Error extracting skeleton: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["nodes"] = None
            input_data["undirected_edges"] = None

        self._build_stage_history(
            input_data,
            reasoning=reasoning,
            input_keys={"premise": "premise"},
            output_keys={"nodes": "nodes", "undirected_edges": "undirected_edges"},
        )
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("UndirectedSkeletonStage: Processing batch with %d samples.", len(inputs))
        # 1. Validate inputs
        for i, input_data in enumerate(inputs):
            if "premise" not in input_data:
                logging.error("Sample %d is missing 'premise' key.", i)
                raise ValueError(f"Sample {i} must contain 'premise'.")
            else:
                logging.debug("Sample %d contains 'premise'.", i)

        # 2. Build prompts
        prompts = []
        for i, input_data in enumerate(inputs):
            try:
                prompt = self.prompt_template.format(premise=input_data["premise"])
                prompts.append(prompt)
                logging.debug("Constructed prompt for sample %d: %s", i, prompt)
            except Exception as e:
                logging.error("Error constructing prompt for sample %d: %s", i, e)
                raise

        logging.debug("All prompts constructed: %s", prompts)

        # 3. Send batch
        try:
            responses = self.client.complete_batch(prompts=prompts)
            logging.info("Batch call returned %d responses.", len(responses))
        except Exception as e:
            logging.error("Batch call failed: %s", e)
            raise

        # 4. Unpack responses into texts and usages, and update token usage
        for i, ((text, reasoning, usage), item) in enumerate(zip(responses, inputs)):
            logging.debug("Raw response text for sample %d: %s", i, text)
            logging.debug("Token usage for sample %d: %s", i, usage)
            self._update_token_usage(item, usage)

        # 5. Parse skeleton from each response text
        for i, ((text, reasoning, _), item) in enumerate(zip(responses, inputs)):
            try:
                skeleton = extract_causal_skeleton_json(answer=text)
                item["nodes"] = skeleton["nodes"]
                item["undirected_edges"] = skeleton["edges"]
                logging.debug("Extracted skeleton for sample %d: nodes: %s, edges: %s",
                              i, skeleton["nodes"], skeleton["edges"])
            except Exception as e:
                logging.error("Error extracting skeleton for sample %d: %s", i, e)
                logging.debug("Problematic response for sample %d: %s", i, text)
                item["nodes"] = None
                item["undirected_edges"] = None

            self._build_stage_history(
                item,
                reasoning=reasoning,
                input_keys={"premise": "premise"},
                output_keys={"nodes": "nodes", "undirected_edges": "undirected_edges"},
            )

        return inputs


class VStructuresStage(Stage):
    """
    Stage for generating the V-structures out of the causal graph and Premise.
    """
    prompt_template = Stage.prompts["v_structures"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Input data must contain: {', '.join(required_keys)}.")

        # 2. Build prompt
        prior_history = self._format_prior_history(input_data)
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            edges=self._format_edges(input_data["undirected_edges"]),
            prior_history=prior_history,
        )

        # 3. Send request to LLM
        logging.info("VStructuresStage: Sending prompt to LLM.")
        response, reasoning, usage = self.client.complete(prompt=prompt)

        # 4. Unpack responses and update token usage
        self._update_token_usage(input_data, usage)
        try:
            v_structures = extract_v_structures_json(answer=response)
            input_data["v_structures"] = v_structures
        except Exception as e:
            logging.error("Error extracting V-structures: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["v_structures"] = None

        self._build_stage_history(
            input_data,
            reasoning=reasoning,
            input_keys={"premise": "premise", "skeleton": "undirected_edges"},
            output_keys={"v_structures": "v_structures"},
        )
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("VStructuresStage: Processing batch with %d samples.", len(inputs))

        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges"}
        for i, input_data in enumerate(inputs):
            if not required_keys.issubset(input_data):
                missing = required_keys - input_data.keys()
                logging.error("Sample %d is missing keys: %s", i, missing)
                raise ValueError(f"Sample {i} must contain: {', '.join(required_keys)}.")

        # 2. Split inputs into valid and already-failed
        valid_indices = []
        failed_indices = []
        for i, input_data in enumerate(inputs):
            if input_data["nodes"] is None or input_data["undirected_edges"] is None:
                logging.warning("Sample %d skipping VStructuresStage: prior stage failed.", i)
                input_data["v_structures"] = None
                failed_indices.append(i)
            else:
                valid_indices.append(i)

        if not valid_indices:
            return inputs

        # 3. Build prompts for valid samples only
        prompts = []
        for i in valid_indices:
            input_data = inputs[i]
            prior_history = self._format_prior_history(input_data)
            prompt = self.prompt_template.format(
                premise=input_data["premise"],
                nodes=input_data["nodes"],
                edges=self._format_edges(input_data["undirected_edges"]),
                prior_history=prior_history,
            )
            prompts.append(prompt)
            logging.debug("Constructed prompt for sample %d: %s", i, prompt)

        logging.debug("All prompts constructed: %s", prompts)

        # 4. Send batch (valid samples only)
        try:
            responses = self.client.complete_batch(prompts=prompts)
            logging.info("Batch call returned %d responses.", len(responses))
        except Exception as e:
            logging.error("Batch call failed: %s", e)
            raise

        # 5. Unpack responses and update token usage for valid samples
        for j, i in enumerate(valid_indices):
            text, reasoning, usage = responses[j]
            logging.debug("Raw response text for sample %d: %s", i, text)
            logging.debug("Token usage for sample %d: %s", i, usage)
            self._update_token_usage(inputs[i], usage)

        # 6. Parse v-structures from each valid response
        for j, i in enumerate(valid_indices):
            text, reasoning, _ = responses[j]
            try:
                v_structures = extract_v_structures_json(answer=text)
                inputs[i]["v_structures"] = v_structures
                logging.debug("Extracted V-structures for sample %d: %s", i, v_structures)
            except Exception as e:
                logging.error("Error extracting V-structures for sample %d: %s", i, e)
                logging.debug("Problematic response for sample %d: %s", i, text)
                inputs[i]["v_structures"] = None

            self._build_stage_history(
                inputs[i],
                reasoning=reasoning,
                input_keys={"premise": "premise", "skeleton": "undirected_edges"},
                output_keys={"v_structures": "v_structures"},
            )

        return inputs

class MeekRulesStage(Stage):
    """
    Stage for applying Meek's rules to the V-structures.
    """
    prompt_template = Stage.prompts["meek_rules"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges", "v_structures"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Meek rules stage input data must contain: {', '.join(required_keys)}.")

        # 2. Build prompt
        prior_history = self._format_prior_history(input_data)
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            edges=self._format_edges(input_data["undirected_edges"]),
            v_structures=input_data["v_structures"],
            prior_history=prior_history,
        )

        # 3. Send request to LLM
        logging.info("MeekRulesStage: Sending prompt to LLM.")
        response, reasoning, usage = self.client.complete(prompt=prompt)

        # 4. Unpack responses and update token usage
        self._update_token_usage(input_data, usage)
        try:
            directed_edges = extract_directed_edges_literal_format_json(answer=response)
            undirected_edges = extract_undirected_edges_literal_format_json(answer=response)
            input_data["directed_edges"] = directed_edges
            input_data["undirected_edges"] = undirected_edges
        except Exception as e:
            logging.error("Error extracting directed edges: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["directed_edges"] = None
            input_data["undirected_edges"] = None

        self._build_stage_history(
            input_data,
            reasoning=reasoning,
            input_keys={"premise": "premise", "skeleton": "undirected_edges", "v_structures": "v_structures"},
            output_keys={"directed_edges": "directed_edges", "undirected_edges": "undirected_edges"},
        )
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("MeekRulesStage: Processing batch with %d samples.", len(inputs))

        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges", "v_structures"}
        for i, input_data in enumerate(inputs):
            missing_keys = required_keys - input_data.keys()
            if missing_keys:
                logging.error("Sample %d is missing keys: %s", i, missing_keys)
                raise ValueError(f"Sample {i}: Input data must contain: {', '.join(required_keys)}.")

        # 2. Split inputs into valid and already-failed
        valid_indices = []
        for i, input_data in enumerate(inputs):
            if input_data["undirected_edges"] is None or input_data["v_structures"] is None:
                logging.warning("Sample %d skipping MeekRulesStage: prior stage failed.", i)
                input_data["directed_edges"] = None
                input_data["undirected_edges"] = None
            else:
                valid_indices.append(i)

        if not valid_indices:
            return inputs

        # 3. Build prompts for valid samples only
        prompts = []
        for i in valid_indices:
            input_data = inputs[i]
            prior_history = self._format_prior_history(input_data)
            prompt = self.prompt_template.format(
                premise=input_data["premise"],
                nodes=input_data["nodes"],
                edges=self._format_edges(input_data["undirected_edges"]),
                v_structures=input_data["v_structures"],
                prior_history=prior_history,
            )
            prompts.append(prompt)
            logging.debug("Constructed prompt for sample %d: %s", i, prompt)

        logging.debug("All prompts constructed for batch: %s", prompts)

        # 4. Send batch (valid samples only)
        try:
            responses = self.client.complete_batch(prompts=prompts)
            logging.info("Batch call returned %d responses.", len(responses))
        except Exception as e:
            logging.error("Batch call failed: %s", e)
            raise

        # 5. Unpack responses and update token usage for valid samples
        for j, i in enumerate(valid_indices):
            text, reasoning, usage = responses[j]
            logging.debug("Raw response text for sample %d: %s", i, text)
            logging.debug("Token usage for sample %d: %s", i, usage)
            self._update_token_usage(inputs[i], usage)

        # 6. Parse directed/undirected edges from each valid response
        for j, i in enumerate(valid_indices):
            text, reasoning, _ = responses[j]
            try:
                directed_edges = extract_directed_edges_literal_format_json(answer=text)
                undirected_edges = extract_undirected_edges_literal_format_json(answer=text)
                logging.debug("Extracted directed_edges for sample %d: %s", i, directed_edges)
            except Exception as e:
                logging.error("Error extracting directed_edges for sample %d: %s", i, e)
                logging.debug("Problematic response for sample %d: %s", i, text)
                directed_edges = None
                undirected_edges = None
            inputs[i]["directed_edges"] = directed_edges
            inputs[i]["undirected_edges"] = undirected_edges

            self._build_stage_history(
                inputs[i],
                reasoning=reasoning,
                input_keys={"premise": "premise", "skeleton": "undirected_edges", "v_structures": "v_structures"},
                output_keys={"directed_edges": "directed_edges", "undirected_edges": "undirected_edges"},
            )

        return inputs


class HypothesisEvaluationStage(Stage):
    """
    Stage for evaluating the hypothesis based on the directed edges.
    """
    prompt_template = Stage.prompts["hypothesis_evaluation"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "directed_edges", "hypothesis", "undirected_edges"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Hypothesis evaluation stage input data must contain: {', '.join(required_keys)}.")

        # 2. Build prompt
        prior_history = self._format_prior_history(input_data)
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            directed_edges=self._format_edges(input_data["directed_edges"]),
            undirected_edges=self._format_edges(input_data["undirected_edges"]),
            hypothesis=input_data["hypothesis"],
            prior_history=prior_history,
        )

        # 3. Send request to LLM
        logging.info("HypothesisEvaluationStage: Sending prompt to LLM.")
        response, reasoning, usage = self.client.complete(prompt=prompt)

        # 4. Unpack responses and update token usage
        self._update_token_usage(input_data, usage)
        try:
            hypothesis_label = extract_hypothesis_answer(answer=response)
            input_data["hypothesis_label"] = hypothesis_label
        except Exception as e:
            logging.error("Error extracting hypothesis_label: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["hypothesis_label"] = None

        self._build_stage_history(
            input_data,
            reasoning=reasoning,
            input_keys={
                "premise": "premise",
                "directed_edges": "directed_edges",
                "undirected_edges": "undirected_edges",
                "hypothesis": "hypothesis",
            },
            output_keys={"hypothesis_label": "hypothesis_label"},
        )
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("HypothesisEvaluationStage: Processing batch with %d samples.", len(inputs))

        # 1. Validate inputs
        required_keys = {"premise", "nodes", "directed_edges", "hypothesis", "undirected_edges"}
        for i, input_data in enumerate(inputs):
            if not required_keys.issubset(input_data):
                missing = required_keys - input_data.keys()
                logging.error("Sample %d is missing keys: %s", i, missing)
                raise ValueError(f"Sample {i} must contain: {', '.join(required_keys)}.")

        # 2. Split inputs into valid and already-failed
        valid_indices = []
        for i, input_data in enumerate(inputs):
            if input_data["directed_edges"] is None or input_data["undirected_edges"] is None:
                logging.warning("Sample %d skipping HypothesisEvaluationStage: prior stage failed.", i)
                input_data["hypothesis_label"] = None
            else:
                valid_indices.append(i)

        if not valid_indices:
            return inputs

        # 3. Build prompts for valid samples only
        prompts = []
        for i in valid_indices:
            input_data = inputs[i]
            prior_history = self._format_prior_history(input_data)
            prompt = self.prompt_template.format(
                premise=input_data["premise"],
                nodes=input_data["nodes"],
                directed_edges=self._format_edges(input_data["directed_edges"]),
                undirected_edges=self._format_edges(input_data["undirected_edges"]),
                hypothesis=input_data["hypothesis"],
                prior_history=prior_history,
            )
            prompts.append(prompt)
            logging.debug("Constructed prompt for sample %d: %s", i, prompt)

        logging.debug("All prompts constructed: %s", prompts)

        # 4. Send batch (valid samples only)
        try:
            responses = self.client.complete_batch(prompts=prompts)
            logging.info("Batch call returned %d responses.", len(responses))
        except Exception as e:
            logging.error("Batch call failed: %s", e)
            raise

        # 5. Unpack responses and update token usage for valid samples
        for j, i in enumerate(valid_indices):
            text, reasoning, usage = responses[j]
            logging.debug("Raw response text for sample %d: %s", i, text)
            logging.debug("Token usage for sample %d: %s", i, usage)
            self._update_token_usage(inputs[i], usage)

        # 6. Parse hypothesis label from each valid response
        for j, i in enumerate(valid_indices):
            text, reasoning, _ = responses[j]
            try:
                hypothesis_label = extract_hypothesis_answer(answer=text)
                inputs[i]["hypothesis_label"] = hypothesis_label
                logging.debug("Extracted hypothesis_label for sample %d: %s", i, hypothesis_label)
            except Exception as e:
                logging.error("Error extracting hypothesis_label for sample %d: %s", i, e)
                logging.debug("Problematic response for sample %d: %s", i, text)
                inputs[i]["hypothesis_label"] = None

            self._build_stage_history(
                inputs[i],
                reasoning=reasoning,
                input_keys={
                    "premise": "premise",
                    "directed_edges": "directed_edges",
                    "undirected_edges": "undirected_edges",
                    "hypothesis": "hypothesis",
                },
                output_keys={"hypothesis_label": "hypothesis_label"},
            )

        return inputs
