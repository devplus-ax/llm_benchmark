"""
Dataset loaders and prompt generators for LLM serving benchmarks.
Supports ShareGPT, synthetic random prompts, Sonnet text, and custom datasets.
"""

import json
import os
from pathlib import Path
import random
from typing import List, Optional


SONNET_SAMPLE = """From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel:
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And tender churl mak'st waste in niggarding:
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee."""

WORDS_CORPUS = [
    "artificial", "intelligence", "distributed", "serving", "inference", "benchmark",
    "throughput", "latency", "acceleration", "quantization", "attention", "transformer",
    "parallelism", "pipeline", "execution", "algorithm", "optimization", "parameter",
    "context", "token", "generation", "sampling", "streaming", "memory", "bandwidth",
    "evaluate", "architecture", "embedding", "vector", "cluster", "node", "network",
    "database", "repository", "security", "guardrail", "adversarial", "vulnerability"
]


def generate_random_prompt(target_tokens: int = 128) -> str:
    """Generate a pseudo-random sentence with roughly target_tokens words."""
    words = []
    # ~1.3 tokens per word approx
    num_words = max(10, int(target_tokens / 1.3))
    for i in range(num_words):
        word = random.choice(WORDS_CORPUS)
        if i % 12 == 0 and i > 0:
            words.append(word + ".")
        elif i % 6 == 0 and i > 0:
            words.append(word + ",")
        else:
            words.append(word)
    sentence = " ".join(words).capitalize()
    if not sentence.endswith("."):
        sentence += "."
    return f"Please analyze and summarize the following technical text:\n{sentence}"


def load_sharegpt_prompts(dataset_path: str, num_prompts: int) -> List[str]:
    """Parse human prompts from ShareGPT format JSON."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"ShareGPT dataset file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[str] = []
    for item in data:
        conversations = item.get("conversations", [])
        for turn in conversations:
            speaker = turn.get("from", "").lower()
            val = turn.get("value", "")
            if speaker in ("human", "user") and len(val.strip()) > 10:
                prompts.append(val.strip())
                break
        if len(prompts) >= num_prompts:
            break

    if not prompts:
        raise ValueError(f"No valid human prompts found in ShareGPT file: {dataset_path}")

    return prompts[:num_prompts]


def load_sonnet_prompts(num_prompts: int) -> List[str]:
    """Load Shakespeare sonnet chunks."""
    # Look for sonnet.txt in vllm/benchmarks if present
    sonnet_path = Path("vllm/benchmarks/sonnet.txt")
    if sonnet_path.is_file():
        try:
            with open(sonnet_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = [l.strip() for l in content.split("\n\n") if len(l.strip()) > 30]
            if lines:
                results = []
                while len(results) < num_prompts:
                    results.extend(lines)
                return results[:num_prompts]
        except Exception:
            pass

    # Fallback to embedded sonnet
    return [f"Please analyze the themes and rhythm in the following excerpt:\n{SONNET_SAMPLE}" for _ in range(num_prompts)]


def load_benchmark_prompts(
    dataset_name: str = "random",
    dataset_path: Optional[str] = None,
    num_prompts: int = 50,
    input_len: int = 128,
) -> List[str]:
    """Factory function to load or generate benchmark prompts."""
    d_name = dataset_name.lower()

    if d_name == "sharegpt":
        if not dataset_path:
            # Check default path
            default_path = Path("./ShareGPT_V3_unfiltered_cleaned_split.json")
            if default_path.is_file():
                dataset_path = str(default_path)
            else:
                # If ShareGPT requested but file not present, generate realistic synthetic prompts
                return [generate_random_prompt(input_len) for _ in range(num_prompts)]
        return load_sharegpt_prompts(dataset_path, num_prompts)

    elif d_name == "sonnet":
        return load_sonnet_prompts(num_prompts)

    elif d_name == "custom" and dataset_path:
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Custom dataset file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix == ".json":
                items = json.load(f)
                if isinstance(items, list):
                    return [str(x) for x in items][:num_prompts]
            lines = [l.strip() for l in f.readlines() if l.strip()]
            return lines[:num_prompts]

    # Default: random synthetic prompts
    return [generate_random_prompt(input_len) for _ in range(num_prompts)]
