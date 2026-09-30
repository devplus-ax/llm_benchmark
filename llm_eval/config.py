"""
Configuration models and CLI / File parsing for the evaluation suite.
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class EndpointConfig:
    host: str = "localhost"
    port: int = 8000
    protocol: str = "http"
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "default"
    api_type: str = "vllm"  # 'vllm', 'openai', 'custom'
    custom_payload_field: str = "prompt"  # 'prompt', 'message', or 'messages'
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    mock_mode: bool = False

    def get_base_url(self) -> str:
        if self.endpoint_url:
            return self.endpoint_url
        return f"{self.protocol}://{self.host}:{self.port}"

    def get_chat_url(self) -> str:
        if self.endpoint_url:
            return self.endpoint_url
        base = self.get_base_url().rstrip("/")
        if self.api_type in ("vllm", "openai"):
            return f"{base}/v1/chat/completions"
        return f"{base}/chat"


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    backend: str = "vllm"  # 'vllm', 'openai', 'custom'
    dataset_name: str = "random"  # 'sharegpt', 'random', 'sonnet', 'custom'
    dataset_path: Optional[str] = None
    request_rate: float = 5.0  # requests per second, or inf (float('inf'))
    num_prompts: int = 50
    max_concurrency: int = 16
    input_len: int = 128
    output_len: int = 128
    stream: bool = True
    temperature: float = 0.0
    num_warmups: int = 0


@dataclass
class OwaspConfig:
    enabled: bool = True
    test_dir: str = "owasp-llm-security-community-tests/tests"
    categories: List[str] = field(default_factory=lambda: ["all"])
    include_agentic: bool = True
    max_tests: Optional[int] = None
    system_prompt: Optional[str] = None


@dataclass
class LlmAttacksConfig:
    enabled: bool = True
    advbench_path: str = "llm-attacks/data/advbench/harmful_behaviors.csv"
    advbench_samples: int = 20
    test_jailbreaks: bool = True
    run_gcg_script: bool = False
    gcg_model: str = "vicuna"
    gcg_setup: str = "behaviors"
    gcg_data_offset: int = 0
    gcg_n_train_data: int = 10


@dataclass
class ReportConfig:
    output_dir: str = "reports"
    report_name: Optional[str] = None
    format: str = "both"  # 'json', 'html', 'both'
    title: str = "Consolidated LLM Benchmark & Security Evaluation Report"
    save_detailed: bool = True


@dataclass
class SuiteConfig:
    suite: str = "all"  # 'all', 'benchmark', 'security', 'owasp', 'llm-attacks'
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    owasp: OwaspConfig = field(default_factory=OwaspConfig)
    llm_attacks: LlmAttacksConfig = field(default_factory=LlmAttacksConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required to parse YAML configs. Please install pyyaml or use JSON.")
            return yaml.safe_load(f) or {}
        elif path.suffix.lower() == ".json":
            return json.load(f)
        else:
            # Try YAML first, then JSON
            content = f.read()
            if YAML_AVAILABLE:
                try:
                    return yaml.safe_load(content) or {}
                except Exception:
                    pass
            return json.loads(content)


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct a full-featured CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified LLM Evaluation Suite: Serving Benchmark, OWASP Security & LLM-Attacks Adversarial Tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General / Config options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML or JSON config file. CLI flags override file values.",
    )
    parser.add_argument(
        "--suite",
        choices=["all", "benchmark", "security", "owasp", "llm-attacks"],
        default="all",
        help="Test suite to run: all, benchmark only, security (owasp + llm-attacks), owasp only, or llm-attacks only.",
    )
    parser.add_argument(
        "--mock",
        "--dry-run",
        action="store_true",
        dest="mock_mode",
        help="Simulate LLM responses (mock engine) without requiring an active server. Ideal for testing & pipeline verification.",
    )

    # Endpoint options
    ep_group = parser.add_argument_group("Endpoint Configuration")
    ep_group.add_argument("--host", type=str, default=None, help="Host address of LLM server (e.g. localhost)")
    ep_group.add_argument("--port", type=int, default=None, help="Port of LLM server (e.g. 8000)")
    ep_group.add_argument("--protocol", choices=["http", "https"], default="http", help="Protocol")
    ep_group.add_argument("--endpoint-url", type=str, default=None, help="Full URL override for chat/completion endpoint")
    ep_group.add_argument("--api-key", type=str, default=None, help="API Authorization token / key")
    ep_group.add_argument("--model", type=str, default=None, help="Model name or identifier (e.g. google/gemma-4-31B-it). Pass 'auto' to discover models via /v1/models and test all.")
    ep_group.add_argument("--api-type", choices=["vllm", "openai", "custom"], default=None, help="API format type")
    ep_group.add_argument("--custom-payload-field", type=str, default="prompt", help="Payload prompt key for custom APIs")
    ep_group.add_argument("--timeout", type=float, default=60.0, help="HTTP request timeout in seconds")

    # Benchmark options
    bm_group = parser.add_argument_group("Benchmark Options")
    bm_group.add_argument("--backend", choices=["vllm", "openai", "custom"], default="vllm", help="Benchmark backend type")
    bm_group.add_argument("--dataset-name", choices=["sharegpt", "random", "sonnet", "custom"], default="random", help="Benchmark dataset")
    bm_group.add_argument("--dataset-path", type=str, default=None, help="Path to dataset file (e.g. ShareGPT JSON)")
    bm_group.add_argument("--request-rate", type=float, default=5.0, help="Requests per second (use 'inf' for unbounded)")
    bm_group.add_argument("--num-prompts", type=int, default=50, help="Number of benchmark prompts to execute")
    bm_group.add_argument("--max-concurrency", type=int, default=16, help="Maximum concurrent requests")
    bm_group.add_argument("--input-len", type=int, default=128, help="Synthetic input prompt token length")
    bm_group.add_argument("--output-len", type=int, default=128, help="Target output token length")
    bm_group.add_argument("--no-stream", action="store_true", help="Disable streaming responses for benchmark")

    # OWASP Security options
    sec_group = parser.add_argument_group("OWASP Security Options")
    sec_group.add_argument("--owasp-dir", type=str, default=None, help="Path to OWASP community tests directory")
    sec_group.add_argument("--owasp-categories", nargs="+", default=["all"], help="Specific OWASP categories (e.g. LLM01 LLM07 ASI01) or 'all'")
    sec_group.add_argument("--no-agentic", action="store_true", help="Exclude agentic OWASP tests (ASI01-ASI10)")
    sec_group.add_argument("--max-owasp-tests", type=int, default=None, help="Limit number of OWASP security tests")
    sec_group.add_argument("--system-prompt", type=str, default=None, help="Custom system prompt to protect / evaluate")

    # LLM-Attacks options
    atk_group = parser.add_argument_group("LLM-Attacks / Adversarial Options")
    atk_group.add_argument("--advbench-path", type=str, default=None, help="Path to AdvBench harmful_behaviors.csv")
    atk_group.add_argument("--advbench-samples", type=int, default=20, help="Number of AdvBench samples to test")
    atk_group.add_argument("--no-jailbreaks", action="store_true", help="Disable jailbreak affix testing in AdvBench")
    atk_group.add_argument("--run-gcg-script", action="store_true", help="Launch llm-attacks GCG whitebox bash script")
    atk_group.add_argument("--gcg-model", type=str, default="vicuna", help="Target model for GCG script (vicuna, llama2)")
    atk_group.add_argument("--gcg-setup", type=str, default="behaviors", help="Setup for GCG script (behaviors, strings)")

    # Report options
    rep_group = parser.add_argument_group("Report Output Options")
    rep_group.add_argument("--output-dir", type=str, default="reports", help="Directory where reports will be saved")
    rep_group.add_argument("--report-name", type=str, default=None, help="Custom base name for report files")
    rep_group.add_argument("--format", choices=["json", "html", "both"], default="both", help="Output report format")
    rep_group.add_argument("--report-title", type=str, default=None, help="Custom title for HTML report")

    return parser


def parse_args_and_config(argv: Optional[List[str]] = None) -> SuiteConfig:
    """Parse CLI arguments, load config file if specified, and merge settings."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # 1. Start with defaults
    cfg = SuiteConfig()

    # 2. If a config file was passed, load and update
    if args.config:
        raw_cfg = load_config_from_file(args.config)
        
        # Merge suite
        if "suite" in raw_cfg:
            cfg.suite = raw_cfg["suite"]
            
        # Merge endpoint
        if "endpoint" in raw_cfg and isinstance(raw_cfg["endpoint"], dict):
            for k, v in raw_cfg["endpoint"].items():
                if hasattr(cfg.endpoint, k):
                    setattr(cfg.endpoint, k, v)
                    
        # Merge benchmark
        if "benchmark" in raw_cfg and isinstance(raw_cfg["benchmark"], dict):
            for k, v in raw_cfg["benchmark"].items():
                if hasattr(cfg.benchmark, k):
                    setattr(cfg.benchmark, k, v)
                    
        # Merge owasp
        if "owasp" in raw_cfg and isinstance(raw_cfg["owasp"], dict):
            for k, v in raw_cfg["owasp"].items():
                if hasattr(cfg.owasp, k):
                    setattr(cfg.owasp, k, v)
                    
        # Merge llm_attacks
        if "llm_attacks" in raw_cfg and isinstance(raw_cfg["llm_attacks"], dict):
            for k, v in raw_cfg["llm_attacks"].items():
                if hasattr(cfg.llm_attacks, k):
                    setattr(cfg.llm_attacks, k, v)
                    
        # Merge report
        if "report" in raw_cfg and isinstance(raw_cfg["report"], dict):
            for k, v in raw_cfg["report"].items():
                if hasattr(cfg.report, k):
                    setattr(cfg.report, k, v)

    # 3. CLI arguments override file values
    if args.suite:
        cfg.suite = args.suite
    if args.mock_mode:
        cfg.endpoint.mock_mode = True

    # Endpoint
    if args.host is not None:
        cfg.endpoint.host = args.host
    if args.port is not None:
        cfg.endpoint.port = args.port
    if args.protocol is not None:
        cfg.endpoint.protocol = args.protocol
    if args.endpoint_url is not None:
        cfg.endpoint.endpoint_url = args.endpoint_url
    if args.api_key is not None:
        cfg.endpoint.api_key = args.api_key
    if args.model is not None:
        cfg.endpoint.model = args.model
    if args.api_type is not None:
        cfg.endpoint.api_type = args.api_type
    if args.custom_payload_field is not None:
        cfg.endpoint.custom_payload_field = args.custom_payload_field
    if args.timeout is not None:
        cfg.endpoint.timeout_seconds = args.timeout

    # Benchmark
    if args.backend is not None:
        cfg.benchmark.backend = args.backend
    if args.dataset_name is not None:
        cfg.benchmark.dataset_name = args.dataset_name
    if args.dataset_path is not None:
        cfg.benchmark.dataset_path = args.dataset_path
    if args.request_rate is not None:
        cfg.benchmark.request_rate = args.request_rate
    if args.num_prompts is not None:
        cfg.benchmark.num_prompts = args.num_prompts
    if args.max_concurrency is not None:
        cfg.benchmark.max_concurrency = args.max_concurrency
    if args.input_len is not None:
        cfg.benchmark.input_len = args.input_len
    if args.output_len is not None:
        cfg.benchmark.output_len = args.output_len
    if args.no_stream:
        cfg.benchmark.stream = False

    # OWASP
    if args.owasp_dir is not None:
        cfg.owasp.test_dir = args.owasp_dir
    if args.owasp_categories:
        cfg.owasp.categories = args.owasp_categories
    if args.no_agentic:
        cfg.owasp.include_agentic = False
    if args.max_owasp_tests is not None:
        cfg.owasp.max_tests = args.max_owasp_tests
    if args.system_prompt is not None:
        cfg.owasp.system_prompt = args.system_prompt

    # LLM-Attacks
    if args.advbench_path is not None:
        cfg.llm_attacks.advbench_path = args.advbench_path
    if args.advbench_samples is not None:
        cfg.llm_attacks.advbench_samples = args.advbench_samples
    if args.no_jailbreaks:
        cfg.llm_attacks.test_jailbreaks = False
    if args.run_gcg_script:
        cfg.llm_attacks.run_gcg_script = True
    if args.gcg_model:
        cfg.llm_attacks.gcg_model = args.gcg_model
    if args.gcg_setup:
        cfg.llm_attacks.gcg_setup = args.gcg_setup

    # Report
    if args.output_dir is not None:
        cfg.report.output_dir = args.output_dir
    if args.report_name is not None:
        cfg.report.report_name = args.report_name
    if args.format is not None:
        cfg.report.format = args.format
    if args.report_title is not None:
        cfg.report.title = args.report_title

    # Set enable flags based on suite selection
    if cfg.suite == "benchmark":
        cfg.benchmark.enabled = True
        cfg.owasp.enabled = False
        cfg.llm_attacks.enabled = False
    elif cfg.suite == "security":
        cfg.benchmark.enabled = False
        cfg.owasp.enabled = True
        cfg.llm_attacks.enabled = True
    elif cfg.suite == "owasp":
        cfg.benchmark.enabled = False
        cfg.owasp.enabled = True
        cfg.llm_attacks.enabled = False
    elif cfg.suite == "llm-attacks":
        cfg.benchmark.enabled = False
        cfg.owasp.enabled = False
        cfg.llm_attacks.enabled = True
    else:  # 'all'
        cfg.benchmark.enabled = True
        cfg.owasp.enabled = True
        cfg.llm_attacks.enabled = True

    return cfg
