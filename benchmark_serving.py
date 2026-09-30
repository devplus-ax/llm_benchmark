#!/usr/bin/env python3
"""
Drop-in compatible benchmark_serving.py script.
Provides compatibility with standard vLLM serving benchmark CLI commands,
while generating modern consolidated JSON and HTML reports.
"""

import argparse
import asyncio
from pathlib import Path
import sys

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_eval.config import BenchmarkConfig, EndpointConfig, ReportConfig, SuiteConfig
from run_eval import main_async


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM-compatible serving benchmark runner with JSON & HTML reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # vLLM benchmark_serving compatibility arguments
    parser.add_argument("--backend", type=str, default="vllm", help="Backend type (vllm, openai, custom)")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--endpoint", type=str, default=None, help="Custom endpoint path")
    parser.add_argument("--model", type=str, default="default", help="Model name or path. Pass 'auto' to discover models from /v1/models and benchmark all.")
    parser.add_argument("--dataset-name", type=str, default="random", help="Dataset name: sharegpt, random, sonnet, custom")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to dataset file")
    parser.add_argument("--request-rate", type=float, default=5.0, help="Request rate in req/s (or inf)")
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of benchmark requests")
    parser.add_argument("--max-concurrency", type=int, default=16, help="Maximum concurrent requests")
    parser.add_argument("--input-len", type=int, default=128, help="Synthetic input prompt length")
    parser.add_argument("--output-len", type=int, default=128, help="Target output length")
    parser.add_argument("--num-warmups", type=int, default=0, help="Number of warmup requests")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming SSE")

    # Output options
    parser.add_argument("--save-result", action="store_true", default=True, help="Save benchmark results")
    parser.add_argument("--result-dir", type=str, default="reports", help="Directory to save reports")
    parser.add_argument("--result-filename", type=str, default=None, help="Custom filename for reports")
    parser.add_argument("--format", choices=["json", "html", "both"], default="both", help="Report format")

    # Security switch
    parser.add_argument("--with-security", action="store_true", help="Also execute OWASP and AdvBench security tests in the same run")
    parser.add_argument("--mock", "--dry-run", action="store_true", dest="mock_mode", help="Simulate inference offline")

    return parser.parse_args()


def main():
    args = parse_args()

    endpoint_cfg = EndpointConfig(
        host=args.host,
        port=args.port,
        model=args.model,
        api_type=args.backend,
        mock_mode=args.mock_mode,
    )

    bench_cfg = BenchmarkConfig(
        enabled=True,
        backend=args.backend,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        request_rate=args.request_rate,
        num_prompts=args.num_prompts,
        max_concurrency=args.max_concurrency,
        input_len=args.input_len,
        output_len=args.output_len,
        num_warmups=args.num_warmups,
        temperature=args.temperature,
        stream=not args.no_stream,
    )

    report_cfg = ReportConfig(
        output_dir=args.result_dir,
        report_name=args.result_filename,
        format=args.format,
        title=f"LLM Serving Benchmark Report - {args.model}",
    )

    suite_mode = "all" if args.with_security else "benchmark"

    cfg = SuiteConfig(
        suite=suite_mode,
        endpoint=endpoint_cfg,
        benchmark=bench_cfg,
        report=report_cfg,
    )
    if not args.with_security:
        cfg.owasp.enabled = False
        cfg.llm_attacks.enabled = False

    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
