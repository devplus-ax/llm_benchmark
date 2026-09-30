#!/usr/bin/env python3
"""
Consolidated LLM Benchmark & Security Evaluation Runner.
Executes vLLM serving throughput benchmarks, OWASP Top 10 security tests,
and LLM-Attacks adversarial evaluations, generating comprehensive JSON and HTML reports.
Supports '--model auto' to discover all models from /v1/models and test them automatically.
"""

import asyncio
import copy
import datetime
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_eval.benchmark.runner import BenchmarkRunner
from llm_eval.client import AsyncLLMClient
from llm_eval.config import SuiteConfig, parse_args_and_config
from llm_eval.reporting.html_reporter import generate_html_report, generate_multi_model_html_report
from llm_eval.reporting.json_reporter import save_json_report
from llm_eval.security.advbench_runner import AdvbenchRunner
from llm_eval.security.gcg_runner import GcgScriptRunner
from llm_eval.security.owasp_runner import OwaspSecurityRunner
from llm_eval.utils import Colors, get_system_info, log_header, log_info, log_success, log_warning


async def run_single_model(cfg: SuiteConfig) -> Dict[str, Any]:
    """Execute benchmark and security tests for a specific model."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    start_perf = time.perf_counter()

    log_header(f"Evaluating Model: {cfg.endpoint.model}")
    log_info(f"Target Endpoint: {cfg.endpoint.get_chat_url()}")
    log_info(f"Suite Mode: {cfg.suite.upper()} | Mock Mode: {cfg.endpoint.mock_mode}")

    benchmark_results: Dict[str, Any] = {}
    owasp_results: Dict[str, Any] = {}
    advbench_results: Dict[str, Any] = {}
    gcg_results: Dict[str, Any] = {}

    # 1. Run Serving Benchmark
    if cfg.benchmark.enabled:
        bm_runner = BenchmarkRunner(cfg.endpoint, cfg.benchmark)
        benchmark_results = await bm_runner.run()

    # 2. Run OWASP Security Tests
    if cfg.owasp.enabled:
        owasp_runner = OwaspSecurityRunner(cfg.endpoint, cfg.owasp)
        owasp_results = await owasp_runner.run()

    # 3. Run LLM-Attacks / AdvBench Evaluation
    if cfg.llm_attacks.enabled:
        adv_runner = AdvbenchRunner(cfg.endpoint, cfg.llm_attacks)
        advbench_results = await adv_runner.run()

        # Optional GCG whitebox script execution
        if cfg.llm_attacks.run_gcg_script:
            gcg_runner = GcgScriptRunner(cfg.llm_attacks)
            gcg_results = gcg_runner.run()

    total_duration = time.perf_counter() - start_perf

    # 4. Synthesize Executive Summary
    owasp_summary = owasp_results.get("summary", {})
    adv_summary = advbench_results.get("summary", {})
    bm_summary = benchmark_results.get("summary", {})
    bm_metrics = benchmark_results.get("metrics", {})

    owasp_score = owasp_summary.get("security_score_percent", 100.0)
    adv_score = adv_summary.get("safety_resistance_percent", 100.0)

    # Combined Security Score (weighted 50% OWASP, 50% AdvBench if both run)
    if cfg.owasp.enabled and cfg.llm_attacks.enabled:
        overall_sec_score = round((owasp_score * 0.5) + (adv_score * 0.5), 1)
    elif cfg.owasp.enabled:
        overall_sec_score = owasp_score
    elif cfg.llm_attacks.enabled:
        overall_sec_score = adv_score
    else:
        overall_sec_score = 100.0

    exec_summary = {
        "overall_security_score": overall_sec_score,
        "total_test_duration_seconds": round(total_duration, 2),
        "benchmark": {
            "request_throughput_req_per_s": bm_summary.get("request_throughput_req_per_s", 0.0),
            "output_token_throughput_tok_per_s": bm_summary.get("output_token_throughput_tok_per_s", 0.0),
            "ttft_p50_ms": bm_metrics.get("ttft_ms", {}).get("p50", 0.0),
            "ttft_p95_ms": bm_metrics.get("ttft_ms", {}).get("p95", 0.0),
            "tpot_p50_ms": bm_metrics.get("tpot_ms", {}).get("p50", 0.0),
            "tpot_p95_ms": bm_metrics.get("tpot_ms", {}).get("p95", 0.0),
            "total_requests": bm_summary.get("total_requests", 0),
            "successful_requests": bm_summary.get("successful_requests", 0),
            "failed_requests": bm_summary.get("failed_requests", 0),
        },
        "security": {
            "owasp_score_percent": owasp_score,
            "owasp_total": owasp_summary.get("total_tests", 0),
            "owasp_passed": owasp_summary.get("passed", 0),
            "owasp_vulnerable": owasp_summary.get("vulnerable", 0),
            "owasp_severities": owasp_summary.get("severity_breakdown", {}),
            "advbench_attacks_tested": adv_summary.get("total_attacks", 0),
            "advbench_attacks_blocked": adv_summary.get("blocked", 0),
            "advbench_attacks_bypassed": adv_summary.get("bypassed", 0),
            "advbench_asr_percent": adv_summary.get("attack_success_rate_percent", 0.0),
            "advbench_resistance_percent": adv_score,
        }
    }

    # Consolidated Report Object
    report_data = {
        "metadata": {
            "timestamp": start_time.isoformat(),
            "model": cfg.endpoint.model,
            "endpoint_url": cfg.endpoint.get_chat_url(),
            "suite": cfg.suite,
            "mock_mode": cfg.endpoint.mock_mode,
            "system_info": get_system_info(),
        },
        "executive_summary": exec_summary,
        "config": cfg.to_dict(),
        "benchmark": benchmark_results,
        "owasp": owasp_results,
        "advbench": advbench_results,
        "gcg": gcg_results,
    }

    # 5. Output Reports
    out_dir = Path(cfg.report.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    sanitized_model = cfg.endpoint.model.replace("/", "_").replace("\\", "_").replace(":", "_")
    base_name = cfg.report.report_name or f"llm_eval_{sanitized_model}_{timestamp_str}"

    json_path = out_dir / f"{base_name}.json"
    html_path = out_dir / f"{base_name}.html"

    log_header(f"Generating Evaluation Reports for {cfg.endpoint.model}")

    if cfg.report.format in ("json", "both"):
        save_json_report(report_data, json_path)

    if cfg.report.format in ("html", "both"):
        generate_html_report(report_data, html_path)

    # 6. Final Summary Card in Console
    log_header(f"Summary: {cfg.endpoint.model}")
    print(f"  • Model:                {cfg.endpoint.model}")
    print(f"  • Total Duration:       {total_duration:.2f}s")
    if cfg.benchmark.enabled:
        print(f"  • Serving Throughput:   {exec_summary['benchmark']['output_token_throughput_tok_per_s']} tok/s ({exec_summary['benchmark']['request_throughput_req_per_s']} req/s)")
        print(f"  • TTFT (P50 / P95):     {exec_summary['benchmark']['ttft_p50_ms']} ms / {exec_summary['benchmark']['ttft_p95_ms']} ms")
        print(f"  • TPOT (P50 / P95):     {exec_summary['benchmark']['tpot_p50_ms']} ms / {exec_summary['benchmark']['tpot_p95_ms']} ms")
    if cfg.owasp.enabled:
        print(f"  • OWASP Security Score: {owasp_score}% ({owasp_summary.get('passed', 0)}/{owasp_summary.get('total_tests', 0)} Passed)")
        print(f"  • OWASP Vulnerabilities:{owasp_summary.get('vulnerable', 0)} (Crit: {owasp_summary.get('severity_breakdown', {}).get('CRITICAL', 0)}, High: {owasp_summary.get('severity_breakdown', {}).get('HIGH', 0)})")
    if cfg.llm_attacks.enabled:
        print(f"  • AdvBench Resistance:  {adv_score}% (ASR: {adv_summary.get('attack_success_rate_percent', 0.0)}%)")
    print(f"  • Overall Health Score: {overall_sec_score}%")
    print(f"\n{Colors.BOLD}{Colors.GREEN}Report:{Colors.RESET} {html_path.resolve()}\n")

    return {
        "model": cfg.endpoint.model,
        "executive_summary": exec_summary,
        "json_path": str(json_path.resolve()),
        "html_path": str(html_path.resolve()),
        "html_file": html_path.name,
        "report_data": report_data,
    }


def ensure_submodules_initialized():
    """Check if submodules (OWASP and AdvBench) have files; if not, auto-initialize them."""
    owasp_test = Path("owasp-llm-security-community-tests/tests/LLM01_prompt_injection.md")
    advbench_test = Path("llm-attacks/data/advbench/harmful_behaviors.csv")

    if not owasp_test.is_file() or not advbench_test.is_file():
        if Path(".gitmodules").is_file():
            log_info("Submodule test files missing or empty. Attempting auto-initialization via git...")
            try:
                import subprocess
                res = subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if res.returncode == 0:
                    log_success("Git submodules initialized successfully!")
                else:
                    log_warning(f"Git submodule update failed:\n{res.stderr.strip()}")
            except Exception as e:
                log_warning(
                    f"Could not auto-initialize submodules: {e}\n"
                    "Please run manually in this directory: git submodule update --init --recursive"
                )


async def main_async(cfg: SuiteConfig):
    # Ensure security submodules are present if security suites are enabled
    if cfg.owasp.enabled or cfg.llm_attacks.enabled:
        ensure_submodules_initialized()

    # Check if auto model discovery is requested
    target_models: List[str] = []

    if cfg.endpoint.model.lower() == "auto":
        log_header("Auto-Discovering Models from LLM Server")
        log_info(f"Querying {cfg.endpoint.get_base_url()}/v1/models ...")
        async with AsyncLLMClient(cfg.endpoint) as client:
            target_models = await client.fetch_models()

        if not target_models:
            log_warning("No models found from /v1/models or endpoint unreachable. Falling back to 'default'.")
            target_models = ["default"]
        else:
            log_success(f"Discovered {len(target_models)} model(s): {', '.join(target_models)}")
    else:
        target_models = [cfg.endpoint.model]

    # Evaluate each target model
    model_reports: List[Dict[str, Any]] = []
    total_models = len(target_models)

    for i, model_name in enumerate(target_models, start=1):
        if total_models > 1:
            log_header(f"Running Evaluation [{i}/{total_models}]: {model_name}")

        model_cfg = copy.deepcopy(cfg)
        model_cfg.endpoint.model = model_name
        # Clear custom report name so each model gets its own distinct report filename
        if total_models > 1:
            model_cfg.report.report_name = None

        report_info = await run_single_model(model_cfg)
        model_reports.append(report_info)

    # If multiple models were tested, generate a consolidated multi-model comparison dashboard!
    if len(model_reports) > 1:
        log_header("Generating Multi-Model Comparative Dashboard")
        out_dir = Path(cfg.report.output_dir)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        multi_json_path = out_dir / f"llm_eval_multi_model_comparison_{timestamp_str}.json"
        multi_html_path = out_dir / f"llm_eval_multi_model_comparison_{timestamp_str}.html"

        multi_data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "endpoint_url": cfg.endpoint.get_chat_url(),
            "total_models": len(model_reports),
            "models": model_reports,
        }

        save_json_report(multi_data, multi_json_path)
        generate_multi_model_html_report(multi_data, multi_html_path)

        # Print comparison table to console
        log_header("Multi-Model Comparative Summary")
        header_line = f"{'Model':<35} | {'Health':<8} | {'Throughput':<15} | {'TTFT P50':<10} | {'OWASP':<8} | {'AdvBench':<10}"
        print(f"{Colors.BOLD}{header_line}{Colors.RESET}")
        print("-" * len(header_line))

        for r in model_reports:
            m = r["model"][:33]
            es = r["executive_summary"]
            bm = es["benchmark"]
            sec = es["security"]

            score = f"{es['overall_security_score']}%"
            tp = f"{bm['output_token_throughput_tok_per_s']} tok/s"
            ttft = f"{bm['ttft_p50_ms']} ms"
            owasp = f"{sec['owasp_score_percent']}%"
            adv = f"{sec['advbench_resistance_percent']}%"

            print(f"{m:<35} | {score:<8} | {tp:<15} | {ttft:<10} | {owasp:<8} | {adv:<10}")

        print(f"\n{Colors.BOLD}{Colors.GREEN}Consolidated Multi-Model Dashboard:{Colors.RESET} {multi_html_path.resolve()}\n")


def main():
    cfg = parse_args_and_config()
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
