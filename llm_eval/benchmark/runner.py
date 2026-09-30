"""
Asynchronous Benchmark Runner for measuring serving throughput and latency distributions.
"""

import asyncio
import math
import random
import time
from typing import Any, Dict, List, Optional

from llm_eval.benchmark.datasets import load_benchmark_prompts
from llm_eval.client import AsyncLLMClient, LLMResponse
from llm_eval.config import BenchmarkConfig, EndpointConfig
from llm_eval.utils import calculate_percentiles, log_header, log_info, log_success, log_warning


class BenchmarkRunner:
    """Orchestrates concurrent load testing and captures low-level latency metrics."""

    def __init__(self, endpoint_cfg: EndpointConfig, bench_cfg: BenchmarkConfig):
        self.endpoint_cfg = endpoint_cfg
        self.bench_cfg = bench_cfg

    async def run(self) -> Dict[str, Any]:
        """Execute the benchmark run and return consolidated metrics."""
        log_header("Running LLM Serving Throughput & Latency Benchmark")
        log_info(f"Backend: {self.bench_cfg.backend} | Target: {self.endpoint_cfg.get_chat_url()}")
        log_info(f"Dataset: {self.bench_cfg.dataset_name} | Prompts: {self.bench_cfg.num_prompts} | Rate: {self.bench_cfg.request_rate} req/s")

        prompts = load_benchmark_prompts(
            dataset_name=self.bench_cfg.dataset_name,
            dataset_path=self.bench_cfg.dataset_path,
            num_prompts=self.bench_cfg.num_prompts,
            input_len=self.bench_cfg.input_len,
        )

        async with AsyncLLMClient(self.endpoint_cfg) as client:
            # 1. Warmup if requested
            if self.bench_cfg.num_warmups > 0:
                log_info(f"Sending {self.bench_cfg.num_warmups} warmup request(s)...")
                warmup_tasks = [
                    client.query(
                        prompt="Hello, ready for inference benchmark?",
                        max_tokens=16,
                        stream=False,
                    )
                    for _ in range(self.bench_cfg.num_warmups)
                ]
                await asyncio.gather(*warmup_tasks, return_exceptions=True)
                log_info("Warmup complete.")

            # 2. Main benchmark load generation
            sem = asyncio.Semaphore(self.bench_cfg.max_concurrency)
            responses: List[LLMResponse] = []
            start_wall_time = time.perf_counter()

            async def send_worker(prompt: str, send_delay: float):
                if send_delay > 0:
                    await asyncio.sleep(send_delay)
                async with sem:
                    res = await client.query(
                        prompt=prompt,
                        max_tokens=self.bench_cfg.output_len,
                        temperature=self.bench_cfg.temperature,
                        stream=self.bench_cfg.stream,
                    )
                    responses.append(res)

            tasks = []
            cumulative_delay = 0.0
            rate = self.bench_cfg.request_rate

            for prompt in prompts:
                if rate > 0 and rate != float("inf"):
                    # Poisson interval between requests
                    interval = random.expovariate(rate)
                    cumulative_delay += interval
                    delay = cumulative_delay
                else:
                    delay = 0.0

                tasks.append(asyncio.create_task(send_worker(prompt, delay)))

            # Wait for all benchmark requests to finish
            await asyncio.gather(*tasks, return_exceptions=True)
            end_wall_time = time.perf_counter()

        duration = max(0.001, end_wall_time - start_wall_time)
        return self._process_results(responses, duration)

    def _process_results(self, responses: List[LLMResponse], duration: float) -> Dict[str, Any]:
        """Aggregate per-request telemetry into percentiles and throughput rates."""
        total_requests = len(responses)
        successful = [r for r in responses if r.success]
        failed = [r for r in responses if not r.success]
        success_count = len(successful)
        failed_count = len(failed)

        total_input_tokens = sum(r.input_tokens for r in successful)
        total_output_tokens = sum(r.output_tokens for r in successful)

        # Latencies in milliseconds
        latencies_ms = [r.latency_s * 1000.0 for r in successful]
        ttfts_ms = [r.ttft_s * 1000.0 for r in successful if r.ttft_s is not None]
        tpots_ms = [r.tpot_s * 1000.0 for r in successful if r.tpot_s is not None]

        all_itls_ms: List[float] = []
        for r in successful:
            all_itls_ms.extend([itl * 1000.0 for itl in r.inter_token_latencies])

        req_throughput = success_count / duration
        out_tok_throughput = total_output_tokens / duration
        in_tok_throughput = total_input_tokens / duration
        total_tok_throughput = (total_input_tokens + total_output_tokens) / duration

        lat_stats = calculate_percentiles(latencies_ms)
        ttft_stats = calculate_percentiles(ttfts_ms)
        tpot_stats = calculate_percentiles(tpots_ms)
        itl_stats = calculate_percentiles(all_itls_ms)

        result = {
            "summary": {
                "duration_seconds": round(duration, 3),
                "total_requests": total_requests,
                "successful_requests": success_count,
                "failed_requests": failed_count,
                "error_rate_percent": round((failed_count / total_requests) * 100, 2) if total_requests else 0.0,
                "request_throughput_req_per_s": round(req_throughput, 2),
                "output_token_throughput_tok_per_s": round(out_tok_throughput, 2),
                "input_token_throughput_tok_per_s": round(in_tok_throughput, 2),
                "total_token_throughput_tok_per_s": round(total_tok_throughput, 2),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            },
            "metrics": {
                "e2e_latency_ms": lat_stats,
                "ttft_ms": ttft_stats,
                "tpot_ms": tpot_stats,
                "itl_ms": itl_stats,
            },
            "config": {
                "backend": self.bench_cfg.backend,
                "dataset_name": self.bench_cfg.dataset_name,
                "num_prompts": self.bench_cfg.num_prompts,
                "request_rate": self.bench_cfg.request_rate,
                "max_concurrency": self.bench_cfg.max_concurrency,
                "input_len": self.bench_cfg.input_len,
                "output_len": self.bench_cfg.output_len,
                "stream": self.bench_cfg.stream,
            },
            "detailed_requests": [r.to_dict() for r in responses[:100]],  # Store sample detailed requests
        }

        # Terminal feedback
        log_success(f"Benchmark finished in {duration:.2f}s!")
        log_info(f"Throughput: {req_throughput:.2f} req/s | {out_tok_throughput:.2f} output tok/s")
        log_info(f"TTFT (P50/P95/P99): {ttft_stats['p50']}ms / {ttft_stats['p95']}ms / {ttft_stats['p99']}ms")
        log_info(f"TPOT (P50/P95/P99): {tpot_stats['p50']}ms / {tpot_stats['p95']}ms / {tpot_stats['p99']}ms")
        if failed_count > 0:
            log_warning(f"Failed requests: {failed_count} ({result['summary']['error_rate_percent']}%)")

        return result
