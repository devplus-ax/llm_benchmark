"""
Automated runner for OWASP LLM Top 10 and Agentic security evaluations.
"""

import asyncio
from typing import Any, Dict, List, Optional

from llm_eval.client import AsyncLLMClient
from llm_eval.config import EndpointConfig, OwaspConfig
from llm_eval.security.indicators import check_vulnerability_indicators, is_refusal
from llm_eval.security.owasp_parser import OwaspTestCase, load_all_owasp_tests
from llm_eval.utils import log_header, log_info, log_success, log_vuln, log_warning


class OwaspSecurityRunner:
    """Executes automated security and prompt injection testing against LLM APIs."""

    def __init__(self, endpoint_cfg: EndpointConfig, owasp_cfg: OwaspConfig):
        self.endpoint_cfg = endpoint_cfg
        self.owasp_cfg = owasp_cfg

    async def run(self) -> Dict[str, Any]:
        """Execute OWASP security suite and return evaluated scorecard."""
        log_header("Running OWASP LLM Top 10 & Agentic Security Suite")
        
        test_cases = load_all_owasp_tests(
            tests_dir=self.owasp_cfg.test_dir,
            categories=self.owasp_cfg.categories,
            include_agentic=self.owasp_cfg.include_agentic,
            max_tests=self.owasp_cfg.max_tests,
        )

        if not test_cases:
            log_warning("No OWASP test cases found or matched category filters.")
            return {
                "summary": {
                    "total_tests": 0,
                    "passed": 0,
                    "vulnerable": 0,
                    "security_score_percent": 100.0,
                    "severity_breakdown": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
                },
                "category_scores": {},
                "test_results": [],
            }

        log_info(f"Loaded {len(test_cases)} OWASP security test cases across categories.")
        
        results: List[Dict[str, Any]] = []
        async with AsyncLLMClient(self.endpoint_cfg) as client:
            for i, tc in enumerate(test_cases, start=1):
                # Query model with security attack prompt
                response = await client.query(
                    prompt=tc.prompt,
                    system_prompt=self.owasp_cfg.system_prompt,
                    max_tokens=256,
                    temperature=0.0,
                    stream=False,
                )

                # Evaluate response
                is_vuln, reasons = check_vulnerability_indicators(
                    response_text=response.text,
                    test_prompt=tc.prompt,
                    expected_behavior=tc.expected_behavior,
                    indicators_raw=tc.vulnerability_indicators,
                )

                status = "VULNERABLE" if is_vuln else "PASS"
                if is_vuln:
                    log_vuln(f"[{tc.test_id}] {tc.title} ({tc.severity}) - FAILED")
                else:
                    log_success(f"[{tc.test_id}] {tc.title} ({tc.severity})")

                results.append({
                    "test_id": tc.test_id,
                    "title": tc.title,
                    "category": tc.category,
                    "severity": tc.severity,
                    "attack_type": tc.attack_type,
                    "strategy": tc.strategy,
                    "prompt": tc.prompt,
                    "expected_behavior": tc.expected_behavior,
                    "response_text": response.text,
                    "status": status,
                    "is_vulnerable": is_vuln,
                    "vulnerability_reasons": reasons,
                    "latency_ms": round(response.latency_s * 1000, 2),
                    "owasp_reference": tc.owasp_reference,
                    "source_file": tc.source_file,
                })

        return self._summarize_results(results)

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate test scores, category breakdowns, and severity metrics."""
        total = len(results)
        vulnerable = [r for r in results if r["is_vulnerable"]]
        passed = [r for r in results if not r["is_vulnerable"]]

        vuln_count = len(vulnerable)
        pass_count = len(passed)
        score_percent = round((pass_count / total) * 100, 1) if total else 100.0

        # Severity breakdown of vulnerabilities
        sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for v in vulnerable:
            sev = v.get("severity", "MEDIUM").upper()
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        # Category pass rate breakdown
        categories: Dict[str, Dict[str, int]] = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0, "vulnerable": 0}
            categories[cat]["total"] += 1
            if r["is_vulnerable"]:
                categories[cat]["vulnerable"] += 1
            else:
                categories[cat]["passed"] += 1

        cat_summary = {}
        for cat, data in categories.items():
            pass_rate = round((data["passed"] / data["total"]) * 100, 1)
            cat_summary[cat] = {
                "total": data["total"],
                "passed": data["passed"],
                "vulnerable": data["vulnerable"],
                "pass_rate_percent": pass_rate,
            }

        log_info(f"OWASP Security Score: {score_percent}% ({pass_count}/{total} Passed, {vuln_count} Vulnerable)")
        if vuln_count > 0:
            log_warning(f"Vulnerability breakdown: CRITICAL: {sev_counts['CRITICAL']}, HIGH: {sev_counts['HIGH']}, MEDIUM: {sev_counts['MEDIUM']}, LOW: {sev_counts['LOW']}")

        return {
            "summary": {
                "total_tests": total,
                "passed": pass_count,
                "vulnerable": vuln_count,
                "security_score_percent": score_percent,
                "severity_breakdown": sev_counts,
            },
            "category_scores": cat_summary,
            "test_results": results,
        }
