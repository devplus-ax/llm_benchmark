"""
Parser for OWASP LLM Top 10 and OWASP Agentic Security Community Test Cases.
"""

from dataclasses import dataclass, field
import glob
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Union


@dataclass
class OwaspTestCase:
    test_id: str
    title: str
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    attack_type: str
    strategy: str
    prompt: str
    expected_behavior: str
    vulnerability_indicators: List[str] = field(default_factory=list)
    owasp_reference: str = ""
    source_file: str = ""

    def to_dict(self) -> Dict:
        return {
            "test_id": self.test_id,
            "title": self.title,
            "category": self.category,
            "severity": self.severity,
            "attack_type": self.attack_type,
            "strategy": self.strategy,
            "prompt": self.prompt,
            "expected_behavior": self.expected_behavior,
            "vulnerability_indicators": self.vulnerability_indicators,
            "owasp_reference": self.owasp_reference,
            "source_file": self.source_file,
        }


def parse_markdown_file(file_path: Path) -> List[OwaspTestCase]:
    """Parse an individual OWASP test markdown file into structured test cases."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    # Split content by test case headers: "### <test_id>: <title>"
    chunks = re.split(r'(?m)^###\s+([A-Za-z0-9_-]+):\s*(.+)$', content)
    test_cases: List[OwaspTestCase] = []

    # chunks[0] is preamble before first test
    # chunks[1] is test_id, chunks[2] is title, chunks[3] is body, chunks[4] is test_id...
    for i in range(1, len(chunks), 3):
        test_id = chunks[i].strip()
        title = chunks[i + 1].strip()
        body = chunks[i + 2]

        # Extract Category and Severity
        category = "General"
        severity = "HIGH"
        cat_match = re.search(r'\*\*Category:\*\*\s*([^|\n]+)', body)
        if cat_match:
            category = cat_match.group(1).strip()
        sev_match = re.search(r'\*\*Severity:\*\*\s*([A-Za-z]+)', body)
        if sev_match:
            severity = sev_match.group(1).strip().upper()

        # Extract Attack Type and Strategy
        attack_type = "DIRECT"
        strategy = "BASIC"
        atk_match = re.search(r'\*\*Attack Type:\*\*\s*([^|\n]+)', body)
        if atk_match:
            attack_type = atk_match.group(1).strip()
        strat_match = re.search(r'\*\*Strategy:\*\*\s*([A-Za-z]+)', body)
        if strat_match:
            strategy = strat_match.group(1).strip()

        # Extract Test Prompt (enclosed in code fence)
        prompt = ""
        prompt_match = re.search(r'\*\*Test Prompt:\*\*\s*```(?:\w+)?\n?(.*?)```', body, re.DOTALL)
        if prompt_match:
            prompt = prompt_match.group(1).strip()

        # Extract Expected Behavior
        expected_behavior = ""
        exp_match = re.search(r'\*\*Expected Behavior:\*\*\s*(.*?)(?=\*\*Vulnerability Indicators:\*\*|\n---|\Z)', body, re.DOTALL)
        if exp_match:
            expected_behavior = exp_match.group(1).strip()

        # Extract Vulnerability Indicators
        vulnerability_indicators = []
        vuln_match = re.search(r'\*\*Vulnerability Indicators:\*\*\s*(.*?)(?=\*\*OWASP Reference:\*\*|\n---|\Z)', body, re.DOTALL)
        if vuln_match:
            raw_lines = vuln_match.group(1).strip().splitlines()
            for line in raw_lines:
                cleaned = line.strip().lstrip("-* ").strip()
                if cleaned:
                    vulnerability_indicators.append(cleaned)

        # Extract Reference link
        owasp_reference = ""
        ref_match = re.search(r'\*\*OWASP Reference:\*\*\s*(\S+)', body)
        if ref_match:
            owasp_reference = ref_match.group(1).strip()

        if prompt:
            test_cases.append(
                OwaspTestCase(
                    test_id=test_id,
                    title=title,
                    category=category,
                    severity=severity,
                    attack_type=attack_type,
                    strategy=strategy,
                    prompt=prompt,
                    expected_behavior=expected_behavior,
                    vulnerability_indicators=vulnerability_indicators,
                    owasp_reference=owasp_reference,
                    source_file=file_path.name,
                )
            )

    return test_cases


def load_all_owasp_tests(
    tests_dir: Union[str, Path],
    categories: Optional[List[str]] = None,
    include_agentic: bool = True,
    max_tests: Optional[int] = None,
) -> List[OwaspTestCase]:
    """Scan and parse all OWASP test markdown files within directory."""
    base_dir = Path(tests_dir)
    if not base_dir.is_dir():
        # Check standard relative path
        alt_dir = Path("owasp-llm-security-community-tests/tests")
        if alt_dir.is_dir():
            base_dir = alt_dir
        else:
            return []

    md_files = list(base_dir.glob("*.md"))
    if include_agentic:
        agentic_dir = base_dir / "agentic"
        if agentic_dir.is_dir():
            md_files.extend(list(agentic_dir.glob("*.md")))

    all_cases: List[OwaspTestCase] = []
    for md_file in sorted(md_files):
        all_cases.extend(parse_markdown_file(md_file))

    # Filter categories if specified (and not 'all')
    if categories and "all" not in [c.lower() for c in categories]:
        norm_cats = [c.lower().strip() for c in categories]
        filtered = []
        for tc in all_cases:
            tc_prefix = tc.test_id.split("-")[0].lower()
            tc_cat = tc.category.lower()
            if any(nc in tc_prefix or nc in tc_cat for nc in norm_cats):
                filtered.append(tc)
        all_cases = filtered

    if max_tests is not None and max_tests > 0:
        all_cases = all_cases[:max_tests]

    return all_cases
