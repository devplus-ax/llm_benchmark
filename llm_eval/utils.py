"""
Utility functions for statistical analysis, logging, and JSON encoding.
"""

import datetime
import json
import math
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Union


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling datetime, numpy types, and custom classes."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        # Check for numpy floats/ints without requiring hard numpy import
        type_name = type(obj).__name__
        if "float" in type_name.lower():
            return float(obj)
        if "int" in type_name.lower():
            return int(obj)
        if "ndarray" in type_name:
            return obj.tolist()
        return str(obj)


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate standard summary statistics and percentiles for a list of floats."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "std": 0.0,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def get_percentile(p: float) -> float:
        if n == 0:
            return 0.0
        idx = (p / 100.0) * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return float(sorted_vals[lower])
        weight = idx - lower
        return float(sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight)

    mean_val = sum(sorted_vals) / n
    variance = sum((x - mean_val) ** 2 for x in sorted_vals) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)

    return {
        "count": n,
        "mean": round(mean_val, 4),
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "median": round(get_percentile(50), 4),
        "p50": round(get_percentile(50), 4),
        "p75": round(get_percentile(75), 4),
        "p90": round(get_percentile(90), 4),
        "p95": round(get_percentile(95), 4),
        "p99": round(get_percentile(99), 4),
        "std": round(std_val, 4),
    }


def get_system_info() -> Dict[str, str]:
    """Collect host system metadata for reporting."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "python_version": platform.python_version(),
    }


class Colors:
    """ANSI color codes for terminal logging."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def log_header(title: str) -> None:
    """Print a styled section header in the terminal."""
    bar = "=" * 70
    print(f"\n{Colors.BOLD}{Colors.CYAN}{bar}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{bar}{Colors.RESET}\n")


def log_info(msg: str) -> None:
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {msg}")


def log_success(msg: str) -> None:
    print(f"{Colors.GREEN}[PASS]{Colors.RESET} {msg}")


def log_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")


def log_vuln(msg: str) -> None:
    print(f"{Colors.RED}[VULNERABLE]{Colors.RESET} {msg}")
