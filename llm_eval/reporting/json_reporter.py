"""
JSON report generator for consolidated benchmark and security results.
"""

import json
from pathlib import Path
from typing import Any, Dict

from llm_eval.utils import SafeJSONEncoder, log_success


def save_json_report(report_data: Dict[str, Any], output_path: Path) -> Path:
    """Serialize evaluation results to a pretty-formatted JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, cls=SafeJSONEncoder, indent=2, ensure_ascii=False)
        
    log_success(f"JSON report saved to: {output_path}")
    return output_path
