"""
Wrapper for executing whitebox GCG (Greedy Coordinate Gradient) scripts in llm-attacks.
"""

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional

from llm_eval.config import LlmAttacksConfig
from llm_eval.utils import log_header, log_info, log_warning


class GcgScriptRunner:
    """Executes or wraps llm-attacks GCG experiments."""

    def __init__(self, attacks_cfg: LlmAttacksConfig):
        self.attacks_cfg = attacks_cfg

    def run(self) -> Dict[str, Any]:
        """Execute the GCG script or check results."""
        log_header("Executing LLM-Attacks GCG Whitebox Optimization")
        
        script_dir = Path("llm-attacks/experiments/launch_scripts")
        if not script_dir.is_dir():
            log_warning(f"GCG launch_scripts directory not found at {script_dir}")
            return {"status": "SKIPPED", "message": f"Directory not found: {script_dir}"}

        model = self.attacks_cfg.gcg_model
        setup = self.attacks_cfg.gcg_setup
        script_path = script_dir / "run_gcg_individual.sh"

        log_info(f"Target script: {script_path} | Model: {model} | Setup: {setup}")

        # Check if bash is available
        bash_cmd = "bash"
        try:
            cmd = [bash_cmd, "run_gcg_individual.sh", model, setup]
            proc = subprocess.run(
                cmd,
                cwd=str(script_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "status": "COMPLETED" if proc.returncode == 0 else "FAILED",
                "return_code": proc.returncode,
                "stdout": proc.stdout[-1000:],
                "stderr": proc.stderr[-1000:],
            }
        except FileNotFoundError:
            log_warning("bash executable not found in PATH on this system. GCG bash script requires a bash shell environment.")
            return {
                "status": "SKIPPED",
                "message": "Bash executable not found. On Windows, run within Git Bash, WSL, or execute python ../main.py directly.",
            }
        except subprocess.TimeoutExpired:
            log_warning("GCG execution timed out (over 120s limit). GCG runs are typically multi-hour GPU jobs.")
            return {
                "status": "TIMEOUT",
                "message": "Optimization step exceeded time limit.",
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
