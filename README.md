# Consolidated LLM Benchmark & Security Suite

A unified framework consolidating **Serving Performance Benchmarks** (vLLM compatible), **OWASP Top 10 Security & Prompt Injection Testing**, and **LLM-Attacks Adversarial Jailbreak Evaluations (AdvBench & GCG)** into a single CLI runner with interactive **JSON** and **HTML** reporting.

---

## 🌟 Key Capabilities

| Domain | Integrated Repo / Source | What It Tests & Measures |
| :--- | :--- | :--- |
| **Serving Benchmark** | `vllm/benchmarks/benchmark_serving.py` | • Time To First Token (TTFT - P50, P75, P90, P95, P99)<br>• Time Per Output Token (TPOT)<br>• Inter-Token Latency (ITL)<br>• Request Throughput (req/s) & Output Token Throughput (tok/s)<br>• Concurrency & Poisson request rate pacing (`--request-rate`)<br>• Datasets: ShareGPT, Random synthetic, Sonnet, or Custom |
| **OWASP Security** | `owasp-llm-security-community-tests` | • 30+ Automated community tests (LLM01-LLM10 & ASI01-ASI10 Agentic)<br>• Direct Prompt Injection & Role Reversal attacks<br>• System prompt leakage detection<br>• Sensitive data disclosure & Unauthorized privilege escalation<br>• Automatic refusal & vulnerability indicator rule engine |
| **Adversarial / Jailbreak** | `llm-attacks` | • AdvBench harmful behaviors dataset evaluation (`520+` behaviors)<br>• Attack Success Rate (ASR %) & Jailbreak Resistance Score %<br>• GCG adversarial affixes & jailbreak template testing<br>• Optional execution wrapper for whitebox GCG bash scripts |
| **Consolidated Reports** | Native HTML & JSON Engine | • Executive Summary Scorecard with Security Health Score %<br>• Interactive responsive HTML dashboard with SVG charts<br>• Searchable & filterable vulnerability tables (Critical, High, Med, Low)<br>• Standalone single-file HTML (opens offline in any browser) |

---

## 🚀 Quick Start

### 1. Run Consolidated Suite (Benchmark + OWASP + LLM-Attacks)
```bash
# Run against a live vLLM or OpenAI-compatible server on localhost:8000
python run_eval.py \
    --host localhost \
    --port 8000 \
    --model google/gemma-4-31B-it \
    --suite all \
    --request-rate 5.0 \
    --num-prompts 50
```

### 2. Auto-Discover & Benchmark All Available Models (`--model auto`)
Query the LLM endpoint (e.g. `GET /v1/models`) to automatically discover all models hosted on the server, evaluate each one, and generate a side-by-side comparative dashboard:
```bash
python run_eval.py --host localhost --port 8000 --model auto --suite all
```

### 3. Run Using a Configuration File (YAML or JSON)
```bash
# Using YAML config
python run_eval.py --config config.example.yaml

# Using JSON config
python run_eval.py --config config.example.json

# Override any config file setting from the CLI
python run_eval.py --config config.example.yaml --request-rate 10.0 --num-prompts 100
```

### 3. Run Benchmark Serving (Drop-in vLLM Compatibility)
Matches the exact `benchmark_serving.py` syntax from vLLM:
```bash
python benchmark_serving.py \
    --backend vllm \
    --host localhost \
    --port 8000 \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --model google/gemma-4-31B-it \
    --request-rate 5.0 \
    --num-prompts 300
```

*(Optional: Add `--with-security` to run OWASP and AdvBench security tests in the same run!)*

### 4. Run Security Tests Only (Against Custom or OpenAI API)
```bash
# Custom API endpoint (e.g. https://your-ai-api.com/chat)
python run_eval.py \
    --suite security \
    --endpoint-url https://your-ai-api.com/chat \
    --api-type custom \
    --custom-payload-field prompt \
    --api-key "YOUR_KEY_HERE"
```

### 5. Offline Dry-Run / Mock Mode (Test without GPU)
Validate the entire pipeline, test cases, and report generation instantly:
```bash
python run_eval.py --mock --suite all --num-prompts 10 --advbench-samples 5
```

---

## ⚙️ CLI Arguments Reference

### Suite Selection
* `--suite {all,benchmark,security,owasp,llm-attacks}`: Select which evaluation modules to execute (Default: `all`).
* `--config <path>`: Path to YAML or JSON config file.
* `--mock`, `--dry-run`: Run in simulated mock mode without active LLM server.

### Endpoint Settings
* `--host <host>`: Server host (Default: `localhost`).
* `--port <port>`: Server port (Default: `8000`).
* `--protocol {http,https}`: Protocol (Default: `http`).
* `--endpoint-url <url>`: Explicit URL override (e.g. `https://api.openai.com/v1/chat/completions` or `https://your-ai.com/chat`).
* `--model <name>`: Model identifier (e.g. `google/gemma-4-31B-it`). Pass `auto` to fetch models from `/v1/models` and evaluate all.
* `--api-type {vllm,openai,custom}`: Payload structure (Default: `vllm`).
* `--custom-payload-field <field>`: Key for custom endpoints (`prompt`, `message`, `messages`).
* `--api-key <token>`: Bearer authorization token.

### Serving Benchmark Settings
* `--backend {vllm,openai,custom}`: Backend type.
* `--dataset-name {sharegpt,random,sonnet,custom}`: Dataset source (Default: `random`).
* `--dataset-path <path>`: Path to ShareGPT or custom dataset.
* `--request-rate <float>`: Target queries per second (use `inf` for unbounded burst).
* `--num-prompts <int>`: Total number of benchmark prompts (Default: `50`).
* `--max-concurrency <int>`: Max simultaneous in-flight requests (Default: `16`).
* `--input-len <int>`: Token length for synthetic inputs (Default: `128`).
* `--output-len <int>`: Token limit for outputs (Default: `128`).
* `--no-stream`: Disable SSE streaming measurement.

### OWASP Security Settings
* `--owasp-dir <path>`: Directory containing test markdown files (Default: `owasp-llm-security-community-tests/tests`).
* `--owasp-categories <cat ...>`: Specific categories to run (e.g. `LLM01 LLM07 ASI01`) or `all`.
* `--no-agentic`: Exclude agentic tests (`ASI01-ASI10`).
* `--max-owasp-tests <int>`: Limit number of tests executed.
* `--system-prompt <str>`: Custom system prompt to evaluate/protect.

### LLM-Attacks & AdvBench Settings
* `--advbench-path <path>`: Path to `harmful_behaviors.csv` (Default: `llm-attacks/data/advbench/harmful_behaviors.csv`).
* `--advbench-samples <int>`: Number of harmful behaviors to evaluate (Default: `20`).
* `--no-jailbreaks`: Disable adversarial suffix and template variations.
* `--run-gcg-script`: Execute local whitebox GCG optimization bash script (`run_gcg_individual.sh`).

### Output & Reporting
* `--output-dir <path>`: Output directory for reports (Default: `reports`).
* `--report-name <name>`: Custom prefix for saved report files.
* `--format {json,html,both}`: Report formats to generate (Default: `both`).
* `--report-title <str>`: Custom title displayed in the HTML dashboard.

---

## 📊 Sample Output & Dashboard

Reports are saved to `reports/`:
* `reports/llm_eval_<model>_<timestamp>.json`: Detailed machine-readable JSON containing full metrics, percentiles, prompts, and evaluation traces.
* `reports/llm_eval_<model>_<timestamp>.html`: Interactive HTML dashboard containing:
  1. **Executive Overview**: High-level scorecards, overall security health score, throughput, latency percentiles, and vulnerability severity breakdown.
  2. **vLLM Benchmark**: TTFT, TPOT, ITL, latency percentiles table (P50 to P99), and throughput statistics.
  3. **OWASP Security Audit**: Interactive filterable and searchable test case cards showing prompt, response, detected indicators, and OWASP remediation links.
  4. **Adversarial / AdvBench**: Harmful behaviors tested, Attack Success Rate (ASR %), and safety refusals.
  5. **Raw JSON**: View and export report data.
