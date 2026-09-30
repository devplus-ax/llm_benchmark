"""
Interactive HTML Report Generator for consolidated LLM benchmark and security evaluations.
Self-contained, responsive dashboard with SVG visualizations, filterable test cases, and metric scorecards.
"""

import json
from pathlib import Path
from typing import Any, Dict

from llm_eval.utils import SafeJSONEncoder, log_success


def generate_html_report(report_data: Dict[str, Any], output_path: Path) -> Path:
    """Generate a self-contained modern HTML dashboard report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = report_data.get("metadata", {})
    exec_summary = report_data.get("executive_summary", {})
    bench_data = report_data.get("benchmark", {})
    owasp_data = report_data.get("owasp", {})
    adv_data = report_data.get("advbench", {})
    gcg_data = report_data.get("gcg", {})

    # Extract score values
    sec_score = exec_summary.get("overall_security_score", 100.0)
    owasp_score = owasp_data.get("summary", {}).get("security_score_percent", 100.0)
    adv_asr = adv_data.get("summary", {}).get("attack_success_rate_percent", 0.0)
    adv_resistance = adv_data.get("summary", {}).get("safety_resistance_percent", 100.0)

    # Benchmark metrics
    bench_summary = bench_data.get("summary", {})
    bench_metrics = bench_data.get("metrics", {})
    ttft_stats = bench_metrics.get("ttft_ms", {})
    tpot_stats = bench_metrics.get("tpot_ms", {})
    e2e_stats = bench_metrics.get("e2e_latency_ms", {})

    # Severity counts
    sev_counts = owasp_data.get("summary", {}).get("severity_breakdown", {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0})
    total_vulns = sum(sev_counts.values())

    # Category breakdown
    cat_scores = owasp_data.get("category_scores", {})

    # OWASP test results
    owasp_tests = owasp_data.get("test_results", [])

    # AdvBench test results
    adv_tests = adv_data.get("test_results", [])

    # Serialized JSON for embed
    json_str = json.dumps(report_data, cls=SafeJSONEncoder, indent=2)

    # Calculate gauge circle dashoffset (circumference = 2 * PI * 40 ~= 251.2)
    circumference = 251.2
    dash_offset = circumference - (circumference * sec_score / 100.0)
    gauge_color = "#10b981" if sec_score >= 80 else ("#f59e0b" if sec_score >= 50 else "#ef4444")

    # Render category progress bars
    cat_rows_html = ""
    for cat_name, cat_info in cat_scores.items():
        rate = cat_info.get("pass_rate_percent", 0.0)
        p_count = cat_info.get("passed", 0)
        v_count = cat_info.get("vulnerable", 0)
        t_count = cat_info.get("total", 0)
        bar_color = "#10b981" if rate >= 80 else ("#f59e0b" if rate >= 50 else "#ef4444")
        cat_rows_html += f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
                <span style="font-weight: 500;">{cat_name}</span>
                <span>{p_count}/{t_count} Passed ({rate}%)</span>
            </div>
            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: {bar_color}; width: {rate}%; height: 100%;"></div>
            </div>
        </div>
        """

    # Render OWASP test cards
    owasp_cards_html = ""
    for tc in owasp_tests:
        status = tc.get("status", "PASS")
        is_vuln = tc.get("is_vulnerable", False)
        badge_cls = "badge-fail" if is_vuln else "badge-pass"
        sev = tc.get("severity", "MEDIUM")
        sev_cls = f"sev-{sev.lower()}"
        reasons_html = ""
        if tc.get("vulnerability_reasons"):
            reasons_html = "<div class='vuln-box'><strong>Vulnerability Triggered:</strong><ul>"
            for r in tc["vulnerability_reasons"]:
                reasons_html += f"<li>{r}</li>"
            reasons_html += "</ul></div>"

        # Escape HTML in prompts and responses
        prompt_esc = tc.get("prompt", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        resp_esc = tc.get("response_text", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        exp_esc = tc.get("expected_behavior", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        owasp_cards_html += f"""
        <div class="test-card" data-status="{status}" data-severity="{sev}" data-category="{tc.get('category', '')}">
            <div class="test-header" onclick="toggleCard(this)">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="badge {badge_cls}">{status}</span>
                    <span class="badge {sev_cls}">{sev}</span>
                    <strong>{tc.get('test_id')}</strong>: {tc.get('title')}
                </div>
                <div style="display: flex; align-items: center; gap: 15px; font-size: 13px; color: #94a3b8;">
                    <span>{tc.get('category')}</span>
                    <span>{tc.get('latency_ms', 0)} ms</span>
                    <span class="toggle-icon">▼</span>
                </div>
            </div>
            <div class="test-body" style="display: none;">
                {reasons_html}
                <div class="section-label">Test Prompt:</div>
                <pre class="code-block">{prompt_esc}</pre>
                <div class="section-label">Model Response:</div>
                <pre class="code-block">{resp_esc if resp_esc else '(No response received)'}</pre>
                <div class="section-label">Expected Safe Behavior:</div>
                <div class="info-text">{exp_esc}</div>
                {f'<div class="section-label"><a href="{tc.get("owasp_reference")}" target="_blank" style="color: #38bdf8;">OWASP Reference Guide &rarr;</a></div>' if tc.get('owasp_reference') else ''}
            </div>
        </div>
        """

    # Render AdvBench cards
    adv_rows_html = ""
    for r in adv_tests:
        status = r.get("status", "BLOCKED")
        bypassed = r.get("bypassed", False)
        badge_cls = "badge-fail" if bypassed else "badge-pass"
        goal_esc = r.get("goal", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        resp_esc = r.get("response_text", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        adv_rows_html += f"""
        <tr>
            <td style="white-space: nowrap;"><strong>{r.get('id')}</strong></td>
            <td><span class="badge {badge_cls}">{status}</span></td>
            <td><code>{r.get('attack_mode')}</code></td>
            <td style="max-width: 320px;">{goal_esc}</td>
            <td style="max-width: 380px; font-size: 12px; color: #cbd5e1;"><div class="truncate-text">{resp_esc[:200]}...</div></td>
            <td>{r.get('latency_ms')} ms</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_data.get('config', {}).get('report', {}).get('title', 'Consolidated LLM Benchmark & Security Report')}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1300px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        h1 {{ font-size: 26px; font-weight: 700; color: #fff; }}
        .meta-tags {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 13px; color: var(--text-secondary); }}
        .meta-tag {{ background: var(--bg-secondary); padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); }}
        
        /* Navigation Tabs */
        .tabs {{ display: flex; gap: 8px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }}
        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 600;
            padding: 10px 18px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{ color: var(--text-primary); }}
        .tab-btn.active {{
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
            background: rgba(56, 189, 248, 0.05);
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .kpi-title {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-secondary); margin-bottom: 8px; }}
        .kpi-value {{ font-size: 28px; font-weight: 700; color: #fff; display: flex; align-items: baseline; gap: 6px; }}
        .kpi-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 6px; }}

        /* Radial Gauge */
        .gauge-wrap {{ display: flex; align-items: center; justify-content: center; position: relative; width: 100px; height: 100px; }}
        .gauge-svg {{ transform: rotate(-90deg); }}
        .gauge-text {{ position: absolute; font-size: 20px; font-weight: 700; text-anchor: middle; dominant-baseline: middle; }}

        /* Badges */
        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-pass {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; }}
        .badge-fail {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #dc2626; }}
        .sev-critical {{ background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid #ef4444; }}
        .sev-high {{ background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #ea580c; }}
        .sev-medium {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }}
        .sev-low {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; }}

        /* Two column layout */
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
        @media(max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
        }}
        .card-title {{ font-size: 16px; font-weight: 700; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }}

        /* Table */
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: rgba(15, 23, 42, 0.6); color: var(--text-secondary); font-weight: 600; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}

        /* Interactive Filter Bar */
        .filter-bar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; align-items: center; }}
        .filter-input {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: #fff;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            flex-grow: 1;
            max-width: 320px;
        }}
        .filter-btn {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }}
        .filter-btn.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}

        /* Test Cards */
        .test-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 10px;
            overflow: hidden;
        }}
        .test-header {{
            padding: 14px 16px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(30, 41, 59, 0.7);
        }}
        .test-header:hover {{ background: rgba(51, 65, 85, 0.5); }}
        .test-body {{ padding: 16px; border-top: 1px solid var(--border); font-size: 13px; }}
        .section-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin: 12px 0 4px 0; }}
        .code-block {{
            background: #0f172a;
            border: 1px solid #1e293b;
            padding: 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-break: break-all;
            color: #e2e8f0;
            max-height: 200px;
            overflow-y: auto;
        }}
        .info-text {{ background: rgba(56, 189, 248, 0.05); padding: 8px 12px; border-radius: 6px; border-left: 3px solid var(--accent); }}
        .vuln-box {{ background: rgba(239, 68, 68, 0.1); border-left: 3px solid var(--danger); padding: 10px 14px; border-radius: 4px; margin-bottom: 12px; color: #fca5a5; }}

        .btn {{
            background: var(--accent);
            color: #000;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn:hover {{ opacity: 0.9; }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>{report_data.get('config', {}).get('report', {}).get('title', 'Consolidated LLM Benchmark & Security Report')}</h1>
            <div class="meta-tags">
                <span class="meta-tag"><strong>Model:</strong> {metadata.get('model', 'default')}</span>
                <span class="meta-tag"><strong>Target Endpoint:</strong> {metadata.get('endpoint_url', 'localhost')}</span>
                <span class="meta-tag"><strong>Suite:</strong> {metadata.get('suite', 'all').upper()}</span>
                <span class="meta-tag"><strong>Timestamp:</strong> {metadata.get('timestamp', '')[:19]}</span>
                <span class="meta-tag"><strong>OS:</strong> {metadata.get('system_info', {}).get('os', 'Unknown')}</span>
            </div>
        </div>
        <div>
            <button class="btn" onclick="exportJson()">Export JSON</button>
        </div>
    </header>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('overview')">Executive Overview</button>
        <button class="tab-btn" onclick="switchTab('benchmark')">vLLM Benchmark ({bench_summary.get('total_requests', 0)} reqs)</button>
        <button class="tab-btn" onclick="switchTab('owasp')">OWASP Security ({len(owasp_tests)} tests)</button>
        <button class="tab-btn" onclick="switchTab('advbench')">LLM-Attacks / AdvBench ({len(adv_tests)} behaviors)</button>
        <button class="tab-btn" onclick="switchTab('raw')">Raw JSON & Config</button>
    </div>

    <!-- TAB 1: EXECUTIVE OVERVIEW -->
    <div id="tab-overview" class="tab-content active">
        <div class="kpi-grid">
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">Overall Security Score</div>
                    <div class="kpi-value" style="color: {gauge_color};">{sec_score}%</div>
                </div>
                <div class="kpi-sub">OWASP: {owasp_score}% | Resistance: {adv_resistance}%</div>
            </div>

            <div class="kpi-card">
                <div>
                    <div class="kpi-title">Vulnerabilities Found</div>
                    <div class="kpi-value" style="color: {'#ef4444' if total_vulns > 0 else '#10b981'};">{total_vulns}</div>
                </div>
                <div class="kpi-sub">Crit: {sev_counts['CRITICAL']} | High: {sev_counts['HIGH']} | Med: {sev_counts['MEDIUM']}</div>
            </div>

            <div class="kpi-card">
                <div>
                    <div class="kpi-title">Serving Throughput</div>
                    <div class="kpi-value">{bench_summary.get('output_token_throughput_tok_per_s', 0)} <span style="font-size: 14px; font-weight: normal; color: var(--text-secondary);">tok/s</span></div>
                </div>
                <div class="kpi-sub">{bench_summary.get('request_throughput_req_per_s', 0)} req/s ({bench_summary.get('successful_requests', 0)} completed)</div>
            </div>

            <div class="kpi-card">
                <div>
                    <div class="kpi-title">TTFT (Time To First Token)</div>
                    <div class="kpi-value">{ttft_stats.get('p50', 0)} <span style="font-size: 14px; font-weight: normal; color: var(--text-secondary);">ms</span></div>
                </div>
                <div class="kpi-sub">P95: {ttft_stats.get('p95', 0)} ms | P99: {ttft_stats.get('p99', 0)} ms</div>
            </div>

            <div class="kpi-card">
                <div>
                    <div class="kpi-title">TPOT (Per Output Token)</div>
                    <div class="kpi-value">{tpot_stats.get('p50', 0)} <span style="font-size: 14px; font-weight: normal; color: var(--text-secondary);">ms</span></div>
                </div>
                <div class="kpi-sub">P95: {tpot_stats.get('p95', 0)} ms | ITL P50: {bench_metrics.get('itl_ms', {}).get('p50', 0)} ms</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <div class="card-title">OWASP Category Security Posture</div>
                {cat_rows_html if cat_rows_html else '<p style="color: var(--text-secondary); font-size: 13px;">No OWASP category tests executed in this run.</p>'}
            </div>

            <div class="card">
                <div class="card-title">Adversarial Resistance & Jailbreak Defense</div>
                <div style="display: flex; align-items: center; justify-content: space-around; margin-top: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 32px; font-weight: 700; color: #10b981;">{adv_data.get('summary', {}).get('blocked', 0)}</div>
                        <div style="font-size: 13px; color: var(--text-secondary);">Attacks Blocked (Refused)</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 32px; font-weight: 700; color: {'#ef4444' if adv_data.get('summary', {}).get('bypassed', 0) > 0 else '#10b981'};">{adv_data.get('summary', {}).get('bypassed', 0)}</div>
                        <div style="font-size: 13px; color: var(--text-secondary);">Attacks Bypassed (ASR: {adv_asr}%)</div>
                    </div>
                </div>
                <div style="margin-top: 25px; padding: 12px; background: rgba(15, 23, 42, 0.6); border-radius: 6px; font-size: 13px;">
                    <strong>Security Recommendation:</strong>
                    {'Review input sanitization and system prompt hardening for identified vulnerability indicators.' if total_vulns > 0 or adv_asr > 0 else 'Model exhibits robust prompt injection defenses and compliant safety refusals under evaluated conditions.'}
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 2: BENCHMARK -->
    <div id="tab-benchmark" class="tab-content">
        <div class="card" style="margin-bottom: 20px;">
            <div class="card-title">Serving Latency Percentiles (ms)</div>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Min</th>
                        <th>Mean</th>
                        <th>P50 (Median)</th>
                        <th>P75</th>
                        <th>P90</th>
                        <th>P95</th>
                        <th>P99</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Time To First Token (TTFT)</strong></td>
                        <td>{ttft_stats.get('min', 0)} ms</td>
                        <td>{ttft_stats.get('mean', 0)} ms</td>
                        <td><strong style="color: var(--accent);">{ttft_stats.get('p50', 0)} ms</strong></td>
                        <td>{ttft_stats.get('p75', 0)} ms</td>
                        <td>{ttft_stats.get('p90', 0)} ms</td>
                        <td>{ttft_stats.get('p95', 0)} ms</td>
                        <td>{ttft_stats.get('p99', 0)} ms</td>
                        <td>{ttft_stats.get('max', 0)} ms</td>
                    </tr>
                    <tr>
                        <td><strong>Time Per Output Token (TPOT)</strong></td>
                        <td>{tpot_stats.get('min', 0)} ms</td>
                        <td>{tpot_stats.get('mean', 0)} ms</td>
                        <td><strong style="color: var(--accent);">{tpot_stats.get('p50', 0)} ms</strong></td>
                        <td>{tpot_stats.get('p75', 0)} ms</td>
                        <td>{tpot_stats.get('p90', 0)} ms</td>
                        <td>{tpot_stats.get('p95', 0)} ms</td>
                        <td>{tpot_stats.get('p99', 0)} ms</td>
                        <td>{tpot_stats.get('max', 0)} ms</td>
                    </tr>
                    <tr>
                        <td><strong>End-to-End Latency</strong></td>
                        <td>{e2e_stats.get('min', 0)} ms</td>
                        <td>{e2e_stats.get('mean', 0)} ms</td>
                        <td><strong style="color: var(--accent);">{e2e_stats.get('p50', 0)} ms</strong></td>
                        <td>{e2e_stats.get('p75', 0)} ms</td>
                        <td>{e2e_stats.get('p90', 0)} ms</td>
                        <td>{e2e_stats.get('p95', 0)} ms</td>
                        <td>{e2e_stats.get('p99', 0)} ms</td>
                        <td>{e2e_stats.get('max', 0)} ms</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="card-title">Throughput & Load Configuration</div>
            <div class="grid-2">
                <div>
                    <table>
                        <tr><td><strong>Total Requests Sent:</strong></td><td>{bench_summary.get('total_requests', 0)}</td></tr>
                        <tr><td><strong>Successful Inferences:</strong></td><td>{bench_summary.get('successful_requests', 0)}</td></tr>
                        <tr><td><strong>Failed Inferences:</strong></td><td>{bench_summary.get('failed_requests', 0)} ({bench_summary.get('error_rate_percent', 0)}%)</td></tr>
                        <tr><td><strong>Total Test Duration:</strong></td><td>{bench_summary.get('duration_seconds', 0)}s</td></tr>
                    </table>
                </div>
                <div>
                    <table>
                        <tr><td><strong>Request Rate:</strong></td><td>{bench_summary.get('request_throughput_req_per_s', 0)} req/s</td></tr>
                        <tr><td><strong>Output Token Rate:</strong></td><td>{bench_summary.get('output_token_throughput_tok_per_s', 0)} tok/s</td></tr>
                        <tr><td><strong>Total Input Tokens:</strong></td><td>{bench_summary.get('total_input_tokens', 0)}</td></tr>
                        <tr><td><strong>Total Output Tokens:</strong></td><td>{bench_summary.get('total_output_tokens', 0)}</td></tr>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 3: OWASP SECURITY -->
    <div id="tab-owasp" class="tab-content">
        <div class="filter-bar">
            <input type="text" id="owasp-search" class="filter-input" placeholder="Search prompt, ID, title, or category..." oninput="filterOwasp()">
            <button class="filter-btn active" onclick="setOwaspFilter('all', this)">All</button>
            <button class="filter-btn" onclick="setOwaspFilter('VULNERABLE', this)">Vulnerable Only ({total_vulns})</button>
            <button class="filter-btn" onclick="setOwaspFilter('PASS', this)">Passed Only</button>
            <button class="filter-btn" onclick="setOwaspFilter('CRITICAL', this)">Critical</button>
            <button class="filter-btn" onclick="setOwaspFilter('HIGH', this)">High</button>
        </div>

        <div id="owasp-cards-container">
            {owasp_cards_html if owasp_cards_html else '<p style="color: var(--text-secondary);">No OWASP tests run in this session.</p>'}
        </div>
    </div>

    <!-- TAB 4: ADVBENCH -->
    <div id="tab-advbench" class="tab-content">
        <div class="card">
            <div class="card-title">LLM-Attacks AdvBench Harmful Behaviors ({len(adv_tests)} Tested)</div>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Status</th>
                        <th>Attack Mode</th>
                        <th>Harmful Goal</th>
                        <th>Model Response Snippet</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {adv_rows_html if adv_rows_html else '<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">No AdvBench tests run in this session.</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>

    <!-- TAB 5: RAW JSON -->
    <div id="tab-raw" class="tab-content">
        <div class="card">
            <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
                <span>Full Evaluation JSON</span>
                <button class="btn" style="padding: 4px 10px; font-size: 11px;" onclick="copyJson()">Copy to Clipboard</button>
            </div>
            <pre class="code-block" id="raw-json-block" style="max-height: 600px;">{json_str}</pre>
        </div>
    </div>
</div>

<script>
    function switchTab(tabId) {{
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        event.target.classList.add('active');
        document.getElementById('tab-' + tabId).classList.add('active');
    }}

    function toggleCard(headerEl) {{
        const body = headerEl.nextElementSibling;
        const icon = headerEl.querySelector('.toggle-icon');
        if (body.style.display === 'none' || !body.style.display) {{
            body.style.display = 'block';
            icon.textContent = '▲';
        }} else {{
            body.style.display = 'none';
            icon.textContent = '▼';
        }}
    }}

    let currentStatusFilter = 'all';
    function setOwaspFilter(filterVal, btn) {{
        currentStatusFilter = filterVal;
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        filterOwasp();
    }}

    function filterOwasp() {{
        const search = document.getElementById('owasp-search').value.toLowerCase();
        const cards = document.querySelectorAll('.test-card');

        cards.forEach(card => {{
            const status = card.getAttribute('data-status');
            const severity = card.getAttribute('data-severity');
            const text = card.textContent.toLowerCase();

            let matchesFilter = true;
            if (currentStatusFilter === 'VULNERABLE' && status !== 'VULNERABLE') matchesFilter = false;
            if (currentStatusFilter === 'PASS' && status !== 'PASS') matchesFilter = false;
            if (currentStatusFilter === 'CRITICAL' && severity !== 'CRITICAL') matchesFilter = false;
            if (currentStatusFilter === 'HIGH' && severity !== 'HIGH') matchesFilter = false;

            let matchesSearch = text.includes(search);
            card.style.display = (matchesFilter && matchesSearch) ? 'block' : 'none';
        }});
    }}

    function copyJson() {{
        const text = document.getElementById('raw-json-block').textContent;
        navigator.clipboard.writeText(text).then(() => alert('Report JSON copied to clipboard!'));
    }}

    function exportJson() {{
        const text = document.getElementById('raw-json-block').textContent;
        const blob = new Blob([text], {{ type: 'application/json' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'llm_eval_report.json';
        a.click();
    }}
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    log_success(f"HTML dashboard report saved to: {output_path}")
    return output_path


def generate_multi_model_html_report(multi_data: Dict[str, Any], output_path: Path) -> Path:
    """Generate a multi-model comparative dashboard report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models_data = multi_data.get("models", [])
    timestamp = multi_data.get("timestamp", "")
    endpoint_url = multi_data.get("endpoint_url", "")

    rows_html = ""
    for item in models_data:
        m_name = item.get("model", "unknown")
        exec_s = item.get("executive_summary", {})
        bm_s = exec_s.get("benchmark", {})
        sec_s = exec_s.get("security", {})
        html_file = item.get("html_file", "#")

        score = exec_s.get("overall_security_score", 100.0)
        score_color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 50 else "#ef4444")
        tok_s = bm_s.get("output_token_throughput_tok_per_s", 0.0)
        req_s = bm_s.get("request_throughput_req_per_s", 0.0)
        ttft_p50 = bm_s.get("ttft_p50_ms", 0.0)
        ttft_p95 = bm_s.get("ttft_p95_ms", 0.0)
        tpot_p50 = bm_s.get("tpot_p50_ms", 0.0)
        owasp_score = sec_s.get("owasp_score_percent", 100.0)
        owasp_vulns = sec_s.get("owasp_vulnerable", 0)
        adv_res = sec_s.get("advbench_resistance_percent", 100.0)
        adv_asr = sec_s.get("advbench_asr_percent", 0.0)

        rows_html += f"""
        <tr>
            <td><strong style="color: #38bdf8;">{m_name}</strong></td>
            <td><strong style="color: {score_color}; font-size: 15px;">{score}%</strong></td>
            <td><strong>{tok_s}</strong> tok/s <span style="font-size: 11px; color: #94a3b8;">({req_s} req/s)</span></td>
            <td>{ttft_p50} ms <span style="font-size: 11px; color: #94a3b8;">(P95: {ttft_p95}ms)</span></td>
            <td>{tpot_p50} ms</td>
            <td>{owasp_score}% <span style="font-size: 11px; color: {'#ef4444' if owasp_vulns > 0 else '#10b981'};">({owasp_vulns} vulns)</span></td>
            <td>{adv_res}% <span style="font-size: 11px; color: #94a3b8;">(ASR: {adv_asr}%)</span></td>
            <td><a href="{html_file}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 600;">View Report &rarr;</a></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Model LLM Evaluation & Security Comparison</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 30px;
        }}
        .container {{ max-width: 1300px; margin: 0 auto; }}
        header {{ margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
        h1 {{ font-size: 26px; color: #fff; margin-bottom: 6px; }}
        .meta {{ font-size: 13px; color: var(--text-secondary); }}
        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: rgba(15, 23, 42, 0.7); color: var(--text-secondary); font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>⚡ Multi-Model LLM Benchmark & Security Comparison</h1>
        <div class="meta">
            Target Endpoint: <strong>{endpoint_url}</strong> | Evaluated Models: <strong>{len(models_data)}</strong> | Date: <strong>{timestamp[:19]}</strong>
        </div>
    </header>

    <div class="card">
        <h2 style="font-size: 18px; margin-bottom: 16px;">Side-by-Side Model Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Overall Health</th>
                    <th>Throughput</th>
                    <th>TTFT (P50)</th>
                    <th>TPOT (P50)</th>
                    <th>OWASP Security</th>
                    <th>AdvBench Defense</th>
                    <th>Detailed Report</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    log_success(f"Multi-Model Comparison HTML dashboard saved to: {output_path}")
    return output_path

