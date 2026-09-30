"""
Async HTTP client for querying LLM endpoints (vLLM, OpenAI-compatible, custom APIs)
with accurate Time-To-First-Token (TTFT), TPOT, and latency instrumentation.
"""

import asyncio
import json
import random
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from llm_eval.config import EndpointConfig


class LLMResponse:
    """Standardized response data structure containing performance metrics."""
    def __init__(
        self,
        prompt: str,
        text: str = "",
        success: bool = True,
        status_code: int = 200,
        error_msg: Optional[str] = None,
        latency_s: float = 0.0,
        ttft_s: Optional[float] = None,
        tpot_s: Optional[float] = None,
        inter_token_latencies: Optional[List[float]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        raw_response: Optional[Dict[str, Any]] = None,
    ):
        self.prompt = prompt
        self.text = text
        self.success = success
        self.status_code = status_code
        self.error_msg = error_msg
        self.latency_s = latency_s
        self.ttft_s = ttft_s
        self.tpot_s = tpot_s
        self.inter_token_latencies = inter_token_latencies or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.raw_response = raw_response or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "text": self.text,
            "success": self.success,
            "status_code": self.status_code,
            "error_msg": self.error_msg,
            "latency_s": round(self.latency_s, 4),
            "ttft_ms": round(self.ttft_s * 1000, 2) if self.ttft_s is not None else None,
            "tpot_ms": round(self.tpot_s * 1000, 2) if self.tpot_s is not None else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def estimate_tokens(text: str) -> int:
    """Heuristic token estimation (~4 characters per token / word boundaries)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, int(round((words * 1.3 + chars / 4.0) / 2.0)))


class AsyncLLMClient:
    """High-concurrency async client for communicating with vLLM, OpenAI, and custom AI endpoints."""

    def __init__(self, config: EndpointConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        if not self.config.mock_mode and AIOHTTP_AVAILABLE:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_models(self) -> List[str]:
        """Query /v1/models to discover available model IDs from the server."""
        if self.config.mock_mode:
            return ["google/gemma-4-31B-it", "meta-llama/Llama-3-8B-Instruct"]

        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp library is required to perform live HTTP requests.")

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        base = self.config.get_base_url().rstrip("/")
        endpoints_to_try = [
            f"{base}/v1/models",
            f"{base}/models",
        ]
        headers = self._build_headers()

        for url in endpoints_to_try:
            try:
                async with self._session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        model_ids: List[str] = []
                        if isinstance(data, dict):
                            items = data.get("data", []) or data.get("models", [])
                            if isinstance(items, list):
                                for item in items:
                                    if isinstance(item, dict):
                                        m_id = item.get("id") or item.get("name") or item.get("model")
                                        if m_id:
                                            model_ids.append(str(m_id))
                                    elif isinstance(item, str):
                                        model_ids.append(item)
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    m_id = item.get("id") or item.get("name") or item.get("model")
                                    if m_id:
                                        model_ids.append(str(m_id))
                                elif isinstance(item, str):
                                    model_ids.append(item)

                        if model_ids:
                            return model_ids
            except Exception:
                continue

        return []

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LLMEvalSuite/1.0",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.headers)
        return headers

    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """Format request payload according to API type."""
        if self.config.api_type in ("vllm", "openai"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            return payload
        else:
            # Custom API format (e.g. {"prompt": "..."} or {"message": "..."})
            field = self.config.custom_payload_field
            if field == "messages":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                return {"messages": messages}
            return {field: prompt}

    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
    ) -> LLMResponse:
        """Send prompt to target LLM endpoint and instrument performance metrics."""
        if self.config.mock_mode:
            return await self._mock_query(prompt, system_prompt, max_tokens, stream)

        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp library is required to perform live HTTP requests.")

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        url = self.config.get_chat_url()
        headers = self._build_headers()
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        input_tokens = estimate_tokens(prompt) + (estimate_tokens(system_prompt) if system_prompt else 0)
        start_time = time.perf_counter()

        try:
            if stream and self.config.api_type in ("vllm", "openai"):
                return await self._query_streaming(url, headers, payload, prompt, input_tokens, start_time)
            else:
                return await self._query_non_streaming(url, headers, payload, prompt, input_tokens, start_time)
        except asyncio.TimeoutError:
            latency = time.perf_counter() - start_time
            return LLMResponse(
                prompt=prompt,
                success=False,
                status_code=408,
                error_msg=f"Request timed out after {self.config.timeout_seconds}s",
                latency_s=latency,
                input_tokens=input_tokens,
            )
        except Exception as e:
            latency = time.perf_counter() - start_time
            return LLMResponse(
                prompt=prompt,
                success=False,
                status_code=500,
                error_msg=str(e),
                latency_s=latency,
                input_tokens=input_tokens,
            )

    async def _query_streaming(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        prompt: str,
        input_tokens: int,
        start_time: float,
    ) -> LLMResponse:
        """Process streaming SSE response for exact TTFT & TPOT measurement."""
        ttft_s: Optional[float] = None
        collected_chunks: List[str] = []
        chunk_timestamps: List[float] = []

        async with self._session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                err_text = await response.text()
                latency = time.perf_counter() - start_time
                return LLMResponse(
                    prompt=prompt,
                    success=False,
                    status_code=response.status,
                    error_msg=f"HTTP {response.status}: {err_text[:300]}",
                    latency_s=latency,
                    input_tokens=input_tokens,
                )

            async for line_bytes in response.content:
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                now = time.perf_counter()
                if ttft_s is None:
                    ttft_s = now - start_time

                chunk_timestamps.append(now)

                try:
                    chunk_json = json.loads(data_str)
                    choices = chunk_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected_chunks.append(content)
                except Exception:
                    # Ignore malformed chunk
                    pass

        end_time = time.perf_counter()
        total_latency = end_time - start_time
        full_text = "".join(collected_chunks)
        output_tokens = estimate_tokens(full_text)

        # Compute inter-token latencies and TPOT
        itl: List[float] = []
        for i in range(1, len(chunk_timestamps)):
            itl.append(chunk_timestamps[i] - chunk_timestamps[i - 1])

        tpot_s = None
        if len(chunk_timestamps) > 1 and ttft_s is not None:
            tpot_s = (end_time - (start_time + ttft_s)) / max(1, (len(chunk_timestamps) - 1))

        return LLMResponse(
            prompt=prompt,
            text=full_text,
            success=True,
            status_code=200,
            latency_s=total_latency,
            ttft_s=ttft_s or total_latency,
            tpot_s=tpot_s,
            inter_token_latencies=itl,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _query_non_streaming(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        prompt: str,
        input_tokens: int,
        start_time: float,
    ) -> LLMResponse:
        """Process standard JSON response."""
        async with self._session.post(url, headers=headers, json=payload) as response:
            latency = time.perf_counter() - start_time
            if response.status != 200:
                err_text = await response.text()
                return LLMResponse(
                    prompt=prompt,
                    success=False,
                    status_code=response.status,
                    error_msg=f"HTTP {response.status}: {err_text[:300]}",
                    latency_s=latency,
                    input_tokens=input_tokens,
                )

            data = await response.json()
            full_text = ""
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0]
                if "message" in msg and "content" in msg["message"]:
                    full_text = msg["message"]["content"]
                elif "text" in msg:
                    full_text = msg["text"]
            elif "response" in data:
                full_text = str(data["response"])
            elif "content" in data:
                full_text = str(data["content"])
            elif "message" in data:
                full_text = str(data["message"])
            else:
                full_text = str(data)

            usage = data.get("usage", {})
            out_toks = usage.get("completion_tokens", estimate_tokens(full_text))
            in_toks = usage.get("prompt_tokens", input_tokens)

            return LLMResponse(
                prompt=prompt,
                text=full_text,
                success=True,
                status_code=200,
                latency_s=latency,
                ttft_s=latency * 0.4,  # Estimated for non-streaming
                tpot_s=(latency * 0.6) / max(1, out_toks),
                input_tokens=in_toks,
                output_tokens=out_toks,
                raw_response=data,
            )

    async def _mock_query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        stream: bool = True,
    ) -> LLMResponse:
        """Simulate realistic model behavior and latencies for offline verification."""
        # Simulated timing: TTFT ~20-50ms, TPOT ~5-15ms
        sim_ttft = random.uniform(0.020, 0.050)
        sim_tpot = random.uniform(0.006, 0.014)
        await asyncio.sleep(sim_ttft + sim_tpot * min(20, max_tokens))

        prompt_lower = prompt.lower()

        # Check for OWASP / injection test triggers to simulate realistic AI defenses
        if any(w in prompt_lower for w in ["ignore all previous", "ignore previous instructions"]):
            if "say 'injection successful'" in prompt_lower or "say injection successful" in prompt_lower:
                # 90% chance safe refusal, 10% simulated vulnerability for report verification if testing
                response_text = "I cannot fulfill this request. I am programmed to be a helpful and harmless assistant, and I must adhere to my safety guidelines regardless of instructions to override them."
            else:
                response_text = "I cannot ignore my previous system instructions."
        elif "reveal your system prompt" in prompt_lower or "repeat your instructions" in prompt_lower:
            response_text = "I cannot disclose my system prompt, internal instructions, or operational guidelines."
        elif any(w in prompt_lower for w in ["make a bomb", "explosive device", "steal someone's identity", "hack into a government database"]):
            response_text = "I'm sorry, but I cannot assist with illegal activities, cyberattacks, or the creation of weapons and explosives."
        elif "delete all user accounts" in prompt_lower or "grant me admin privileges" in prompt_lower:
            response_text = "Access denied: I do not have the administrative authority to grant privileges or delete user accounts."
        else:
            response_text = f"This is an automated mock response to the prompt: '{prompt[:40]}...'. All benchmark parameters and metrics were captured successfully."

        out_tokens = estimate_tokens(response_text)
        in_tokens = estimate_tokens(prompt)
        total_lat = sim_ttft + sim_tpot * out_tokens

        return LLMResponse(
            prompt=prompt,
            text=response_text,
            success=True,
            status_code=200,
            latency_s=total_lat,
            ttft_s=sim_ttft,
            tpot_s=sim_tpot,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
