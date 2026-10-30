"""Model router: one OpenAI-compatible client for every provider, routes from config.yaml.

Every LLM call in the project goes through `Router.chat`, which
- enforces the privacy rule (D2): `private=True` calls may only use providers with private_ok,
- caches responses by content hash (D5), so re-runs cost nothing,
- retries with backoff and falls back to the route's `fallback` tier,
- appends one JSON line per call to router/logs/calls.jsonl (no prompt text: logs stay shareable),
  which is the source for the RQ1 cost/latency matrix.
"""

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

CACHE = Path(".cache/llm")
RETRYABLE = (RateLimitError, APIConnectionError, InternalServerError)


class PrivacyError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    provider: str
    model: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    cached: bool


def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()
    return yaml.safe_load(open(path))


class Router:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.log_path = Path(self.cfg["paths"]["router_log"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        self._clients: dict[str, OpenAI] = {}
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    def route(self, task: str) -> tuple[str, str]:
        r = self.cfg["routes"][task]
        return r["provider"], r["model"]

    def client(self, provider: str) -> OpenAI:
        if provider not in self._clients:
            p = self.cfg["providers"][provider]
            if p.get("enabled") is False:
                raise RuntimeError(f"provider {provider} is disabled in config.yaml")
            key = os.environ.get(p["api_key_env"], "") if p["api_key_env"] else "ollama"
            if not key:
                raise RuntimeError(f"{p['api_key_env']} is not set in .env")
            base = os.environ.get("OLLAMA_BASE_URL", p["base_url"]) if provider == "ollama" else p["base_url"]
            self._clients[provider] = OpenAI(base_url=base, api_key=key, timeout=p.get("timeout_s", 600),
                                             max_retries=0)
        return self._clients[provider]

    def _throttle(self, provider: str):
        rpm = self.cfg["providers"][provider].get("rpm")
        if not rpm:
            return
        with self._lock:
            wait = self._last_call.get(provider, 0) + 60 / rpm - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call[provider] = time.time()

    def cost(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """List-price cost; local models cost 0, hosted models without a listed price return None."""
        if self.cfg["providers"][provider]["private_ok"]:
            return 0.0
        price = (self.cfg.get("prices_per_1m") or {}).get(model)
        return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1e6 if price else None

    def chat(self, task: str, messages: list[dict], *, private: bool, doc_id: str = "", run: int = 0,
             provider: str | None = None, model: str | None = None, json_mode: bool = True,
             max_tokens: int | None = None) -> Reply:
        """Send one chat request. `provider`/`model` pin a model (benchmarks): no fallback then, so a
        result is always from the model it is labeled with. Route-based calls may fall back."""
        tiers = [(provider, model)] if provider else [self.route(task)]
        fb = self.cfg["routes"].get(task, {}).get("fallback")
        if fb and not provider:
            tiers.append((fb["provider"], fb["model"]))
        error = None
        for prov, mod in tiers:
            if private and not self.cfg["providers"][prov]["private_ok"]:
                if (prov, mod) == tiers[0]:
                    raise PrivacyError(f"task {task!r}: private data may not go to {prov} (D2)")
                continue
            try:
                return self._call(task, prov, mod, messages, doc_id, run, json_mode, max_tokens)
            except (APIStatusError, *RETRYABLE) as e:
                error = e
        raise error

    def _call(self, task, provider, model, messages, doc_id, run, json_mode, max_tokens) -> Reply:
        temperature = self.cfg["llm"]["temperature"]
        extra = dict(self.cfg.get("model_params", {}).get(model, {}))
        # reasoning models (gpt-5-mini): max_tokens is renamed and also has to cover the hidden reasoning tokens
        tokens_field = extra.pop("max_tokens_field", "max_tokens")
        tokens_add = extra.pop("max_tokens_add", 0)
        body = {"model": model, "messages": messages, "temperature": temperature, **extra}
        body = {k: v for k, v in body.items() if v is not None}   # `temperature: null` = use the model default
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if max_tokens:
            body[tokens_field] = max_tokens + tokens_add
        key = hashlib.sha256(json.dumps([provider, body, run], sort_keys=True).encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            c = json.loads(path.read_text())
            reply = Reply(c["text"], provider, model, c["latency_s"], c["prompt_tokens"], c["completion_tokens"],
                          c["cost_usd"], True)
        else:
            reply = self._send(provider, body)
            path.write_text(json.dumps(reply.__dict__))
        self._log(task, reply, doc_id, run, ok=True)
        return reply

    @retry(retry=retry_if_exception_type(RETRYABLE), wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(10),
           reraise=True)
    def _send(self, provider: str, body: dict) -> Reply:
        self._throttle(provider)
        t0 = time.perf_counter()
        r = self.client(provider).chat.completions.create(**body)
        latency = time.perf_counter() - t0
        u = r.usage
        pt, ct = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
        return Reply(r.choices[0].message.content or "", provider, body["model"], round(latency, 3), pt, ct,
                     self.cost(provider, body["model"], pt, ct), False)

    def _log(self, task: str, reply: Reply, doc_id: str, run: int, ok: bool):
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "provider": reply.provider,
               "model": reply.model, "doc_id": doc_id, "run": run, "latency_s": reply.latency_s,
               "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
               "cost_usd": None if reply.cost_usd is None else round(reply.cost_usd, 8), "cached": reply.cached, "ok": ok}
        with self._lock, open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")


_default: Router | None = None


def get_router() -> Router:
    global _default
    if _default is None:
        _default = Router()
    return _default
