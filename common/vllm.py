"""
Shared vLLM client (OpenAI-compatible API): chat + embeddings.

Uses the same environment variables as naive/cluster_incidents.py:
  VLLM_EMBED_URL, VLLM_EMBED_MODEL, VLLM_LLM_URL, VLLM_LLM_MODEL,
  VLLM_API_KEY, TIMEOUT_SECONDS, EMBED_API_BATCH_SIZE, EMBED_PREFIX,
  LLM_TEMPERATURE, LLM_MAX_TOKENS, JSON_MODE
New:
  LLM_CONCURRENCY    parallel requests to the vLLM server (default 8).
                     vLLM batches concurrent requests on the server side.
  LLM_RETRIES        retries per request on errors (default 3)
"""

import os
import re
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

VLLM_EMBED_URL = os.getenv("VLLM_EMBED_URL", "http://localhost:8000/v1/embeddings")
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "BAAI/bge-m3")
VLLM_LLM_URL = os.getenv("VLLM_LLM_URL", "http://localhost:8001/v1/chat/completions")
VLLM_LLM_MODEL = os.getenv("VLLM_LLM_MODEL", "qwen:3.8:27b")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "300"))
EMBED_API_BATCH_SIZE = int(os.getenv("EMBED_API_BATCH_SIZE", "16"))
EMBED_PREFIX = os.getenv("EMBED_PREFIX", "")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
JSON_MODE = os.getenv("JSON_MODE", "0").lower() == "1"
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "8"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

def post_json(url, payload, timeout=TIMEOUT_SECONDS):
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY and VLLM_API_KEY.lower() != "empty":
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from {url}\nResponse: {r.text[:1000]}")
    return r.json()


def _with_retries(fn, what):
    last = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # network, HTTP, parse errors
            last = e
            if attempt < LLM_RETRIES:
                wait = 2 ** attempt
                print(f"[retry] {what} failed ({str(e)[:120]}), retry in {wait}s")
                time.sleep(wait)
    raise last


# ----------------------------------------------------------------------
# CHAT
# ----------------------------------------------------------------------

def chat(system, user, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
    """One chat call. Returns the content string."""
    # Qwen3 thinking models: /no_think disables the hidden reasoning chain.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "/no_think " + user},
    ]
    payload = {
        "model": VLLM_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if JSON_MODE:
        payload["response_format"] = {"type": "json_object"}

    data = post_json(VLLM_LLM_URL, payload, timeout=600)
    try:
        choice = data["choices"][0]
    except Exception as e:
        raise RuntimeError(f"Bad LLM response shape: {str(data)[:500]}") from e
    content = choice.get("message", {}).get("content")
    if content is None or not content.strip():
        raise RuntimeError(
            f"LLM returned empty content. finish_reason={choice.get('finish_reason')}."
        )
    return content


def parse_json_response(text):
    """Extract the first JSON object from an LLM answer. Returns None on failure."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def chat_json(system, user, validate=None, **kw):
    """Chat call that must return a JSON object. Retries on bad JSON.

    validate: optional function(obj) -> bool; False triggers a retry.
    """
    def once():
        content = chat(system, user, **kw)
        obj = parse_json_response(content)
        if obj is None:
            raise ValueError(f"No valid JSON in answer: {content[:200]}")
        if validate is not None and not validate(obj):
            raise ValueError(f"JSON failed validation: {content[:200]}")
        return obj
    return _with_retries(once, "chat_json")


def map_concurrent(fn, items, concurrency=LLM_CONCURRENCY, desc="llm"):
    """Run fn(item) for all items with a thread pool. Keeps the input order.

    If fn raises after its own retries, the result for that item is the
    exception object, so one bad item does not stop the run.
    """
    items = list(items)
    results = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as e:
                print(f"[{desc}] item {i} failed: {str(e)[:200]}")
                results[i] = e
            done += 1
            if done % max(1, len(items) // 20) == 0 or done == len(items):
                print(f"[{desc}] {done}/{len(items)}")
    return results


# ----------------------------------------------------------------------
# EMBEDDINGS
# ----------------------------------------------------------------------

def _extract_embeddings(payload):
    if isinstance(payload, dict) and "data" in payload:
        items = payload["data"]
    elif isinstance(payload, list):
        items = payload
    else:
        items = [payload]
    if items and all(isinstance(x, dict) and "index" in x for x in items):
        items = sorted(items, key=lambda x: x.get("index", 0))
    out = []
    for item in items:
        emb = (item.get("embedding") or item.get("embeddings") or item.get("values") or []) \
            if isinstance(item, dict) else item
        out.append([float(v) for v in emb])
    return out


def embed_batch(texts):
    if EMBED_PREFIX:
        texts = [f"{EMBED_PREFIX}{t}" for t in texts]
    payload = {"model": VLLM_EMBED_MODEL, "input": texts, "encoding_format": "float"}
    embs = _with_retries(lambda: _extract_embeddings(post_json(VLLM_EMBED_URL, payload)),
                         "embed")
    if len(embs) != len(texts):
        raise RuntimeError(f"Embedding API returned {len(embs)} vectors for {len(texts)} texts")
    return embs
