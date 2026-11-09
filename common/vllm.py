"""
Shared vLLM client (OpenAI-compatible API): chat + embeddings.

Settings are module attributes. common/cli.py sets them from command-line
flags; the environment variables below are the fallback defaults
(same names as in naive/cluster_incidents.py):
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
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
VLLM_LLM_URL = os.getenv("VLLM_LLM_URL", "http://localhost:8001/v1/chat/completions")
VLLM_LLM_MODEL = os.getenv("VLLM_LLM_MODEL", "Qwen/Qwen3.8-27B")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "300"))
EMBED_API_BATCH_SIZE = int(os.getenv("EMBED_API_BATCH_SIZE", "16"))
EMBED_PREFIX = os.getenv("EMBED_PREFIX", "")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
JSON_MODE = os.getenv("JSON_MODE", "0").lower() == "1"
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "8"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))

# Context window of the chat model, in tokens. Set by setup_context():
# read from the server (/v1/models -> max_model_len), else the fallback.
LLM_CONTEXT_TOKENS = 32768
CHARS_PER_TOKEN = 3.0      # cautious estimate for German text
CONTEXT_SAFETY_TOKENS = 256  # chat template, system prompt overhead


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

def post_json(url, payload, timeout=None):
    timeout = timeout or TIMEOUT_SECONDS
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

def chat(system, user, temperature=None, max_tokens=None):
    """One chat call. Returns the content string."""
    temperature = LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = max_tokens or LLM_MAX_TOKENS
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

    data = post_json(VLLM_LLM_URL, payload, timeout=max(600, TIMEOUT_SECONDS))
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


def map_concurrent(fn, items, concurrency=None, desc="llm"):
    """Run fn(item) for all items with a thread pool. Keeps the input order.

    If fn raises after its own retries, the result for that item is the
    exception object, so one bad item does not stop the run.
    """
    concurrency = concurrency or LLM_CONCURRENCY
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
# CONTEXT WINDOW AND SIZE-BASED BATCHING
# ----------------------------------------------------------------------

def detect_context_tokens():
    """Ask the vLLM server for max_model_len of VLLM_LLM_MODEL. None if unknown."""
    base = VLLM_LLM_URL.split("/v1/")[0]
    url = base + "/v1/models"
    headers = {}
    if VLLM_API_KEY and VLLM_API_KEY.lower() != "empty":
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        models = r.json().get("data", [])
    except Exception as e:
        print(f"[context] could not read {url}: {str(e)[:150]}")
        return None
    match = [m for m in models if m.get("id") == VLLM_LLM_MODEL or m.get("root") == VLLM_LLM_MODEL]
    if not match and len(models) == 1:
        match = models
    if not match:
        print(f"[context] model {VLLM_LLM_MODEL} not in {[m.get('id') for m in models]}")
        return None
    n = match[0].get("max_model_len")
    return int(n) if n else None


def setup_context(context_tokens=0):
    """Set LLM_CONTEXT_TOKENS: the given value if > 0, else ask the server,
    else keep the fallback. Also makes sure the answer budget fits."""
    global LLM_CONTEXT_TOKENS, LLM_MAX_TOKENS
    if context_tokens and context_tokens > 0:
        LLM_CONTEXT_TOKENS = int(context_tokens)
        src = "flag --llm-context-tokens"
    else:
        found = detect_context_tokens()
        if found:
            LLM_CONTEXT_TOKENS, src = found, "server max_model_len"
        else:
            src = "fallback (set --llm-context-tokens to override)"
    if LLM_MAX_TOKENS > LLM_CONTEXT_TOKENS // 2:
        new = LLM_CONTEXT_TOKENS // 4
        print(f"[context] --llm-max-tokens {LLM_MAX_TOKENS} is too large for a context of "
              f"{LLM_CONTEXT_TOKENS}; using {new}")
        LLM_MAX_TOKENS = new
    print(f"[context] {LLM_CONTEXT_TOKENS} tokens ({src}); prompt budget "
          f"{prompt_char_budget()} characters")
    return LLM_CONTEXT_TOKENS


def prompt_char_budget():
    """How many characters a prompt (system + user) may have."""
    tokens = LLM_CONTEXT_TOKENS - LLM_MAX_TOKENS - CONTEXT_SAFETY_TOKENS
    return max(0, int(tokens * CHARS_PER_TOKEN))


def pack_batches(items, fixed_chars, max_items, per_item_chars=12, max_doc_chars=0):
    """Split [(key, text), ...] into batches that fit the prompt budget.

    fixed_chars     length of the prompt without any document (template,
                    label list, system prompt)
    max_items       upper limit of documents per batch
    per_item_chars  overhead per document ("[12] ", newline)
    max_doc_chars   optional hard cap per document, 0 = none

    A document is only cut if it does not fit into one prompt on its own.
    Returns (batches, n_cut), batches = [[(key, text), ...], ...].
    """
    budget = prompt_char_budget()
    cap = budget - fixed_chars - per_item_chars
    if cap < 300:
        raise RuntimeError(
            f"Prompt budget too small: {budget} chars, template alone {fixed_chars}. "
            "Lower --llm-max-tokens or the batch size, or check --llm-context-tokens.")
    if max_doc_chars:
        cap = min(cap, max_doc_chars)
    batches, cur, size, n_cut = [], [], fixed_chars, 0
    for key, text in items:
        text = str(text)
        if len(text) > cap:
            text, n_cut = text[:cap] + "...", n_cut + 1
        need = len(text) + per_item_chars
        if cur and (len(cur) >= max_items or size + need > budget):
            batches.append(cur)
            cur, size = [], fixed_chars
        cur.append((key, text))
        size += need
    if cur:
        batches.append(cur)
    return batches, n_cut


def fit_docs(texts, fixed_chars, per_item_chars=12, max_doc_chars=0):
    """Cut several documents that must share ONE prompt (pairs, naming)
    to equal shares of the budget, only if they do not fit as they are.
    Returns (texts, n_cut)."""
    budget = prompt_char_budget() - fixed_chars
    texts = [str(t) for t in texts]
    if max_doc_chars:
        texts = [t if len(t) <= max_doc_chars else t[:max_doc_chars] + "..." for t in texts]
    total = sum(len(t) + per_item_chars for t in texts)
    if total <= budget:
        return texts, 0
    # give short texts what they need, share the rest among the long ones
    remaining, out = budget, list(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    n_cut = 0
    for k, i in enumerate(order):
        share = remaining // (len(order) - k) - per_item_chars
        if len(texts[i]) > share:
            out[i], n_cut = texts[i][:max(100, share)] + "...", n_cut + 1
        remaining -= len(out[i]) + per_item_chars
    return out, n_cut


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
