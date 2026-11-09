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
import threading
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
LLM_THINKING = os.getenv("LLM_THINKING", "0").lower() in ("1", "true", "yes")


# Where failed LLM answers are logged in full (set by the scripts to the
# output folder); None = no log.
DEBUG_DIR = None
_debug_lock = threading.Lock()
_last = threading.local()  # finish_reason of the last chat() in this thread


def log_failed_answer(error, content, prompt=None):
    if DEBUG_DIR is None:
        return
    from pathlib import Path
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": str(error),
           "finish_reason": getattr(_last, "finish", None),
           "answer_chars": len(content or ""), "answer": content,
           "prompt_start": (prompt or "")[:1500]}
    with _debug_lock:
        Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
        with open(Path(DEBUG_DIR) / "llm_errors.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class NoRetryError(RuntimeError):
    """An error that a plain retry with the same request cannot fix."""


def _thinking_switch(payload, messages):
    """Turn the model's hidden thinking off, unless --llm-thinking is set.
    Two switches: chat_template_kwargs (vLLM, Qwen3 chat templates) and
    the /no_think soft switch in the user turn (older Qwen3 models)."""
    if LLM_THINKING:
        return
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    for m in messages:
        if m["role"] == "user" and not m["content"].startswith("/no_think"):
            m["content"] = "/no_think " + m["content"]

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
        except NoRetryError:
            raise
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
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    payload = {
        "model": VLLM_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    _thinking_switch(payload, messages)
    if JSON_MODE:
        payload["response_format"] = {"type": "json_object"}

    prompt_tokens = (len(system) + len(user)) / CHARS_PER_TOKEN
    room = int(LLM_CONTEXT_TOKENS - prompt_tokens - CONTEXT_SAFETY_TOKENS)
    for attempt in range(2):
        data = post_json(VLLM_LLM_URL, payload, timeout=max(600, TIMEOUT_SECONDS))
        try:
            choice = data["choices"][0]
        except Exception as e:
            raise RuntimeError(f"Bad LLM response shape: {str(data)[:500]}") from e
        content = choice.get("message", {}).get("content")
        finish = choice.get("finish_reason")
        _last.finish = finish
        if content is not None and content.strip():
            return content
        if finish != "length":
            raise RuntimeError(f"LLM returned empty content. finish_reason={finish}.")
        # The answer budget ran out before any answer text: usually hidden
        # thinking. Try once more with double the budget, if the context allows.
        bigger = min(payload["max_tokens"] * 2, room)
        if attempt == 0 and bigger > payload["max_tokens"]:
            payload["max_tokens"] = bigger
            continue
        break
    raise NoRetryError(
        f"LLM returned empty content: the answer budget of {payload['max_tokens']} tokens "
        "ran out before any answer text. The model probably still thinks. Try "
        "--llm-max-tokens 8000 or more, or a model without thinking.")


def _repair_truncated(text):
    """Close a JSON object that was cut off: drop the incomplete last element,
    then add the missing ] and }. Returns a string or None."""
    stack, in_str, esc, last_ok = [], False, False, None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                last_ok = (i, list(stack))  # end of a complete string
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if stack:
                stack.pop()
            last_ok = (i, list(stack))
    if last_ok is None:
        return None
    i, st = last_ok
    head = text[:i + 1].rstrip()
    # a dangling key ("key") without value cannot be closed sensibly
    if head.endswith('"') and st and st[-1] == "{" and text[i + 1:].lstrip().startswith(":"):
        return None
    closers = "".join("]" if c == "[" else "}" for c in reversed(st))
    return head + closers


def parse_json_response(text):
    """Extract the first JSON object from an LLM answer. Returns None on failure.

    Tolerates text before/after the JSON and repairs a cut-off answer up to
    its last complete element.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    end = text.rfind("}")
    if end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    repaired = _repair_truncated(text[start:])
    if repaired:
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def chat_json(system, user, validate=None, **kw):
    """Chat call that must return a JSON object. Retries on bad JSON.

    validate: optional function(obj) -> bool; False triggers a retry.
    """
    def once():
        content = chat(system, user, **kw)
        cut = getattr(_last, "finish", None) == "length"
        obj = parse_json_response(content)
        why = None
        if obj is None:
            why = "No valid JSON in answer"
        elif validate is not None and not validate(obj):
            why = "JSON failed validation"
        if why:
            if cut:
                why += (f" (answer cut at the token limit after {len(content)} characters; "
                        "the model probably repeats itself)")
            log_failed_answer(why, content, user)
            raise ValueError(f"{why}: {content[:150]!r}")
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
    done = ok = no_retry = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
                ok += 1
            except Exception as e:
                print(f"[{desc}] item {i} failed: {str(e)[:200]}")
                results[i] = e
                no_retry += isinstance(e, NoRetryError)
                # the same unfixable error again and again, nothing worked: stop
                if no_retry >= 3 and ok == 0:
                    for other in futs:
                        other.cancel()
                    raise SystemExit(f"error: [{desc}] stopped after {no_retry} failures "
                                     f"and no success:\n  {e}")
            done += 1
            if done % max(1, len(items) // 20) == 0 or done == len(items):
                print(f"[{desc}] {done}/{len(items)}")
    return results


# ----------------------------------------------------------------------
# CONTEXT WINDOW AND SIZE-BASED BATCHING
# ----------------------------------------------------------------------

def check_chat_server():
    """One tiny chat request at startup. Stops the run with a clear message
    if the server cannot be reached, rejects the key or does not know the
    model, instead of failing later in every call."""
    messages = [{"role": "user", "content": "Antworte nur mit OK."}]
    payload = {"model": VLLM_LLM_MODEL, "max_tokens": 200, "temperature": 0,
               "messages": messages}
    _thinking_switch(payload, messages)
    try:
        data = post_json(VLLM_LLM_URL, payload, timeout=min(120, TIMEOUT_SECONDS))
        if not data.get("choices"):
            raise RuntimeError(f"no 'choices' in answer: {str(data)[:200]}")
    except Exception as e:
        raise SystemExit(
            f"error: chat server check failed for --llm-url {VLLM_LLM_URL} "
            f"with --llm-model {VLLM_LLM_MODEL}:\n  {str(e)[:500]}\n"
            "Check the URL, the model name and VLLM_API_KEY.")
    msg = data["choices"][0].get("message", {})
    content = (msg.get("content") or "").strip()
    thinks = bool(msg.get("reasoning_content") or msg.get("reasoning")) or "<think>" in content
    print(f"[check] chat server OK: {VLLM_LLM_URL} ({VLLM_LLM_MODEL})")
    if not LLM_THINKING and (thinks or not content):
        print("[check] WARNING: the model still thinks although thinking is switched off "
              f"(answer: {content[:60]!r}). Answers may come back empty. If so, raise "
              "--llm-max-tokens (e.g. 8000) or use a model without thinking.")


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
