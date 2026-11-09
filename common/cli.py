"""
Command-line flags shared by all pipelines.

Every flag falls back to an environment variable (shown in --help), then to
the default. The API key is read only from the environment (VLLM_API_KEY),
so it does not end up in the shell history.
"""

import os
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def env(name, default, typ=str):
    """Default for a flag: environment variable if set, else default."""
    v = os.getenv(name)
    if v is None or v == "":
        return default
    if typ is bool:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return typ(v)


class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                argparse.RawDescriptionHelpFormatter):
    def _get_help_string(self, action):
        if action.default is None:  # per-format default, explained in the help text
            return action.help
        return super()._get_help_string(action)


def make_parser(description):
    return argparse.ArgumentParser(description=description, formatter_class=Formatter)


def add_flag(group, flag, envname, default, typ=str, help="", **kw):
    """Add --flag with env fallback. typ=bool gives --flag / --no-flag."""
    h = f"{help} [env: {envname}]" if envname else help
    if typ is bool:
        group.add_argument(flag, default=env(envname, default, bool) if envname else default,
                           action=argparse.BooleanOptionalAction, help=h, **kw)
    else:
        group.add_argument(flag, default=env(envname, default, typ) if envname else default,
                           type=typ, help=h, **kw)


def add_common_args(parser, embeddings=True, llm=True):
    g = parser.add_argument_group("data")
    add_flag(g, "--format", "DATA_PROFILE", "incidents", str,
             "input format: incidents (ServiceNow) or cirs", choices=["incidents", "cirs"])
    add_flag(g, "--input", "CLEANED_FILE", None, str,
             "cleaned input JSON/CSV (default: cleaned_<format>.json)")
    add_flag(g, "--output-dir", "OUTPUT_DIR", None, str,
             "output folder (default: cluster_output/<format>/<approach>)")
    add_flag(g, "--hypotheses-file", "HYPOTHESES_FILE", None, str,
             "hypothesis clusters, JSON {name: [ids]} (default: hypotheses/<format>.json)")
    add_flag(g, "--seed", "SEED", 42, int, "random seed")
    add_flag(g, "--resume", "RESUME", True, bool, "reuse saved stage results")
    add_flag(g, "--plot", "PLOT", True, bool, "make the UMAP plot")

    if embeddings:
        g = parser.add_argument_group("embeddings (vLLM API)")
        add_flag(g, "--embed-url", "VLLM_EMBED_URL", "http://localhost:8000/v1/embeddings", str,
                 "embedding endpoint")
        add_flag(g, "--embed-model", "VLLM_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B", str,
                 "embedding model")
        add_flag(g, "--embed-batch-size", "EMBED_API_BATCH_SIZE", 16, int, "texts per embedding call")
        add_flag(g, "--embed-prefix", "EMBED_PREFIX", "", str, "prefix added to every text")
        add_flag(g, "--embed-cache-dir", "EMBED_CACHE_DIR", "cluster_output/embed_cache", str,
                 "embedding cache folder")

    if llm:
        g = parser.add_argument_group("LLM (vLLM API)")
        add_flag(g, "--llm-url", "VLLM_LLM_URL", "http://localhost:8001/v1/chat/completions", str,
                 "chat endpoint")
        add_flag(g, "--llm-model", "VLLM_LLM_MODEL", "Qwen/Qwen3.8-27B", str, "chat model")
        add_flag(g, "--llm-concurrency", "LLM_CONCURRENCY", 8, int,
                 "parallel requests (vLLM batches them on the server)")
        add_flag(g, "--llm-retries", "LLM_RETRIES", 3, int, "retries per failed request")
        add_flag(g, "--llm-temperature", "LLM_TEMPERATURE", 0.1, float, "sampling temperature")
        add_flag(g, "--llm-max-tokens", "LLM_MAX_TOKENS", 4000, int, "max tokens per answer")
        add_flag(g, "--llm-context-tokens", "LLM_CONTEXT_TOKENS", 0, int,
                 "context length of the chat model; 0 = ask the server (/v1/models "
                 "max_model_len), fallback 32768. Batches are split to fit it.")
        add_flag(g, "--chars-per-token", "CHARS_PER_TOKEN", 3.0, float,
                 "estimate used to turn the token budget into characters")
        add_flag(g, "--json-mode", "JSON_MODE", False, bool, "request response_format=json_object")
        add_flag(g, "--timeout", "TIMEOUT_SECONDS", 300, int, "HTTP timeout in seconds")
        add_flag(g, "--llm-name-check", "LLM_NAME_CHECK", True, bool,
                 "let the LLM judge if cluster names match the hypothesis cluster names")
    return parser


def apply_common(args, approach):
    """Resolve per-format defaults and push settings into the common modules."""
    from common import vllm, embeddings, profile

    profile.set_profile(args.format)
    if not args.input:
        args.input = f"cleaned_{args.format}.json"
    if not args.output_dir:
        args.output_dir = f"cluster_output/{args.format}/{approach}"
    args.output_dir = Path(args.output_dir)
    if not args.hypotheses_file:
        args.hypotheses_file = str(REPO_ROOT / "hypotheses" / f"{args.format}.json")

    if hasattr(args, "embed_url"):
        vllm.VLLM_EMBED_URL = args.embed_url
        vllm.VLLM_EMBED_MODEL = args.embed_model
        vllm.EMBED_API_BATCH_SIZE = args.embed_batch_size
        vllm.EMBED_PREFIX = args.embed_prefix
        embeddings.EMBED_CACHE_DIR = Path(args.embed_cache_dir)
    if hasattr(args, "llm_url"):
        vllm.VLLM_LLM_URL = args.llm_url
        vllm.VLLM_LLM_MODEL = args.llm_model
        vllm.LLM_CONCURRENCY = args.llm_concurrency
        vllm.LLM_RETRIES = args.llm_retries
        vllm.LLM_TEMPERATURE = args.llm_temperature
        vllm.LLM_MAX_TOKENS = args.llm_max_tokens
        vllm.JSON_MODE = args.json_mode
        vllm.TIMEOUT_SECONDS = args.timeout
        vllm.CHARS_PER_TOKEN = args.chars_per_token
    return args


def setup_llm_context(args):
    """Read the context length (flag or server) before the first LLM call."""
    from common import vllm
    vllm.setup_context(args.llm_context_tokens)


def print_settings(args):
    print("Settings:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}")
