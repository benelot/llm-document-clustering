#!/usr/bin/env python3
"""
Approach 3: Text clustering as classification with an LLM
(after Huang & He, "Text Clustering as Classification with LLMs",
arXiv 2410.00927, https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM).

No seed labels: the LLM finds the topics itself.

Stages (each stage result is saved; --resume reuses it on the next run):
  1. generate   LLM reads shuffled batches of incidents, proposes topic labels
                -> stages/raw_labels.json
  2. merge      LLM merges similar labels into one final list (target range)
                -> stages/merged_labels.json   (you may edit this file by hand
                   and re-run: classification then uses your edited list)
  3. classify   LLM assigns each incident to one label, or 0 = "Sonstiges"
                -> stages/classifications.jsonl
  4. report     common/report.py: assignments, clusters, UMAP plot, validation

Examples (run from the folder that holds the cleaned input):
  python <repo>/llm_classification/run.py --format incidents
  python <repo>/llm_classification/run.py --format cirs --llm-concurrency 16
"""

import sys
import json
import random
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import vllm                                   # noqa: E402
from common.data import load_cleaned, truncate            # noqa: E402
from common.embeddings import embed_texts                 # noqa: E402
from common.report import write_report, relabel_by_size   # noqa: E402
from common.profile import P                              # noqa: E402
from common.hypotheses import load_hypotheses                         # noqa: E402
from common import cli                                    # noqa: E402

# ----------------------------------------------------------------------
# CONFIG (set from command-line flags in main(); run with --help)
# ----------------------------------------------------------------------

OUTPUT_DIR = STAGE_DIR = None
RESUME, SEED, PLOT = True, 42, True
GEN_BATCH_SIZE, GEN_MAX_CHARS, GEN_SAMPLE_SIZE = 15, 600, 0
MERGE_CHUNK_SIZE, TARGET_MIN_LABELS, TARGET_MAX_LABELS = 200, 15, 40
CLASSIFY_BATCH_SIZE, CLASSIFY_MAX_CHARS = 10, 1000
NOISE_NAME = "Sonstiges"
DOMAIN = P["system"]


def parse_args(argv=None):
    p = cli.make_parser(__doc__)
    cli.add_common_args(p)
    g = p.add_argument_group("1. label generation")
    cli.add_flag(g, "--gen-batch-size", "GEN_BATCH_SIZE", 15, int, "documents per generation call")
    cli.add_flag(g, "--gen-max-chars", "GEN_MAX_CHARS", 600, int, "characters per document in generation")
    cli.add_flag(g, "--gen-sample-size", "GEN_SAMPLE_SIZE", 0, int,
                 "documents used for generation, 0 = all")
    g = p.add_argument_group("2. label merging")
    cli.add_flag(g, "--merge-chunk-size", "MERGE_CHUNK_SIZE", 200, int, "labels per pre-merge call")
    cli.add_flag(g, "--target-min-labels", "TARGET_MIN_LABELS", 15, int, "min topics in final list")
    cli.add_flag(g, "--target-max-labels", "TARGET_MAX_LABELS", 40, int, "max topics in final list")
    g = p.add_argument_group("3. classification")
    cli.add_flag(g, "--classify-batch-size", "CLASSIFY_BATCH_SIZE", 10, int,
                 "documents per classification call, 1 = as in the paper")
    cli.add_flag(g, "--classify-max-chars", "CLASSIFY_MAX_CHARS", 1000, int,
                 "characters per document in classification")
    return cli.apply_common(p.parse_args(argv), "llm_classification")


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 1. LABEL GENERATION
# ----------------------------------------------------------------------

def _gen_prompt(texts):
    items = "\n".join(f"[{i + 1}] {truncate(t, GEN_MAX_CHARS)}" for i, t in enumerate(texts))
    return f"""Hier sind {len(texts)} {P['items_long']}.

Finde die Themen, die in diesen {P['items']} vorkommen. Ein Thema ist eine
wiederkehrende {P['category']} (z.B. eine Systemfunktion, ein Prozess oder
eine Geräteart), nicht ein Einzelfall. Keine Patientennamen, Daten oder
Ticketnummern in den Labels.

Regeln:
- Labels auf Deutsch, 2 bis 6 Wörter.
- Ein Label pro Thema; mehrere {P['items']} können dasselbe Label teilen.

Gib JSON zurück: {{"labels": ["Label 1", "Label 2", ...]}}

{P['items']}:
{items}
"""


def stage_generate(df):
    path = STAGE_DIR / "raw_labels.json"
    if RESUME and path.exists():
        print(f"[generate] reuse {path}")
        return _read_json(path)

    idx = list(range(len(df)))
    random.Random(SEED).shuffle(idx)
    if GEN_SAMPLE_SIZE > 0:
        idx = idx[:GEN_SAMPLE_SIZE]
    texts = df["consolidated_text"].tolist()
    batches = [[texts[i] for i in idx[s:s + GEN_BATCH_SIZE]]
               for s in range(0, len(idx), GEN_BATCH_SIZE)]
    print(f"[generate] {len(batches)} batches of up to {GEN_BATCH_SIZE} incidents")

    def run(batch):
        obj = vllm.chat_json(DOMAIN, _gen_prompt(batch),
                             validate=lambda o: isinstance(o.get("labels"), list))
        return [str(l).strip() for l in obj["labels"] if str(l).strip()]

    res = vllm.map_concurrent(run, batches, desc="generate")
    raw = [r for r in res if not isinstance(r, Exception)]
    print(f"[generate] {sum(len(r) for r in raw)} raw labels from "
          f"{len(raw)}/{len(batches)} batches")
    _write_json(path, raw)
    return raw


# ----------------------------------------------------------------------
# 2. LABEL MERGING
# ----------------------------------------------------------------------

def _merge_chunk_prompt(labels):
    items = "\n".join(f"- {l}" for l in labels)
    return f"""Hier ist eine Liste von Themen-Labels für {P['items_long']} aus einem Spital.
Fasse Labels zusammen, die dasselbe Thema meinen (Synonyme, andere Wortwahl,
Teilaspekte desselben Problems). Behalte unterschiedliche Themen getrennt.

Gib JSON zurück: {{"labels": ["Label 1", "Label 2", ...]}}

Labels:
{items}
"""


def _final_merge_prompt(labels_with_counts):
    items = "\n".join(f"- {l}" + (f" ({n}x)" if n else "") for l, n in labels_with_counts)
    return f"""Hier ist eine Liste von Themen-Labels für {P['items_long']} aus einem Spital
(in Klammern, wie oft ein Label vorgeschlagen wurde, falls bekannt).

Erstelle daraus die finale Themenliste mit {TARGET_MIN_LABELS} bis
{TARGET_MAX_LABELS} Themen:
- Fasse Labels zusammen, die dasselbe Thema meinen.
- Häufige Themen müssen erhalten bleiben.
- Die Themen sollen sich klar voneinander abgrenzen.
- Kein Sammelthema wie "Sonstiges" (das gibt es schon separat).
- Name auf Deutsch, 2 bis 6 Wörter. Beschreibung: 1 Satz, woran man eine
  {P['item']} zu diesem Thema erkennt.

Gib JSON zurück:
{{"labels": [{{"name": "...", "description": "..."}}, ...]}}

Labels:
{items}
"""


def stage_merge(raw):
    path = STAGE_DIR / "merged_labels.json"
    if RESUME and path.exists():
        print(f"[merge] reuse {path}")
        return _read_json(path)

    counts = Counter()
    display = {}
    for batch in raw:
        for l in batch:
            key = " ".join(l.lower().split())
            counts[key] += 1
            display.setdefault(key, l)
    labels = [display[k] for k, _ in counts.most_common()]
    print(f"[merge] {len(labels)} distinct raw labels")

    with_counts = [(display[k], n) for k, n in counts.most_common()]
    rnd = 0
    while len(labels) > MERGE_CHUNK_SIZE:
        rnd += 1
        labels = sorted(labels, key=str.lower)
        chunks = [labels[i:i + MERGE_CHUNK_SIZE] for i in range(0, len(labels), MERGE_CHUNK_SIZE)]
        print(f"[merge] round {rnd}: {len(labels)} labels in {len(chunks)} chunks")
        res = vllm.map_concurrent(
            lambda ch: vllm.chat_json(DOMAIN, _merge_chunk_prompt(ch),
                                      validate=lambda o: isinstance(o.get("labels"), list))["labels"],
            chunks, desc="merge")
        new = []
        for ch, r in zip(chunks, res):
            new.extend(ch if isinstance(r, Exception) else [str(x).strip() for x in r if str(x).strip()])
        new = list(dict.fromkeys(new))
        if len(new) >= len(labels) * 0.95:  # no progress, stop
            labels = new
            break
        labels = new
        with_counts = [(l, None) for l in labels]

    def valid(o):
        ls = o.get("labels")
        return isinstance(ls, list) and len(ls) >= 2 and all(isinstance(x, dict) and x.get("name") for x in ls)

    obj = vllm.chat_json(DOMAIN, _final_merge_prompt(with_counts), validate=valid)
    final = []
    seen = set()
    for x in obj["labels"]:
        name = str(x["name"]).strip()
        if name.lower() in seen or name.lower() == NOISE_NAME.lower():
            continue
        seen.add(name.lower())
        final.append({"name": name, "description": str(x.get("description", "")).strip()})
    print(f"[merge] final list: {len(final)} labels")
    _write_json(path, final)
    return final


# ----------------------------------------------------------------------
# 3. CLASSIFICATION
# ----------------------------------------------------------------------

def _classify_prompt(label_list, texts):
    labels = "\n".join(f"{i + 1}: {l['name']} - {l.get('description', '')}"
                       for i, l in enumerate(label_list))
    items = "\n".join(f"[{i + 1}] {truncate(t, CLASSIFY_MAX_CHARS)}" for i, t in enumerate(texts))
    example = ", ".join(f'"{i + 1}": <nr>' for i in range(min(len(texts), 3)))
    return f"""Ordne jede {P['item']} genau einem Thema zu.

Themen:
0: {NOISE_NAME} - passt zu keinem der Themen
{labels}

Gib JSON zurück, mit der Themennummer pro {P['item']}:
{{"assignments": {{{example}{", ..." if len(texts) > 3 else ""}}}}}

{P['items']}:
{items}
"""


def stage_classify(df, label_list):
    path = STAGE_DIR / "classifications.jsonl"
    done = {}
    if RESUME and path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["incident_id"]] = r["label"]
        print(f"[classify] resume: {len(done)} already classified")
    elif path.exists():
        path.unlink()

    todo = df[~df["incident_id"].isin(done)]
    rows = list(zip(todo["incident_id"], todo["consolidated_text"]))
    batches = [rows[i:i + CLASSIFY_BATCH_SIZE] for i in range(0, len(rows), CLASSIFY_BATCH_SIZE)]
    k = len(label_list)
    lock = threading.Lock()
    print(f"[classify] {len(rows)} incidents in {len(batches)} batches, {k} labels")

    def ask(batch):
        n = len(batch)

        def valid(o):
            a = o.get("assignments")
            if not isinstance(a, dict):
                return False
            try:
                return all(0 <= int(a[str(i + 1)]) <= k for i in range(n))
            except (KeyError, ValueError, TypeError):
                return False

        obj = vllm.chat_json(DOMAIN, _classify_prompt(label_list, [t for _, t in batch]),
                             validate=valid)
        return [int(obj["assignments"][str(i + 1)]) for i in range(n)]

    def run(batch):
        try:
            labs = ask(batch)
        except Exception:
            # batch failed after retries: classify one by one
            labs = []
            for item in batch:
                try:
                    labs.append(ask([item])[0])
                except Exception:
                    labs.append(None)  # unclassifiable -> Sonstiges, flagged
        with lock, open(path, "a", encoding="utf-8") as f:
            for (iid, _), lab in zip(batch, labs):
                f.write(json.dumps({"incident_id": iid, "label": lab}) + "\n")
        return labs

    vllm.map_concurrent(run, batches, desc="classify")

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            done[r["incident_id"]] = r["label"]
    return done


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main(argv=None):
    global OUTPUT_DIR, STAGE_DIR, RESUME, SEED, PLOT, DOMAIN
    global GEN_BATCH_SIZE, GEN_MAX_CHARS, GEN_SAMPLE_SIZE
    global MERGE_CHUNK_SIZE, TARGET_MIN_LABELS, TARGET_MAX_LABELS
    global CLASSIFY_BATCH_SIZE, CLASSIFY_MAX_CHARS
    args = parse_args(argv)
    cli.print_settings(args)
    OUTPUT_DIR, STAGE_DIR = args.output_dir, args.output_dir / "stages"
    RESUME, SEED, PLOT, DOMAIN = args.resume, args.seed, args.plot, P["system"]
    GEN_BATCH_SIZE, GEN_MAX_CHARS, GEN_SAMPLE_SIZE = \
        args.gen_batch_size, args.gen_max_chars, args.gen_sample_size
    MERGE_CHUNK_SIZE, TARGET_MIN_LABELS, TARGET_MAX_LABELS = \
        args.merge_chunk_size, args.target_min_labels, args.target_max_labels
    CLASSIFY_BATCH_SIZE, CLASSIFY_MAX_CHARS = args.classify_batch_size, args.classify_max_chars

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cleaned(args.input)

    raw = stage_generate(df)
    label_list = stage_merge(raw)
    result = stage_classify(df, label_list)

    lab = [result.get(i) for i in df["incident_id"]]
    failed = [l is None for l in lab]
    # label k (1..K) -> cluster k-1, 0/None -> -1 (Sonstiges)
    raw_ids = [(-1 if (l is None or l == 0) else l - 1) for l in lab]
    labels, mapping = relabel_by_size(raw_ids)
    info = {new: {"cluster_name": label_list[old]["name"],
                  "description": label_list[old].get("description", "")}
            for old, new in mapping.items() if old != -1}
    unused = [label_list[i]["name"] for i in range(len(label_list)) if i not in mapping]
    if unused:
        print(f"[report] {len(unused)} labels got no incident: {unused}")

    vectors = None
    if PLOT:
        print("Embedding incidents for the UMAP plot...")
        vectors = embed_texts(df["consolidated_text"].tolist())

    write_report(
        df, labels, info, OUTPUT_DIR,
        method="LLM classification", vectors=vectors, noise_name=NOISE_NAME,
        params={
            "format": args.format, "input": args.input,
            "LLM": vllm.VLLM_LLM_MODEL, "GEN_BATCH_SIZE": GEN_BATCH_SIZE,
            "GEN_SAMPLE_SIZE": GEN_SAMPLE_SIZE or "all",
            "TARGET_LABELS": f"{TARGET_MIN_LABELS}-{TARGET_MAX_LABELS}",
            "CLASSIFY_BATCH_SIZE": CLASSIFY_BATCH_SIZE, "SEED": SEED,
            "labels in final list": len(label_list),
            "labels without incidents": len(unused),
            "incidents not classifiable (LLM error)": sum(failed),
        },
        extra_cols={"llm_failed": failed},
        hypotheses=load_hypotheses(args.hypotheses_file), name_check=args.llm_name_check,
    )


if __name__ == "__main__":
    main()
