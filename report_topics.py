#!/usr/bin/env python3
"""
Themenübersicht for one finished run: per topic the number of tickets, the
share, the short description and 2-3 typical examples, shortened to one line.

Works for every run folder written by common/report.py (all approaches,
all perspectives).

Examples (choose with --examples):
  short  (default) the original short description ("Kurzbeschreibung") of
         the most typical tickets, cut to --max-chars. No LLM, exact wording.
  llm    the LLM reads the most typical tickets of a topic and writes short,
         distinct one-line examples without names. More concise, but the
         wording is the LLM's. Needs --llm-url (or VLLM_LLM_URL).

"Typical" = closest to the centre of the topic on the run's UMAP map
(supervised coordinates if present). Near-duplicate examples are skipped.

Outputs in the run folder (the two modes do not overwrite each other):
  themenuebersicht_<mode>.md     for reading and copying into a report
  themenuebersicht_<mode>.xlsx   one row per topic

Examples:
  python report_topics.py --run cluster_output/incidents/llm_classification/process
  python report_topics.py --run cluster_output/incidents/llm_classification/process --examples llm
"""

import re
import sys
import json
import difflib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cli, vllm  # noqa: E402
from common.profile import P, set_profile  # noqa: E402

NOISE_NAME = "Sonstiges"
# what one document is called in the overview, per format
UNIT = {"incidents": "Tickets", "survey": "Antworten", "cirs": "Meldungen"}
UNIT_NAME = "Tickets"


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------

def load_run(run, input_path=None):
    run = Path(run)
    a = pd.read_csv(run / "assignments.csv", dtype={"incident_id": str})
    c = pd.read_csv(run / "clusters.csv")
    info = {}
    if (run / "run_info.json").exists():
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    params = info.get("params", {})
    fmt = params.get("format", "incidents")

    # full texts from the cleaned input (fallback: the preview in assignments.csv)
    path = Path(input_path or params.get("input") or f"cleaned_{fmt}.json")
    if path.exists():
        src = (pd.read_csv(path, dtype={"incident_id": str}) if path.suffix == ".csv"
               else pd.read_json(path, orient="records", dtype={"incident_id": str}))
        texts = dict(zip(src["incident_id"].astype(str), src["consolidated_text"].astype(str)))
        print(f"Texts from {path}")
    else:
        texts = {}
        print(f"[warning] {path} not found, using the 300-character preview of assignments.csv")
    a["text"] = [texts.get(i, p) for i, p in zip(a["incident_id"], a["text_preview"].fillna(""))]
    return a, c, info, fmt


def short_text(text, max_chars):
    """One-line version of a ticket: its short description if there is one."""
    text = str(text)
    m = re.search(r"SHORT_DESC:\s*(.*?)(?:\n\s*\n\s*DESCRIPTION:|$)", text, flags=re.S)
    s = m.group(1).strip() if m and m.group(1).strip() else ""
    if not s:  # CIRS: case title, plus the start of "Was ist passiert?"
        t = re.search(r"^Titel:\s*(.+)$", text, flags=re.M)
        e = re.search(r"^Was ist passiert\?:\s*(.+)$", text, flags=re.M)
        if t or e:
            s = " – ".join(x.group(1).strip() for x in (t, e) if x)
    if not s:
        m = re.search(r"DESCRIPTION:\s*(.*?)(?:\n\s*\n|$)", text, flags=re.S)
        s = m.group(1).strip() if m else ""
    if not s:  # survey and other formats: first non-empty paragraph
        s = next((p.strip() for p in text.split("\n\n") if p.strip()), text)
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_chars:
        cut = s[:max_chars].rsplit(" ", 1)[0]
        s = (cut if len(cut) > max_chars * 0.6 else s[:max_chars]).rstrip(" ,.;:-") + " …"
    return s


def typical_order(g):
    """Rows of one topic, most typical first (closest to the topic centre)."""
    for x, y in (("umap_sup_x", "umap_sup_y"), ("umap_x", "umap_y")):
        if x in g and g[x].notna().all():
            pts = g[[x, y]].to_numpy(float)
            centre = np.median(pts, axis=0)
            return g.iloc[np.argsort(np.linalg.norm(pts - centre, axis=1))]
    return g  # no coordinates: file order


def _similar(a, b):
    norm = lambda s: re.sub(r"[\d\W]+", " ", s.lower()).strip()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() > 0.8


def pick_short_examples(g, n, max_chars):
    out = []
    for _, r in typical_order(g).iterrows():
        s = short_text(r["text"], max_chars)
        if not s or any(_similar(s, t) for _, t in out):
            continue
        out.append((r["incident_id"], s))
        if len(out) == n:
            break
    return out


# ----------------------------------------------------------------------
# LLM EXAMPLES
# ----------------------------------------------------------------------

def _llm_prompt(name, description, tickets, n, max_chars):
    items = "\n".join(f"[{iid}] {t}" for iid, t in tickets)
    words = max(8, max_chars // 8)
    unit = UNIT_NAME
    return f"""Hier sind typische {unit} zum Thema "{name}".
Beschreibung des Themas: {description or '-'}

Wähle {n} möglichst unterschiedliche, typische {unit} aus und fasse jedes in
einer kurzen, prägnanten Zeile zusammen (höchstens {words} Wörter, Deutsch).
- Nenne das konkrete Problem, nicht nur das Thema.
- Keine Namen von Patientinnen, Patienten oder Mitarbeitenden, keine Geburtsdaten.
- Verwende nur die angegebenen Nummern in eckigen Klammern.

Gib JSON zurück:
{{"examples": [{{"id": "<Nummer>", "text": "<eine Zeile>"}}, ...]}}

{unit}:
{items}
"""


def llm_examples(topics, n, max_chars, pool):
    """topics: list of (name, description, group). Returns list of example lists."""
    jobs = []
    for name, desc, g in topics:
        rows = typical_order(g).head(pool)
        tickets = list(zip(rows["incident_id"], rows["text"]))
        fixed = len(P["system"]) + len(_llm_prompt(name, desc, [], n, max_chars)) + 50
        texts, _ = vllm.fit_docs([t for _, t in tickets], fixed, per_item_chars=20)
        jobs.append((name, desc, list(zip([i for i, _ in tickets], texts)), g))

    def run(job):
        name, desc, tickets, _ = job
        ids = {i for i, _ in tickets}

        def valid(o):
            ex = o.get("examples")
            return (isinstance(ex, list) and ex
                    and all(isinstance(e, dict) and str(e.get("id")) in ids and e.get("text")
                            for e in ex))
        o = vllm.chat_json(P["system"], _llm_prompt(name, desc, tickets, n, max_chars),
                           validate=valid)
        seen, out = set(), []
        for e in o["examples"]:
            i = str(e["id"])
            if i not in seen:
                seen.add(i)
                out.append((i, re.sub(r"\s+", " ", str(e["text"])).strip()))
        return out[:n]

    res = vllm.map_concurrent(run, jobs, desc="examples")
    out = []
    for (name, _, _, g), r in zip(jobs, res):
        if isinstance(r, Exception):
            print(f"[warning] LLM examples failed for '{name}', using short examples instead")
            r = pick_short_examples(g, n, max_chars)
        out.append(r)
    return out


# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------

def write_markdown(path, title, header, rows):
    lines = [f"# Themenübersicht: {title}", "", header, ""]
    for r in rows:
        lines.append(f"## {r['rank']}. {r['topic']} ({r['tickets']} {UNIT_NAME}, {r['share']:.1f}%)")
        if r["description"]:
            lines.append(r["description"])
        lines.append("")
        lines += [f"- {iid}: {text}" for iid, text in r["examples"]]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_excel(path, title, header, rows, n, info):
    from openpyxl.styles import Alignment, Font
    table = pd.DataFrame([{
        "Rang": r["rank"], "Thema": r["topic"], UNIT_NAME: r["tickets"],
        "Anteil (%)": round(r["share"], 1), "Beschreibung": r["description"],
        **{f"Beispiel {k + 1}": (f"{r['examples'][k][0]}: {r['examples'][k][1]}"
                                 if k < len(r["examples"]) else "") for k in range(n)},
    } for r in rows])
    meta = pd.DataFrame({"Feld": ["Lauf", "Übersicht"] + list(info.get("params", {}).keys()),
                         "Wert": [title, header] + list(info.get("params", {}).values())})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="Themen", index=False)
        meta.to_excel(xw, sheet_name="Info", index=False)
        ws = xw.sheets["Themen"]
        widths = {"A": 6, "B": 38, "C": 9, "D": 11, "E": 50}
        for k in range(n):
            widths[chr(ord("F") + k)] = 60
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in ws[1]:
            cell.font = Font(bold=True)
        ws.freeze_panes = "C2"
        xw.sheets["Info"].column_dimensions["A"].width = 30
        xw.sheets["Info"].column_dimensions["B"].width = 100


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main(argv=None):
    p = cli.make_parser(__doc__)
    p.add_argument("--run", required=True, help="run folder with assignments.csv and clusters.csv")
    cli.add_flag(p, "--examples", "REPORT_EXAMPLES", "short", str,
                 "short = original short description, llm = condensed by the LLM",
                 choices=["short", "llm"])
    cli.add_flag(p, "--n-examples", "REPORT_N_EXAMPLES", 3, int, "examples per topic")
    cli.add_flag(p, "--max-chars", "REPORT_MAX_CHARS", 120, int,
                 "length of one example line in characters (llm: about this many / 8 words)")
    cli.add_flag(p, "--llm-pool", "REPORT_LLM_POOL", 10, int,
                 "llm mode: number of typical tickets the LLM chooses from")
    cli.add_flag(p, "--input", "CLEANED_FILE", None, str,
                 "cleaned input for the full texts (default: from run_info.json)")
    g = p.add_argument_group("LLM (only for --examples llm)")
    cli.add_flag(g, "--llm-url", "VLLM_LLM_URL", "http://localhost:8001/v1/chat/completions", str,
                 "chat endpoint")
    cli.add_flag(g, "--llm-model", "VLLM_LLM_MODEL", "Qwen/Qwen3.8-27B", str, "chat model")
    cli.add_flag(g, "--llm-concurrency", "LLM_CONCURRENCY", 8, int, "parallel requests")
    cli.add_flag(g, "--llm-context-tokens", "LLM_CONTEXT_TOKENS", 0, int,
                 "context length, 0 = ask the server")
    args = p.parse_args(argv)

    run = Path(args.run)
    a, c, info, fmt = load_run(run, args.input)
    set_profile(fmt)
    global UNIT_NAME
    UNIT_NAME = UNIT.get(fmt, "Dokumente")
    method = info.get("method", run.name)

    if args.examples == "llm":
        cli.check_url(args.llm_url, "--llm-url", "VLLM_LLM_URL")
        vllm.VLLM_LLM_URL, vllm.VLLM_LLM_MODEL = args.llm_url, args.llm_model
        vllm.LLM_CONCURRENCY = args.llm_concurrency
        vllm.DEBUG_DIR = run
        vllm.check_chat_server()
        vllm.setup_context(args.llm_context_tokens)

    # topics by size, Sonstiges/noise last
    c = c.sort_values("size", ascending=False)
    topics = []
    for _, r in c.iterrows():
        g = a[a["cluster_id"] == r["cluster_id"]]
        desc = str(r.get("description", "") if pd.notna(r.get("description", "")) else "")
        topics.append((str(r["cluster_name"]), desc, g))
    noise = a[a["cluster_id"] == -1]
    if len(noise):
        name = noise["cluster_name"].iloc[0] if "cluster_name" in noise else NOISE_NAME
        topics.append((str(name), "passt zu keinem Thema", noise))

    if args.examples == "llm":
        examples = llm_examples(topics, args.n_examples, args.max_chars, args.llm_pool)
    else:
        examples = [pick_short_examples(g, args.n_examples, args.max_chars) for _, _, g in topics]

    total = len(a)
    rows = [{"rank": k + 1, "topic": name, "tickets": len(g), "share": 100 * len(g) / total,
             "description": desc, "examples": ex}
            for k, ((name, desc, g), ex) in enumerate(zip(topics, examples))]
    header = (f"{total} {UNIT_NAME}, {len(c)} Themen"
              + (f", {len(noise)} {topics[-1][0]} ({100 * len(noise) / total:.1f}%)" if len(noise) else "")
              + f". Beispiele: {f'Kurzbeschreibung der typischsten {UNIT_NAME}' if args.examples == 'short' else 'vom LLM gekürzt'}.")

    md = run / f"themenuebersicht_{args.examples}.md"
    xl = run / f"themenuebersicht_{args.examples}.xlsx"
    write_markdown(md, method, header, rows)
    write_excel(xl, method, header, rows, args.n_examples, info)
    print(md.read_text(encoding="utf-8")[:3000])
    print(f"\nSaved: {md}\nSaved: {xl}")


if __name__ == "__main__":
    main()
