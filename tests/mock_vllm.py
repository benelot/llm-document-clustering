#!/usr/bin/env python3
"""
Mock vLLM server for smoke tests. Answers /v1/embeddings and
/v1/chat/completions with fake but plausible results, based on keywords of
tests/make_synthetic_data.py. About 5% of chat answers are broken JSON, to
test retries.

Usage: python tests/mock_vllm.py [port]      (default 8099)
"""

import re
import sys
import json
import random
import hashlib
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

KEYS = {
    "drucker": "Drucker und Etiketten", "scanner": "Drucker und Etiketten",
    "perfusor": "Perfusor Einheiten", "pumpe": "Perfusor Einheiten",
    "warnmeldung": "Warnmeldungen Verordnung",
    "stammdaten": "Medikamenten-Stammdaten",
    "bestellung": "Bestellung und Verrechnung", "verrechnung": "Bestellung und Verrechnung",
    "ansicht": "Ansichten in EPIC", "darstellung": "Ansichten in EPIC",
    "login": "Login und Passwort", "passwort": "Login und Passwort",
    "schulung": "Schulung", "ki ": "KI-Potenzial",
}
VARIANTS = {  # raw label variants, to test merging
    "Drucker und Etiketten": ["Drucker und Etiketten", "Druckerprobleme", "Etikettendruck"],
    "Perfusor Einheiten": ["Perfusor Einheiten", "Perfusor Dosierung"],
}
rnd = random.Random(1)


def topic(text):
    t = text.lower()
    for k, v in KEYS.items():
        if k in t:
            return v
    return None


def embed(text):
    v = [0.0] * 64
    for w in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % 64] += 1.0
    tp = topic(text)
    if tp:
        h = int(hashlib.md5(tp.encode()).hexdigest(), 16)
        for i in range(4):
            v[(h >> (i * 6)) % 64] += 4.0
    return v


def items(prompt):
    return re.findall(r"^\[(\d+)\] (.*)$", prompt, flags=re.M)


def answer(user):
    if "Hypothesen-Cluster:" in user:  # hypothesis name check
        cat = re.search(r"Hypothesen-Cluster: (.*)", user).group(1)
        name = re.search(r"Cluster-Name: (.*)", user).group(1)
        same = topic(cat) is not None and topic(cat) == topic(name)
        return {"match": "passt" if same else "passt nicht", "reason": "mock"}
    if "Finde die Themen" in user:
        labs = {rnd.choice(VARIANTS.get(topic(t), [topic(t)])) for _, t in items(user) if topic(t)}
        return {"labels": sorted(labs)}
    if "Fasse Labels zusammen, die dasselbe Thema meinen (Synonyme" in user:
        labs = re.findall(r"^- (.*)$", user, flags=re.M)
        return {"labels": sorted({topic(l) or l for l in labs})}
    if "finale Themenliste" in user:
        labs = [re.sub(r" \(\d+x\)$", "", l) for l in re.findall(r"^- (.*)$", user, flags=re.M)]
        canon = []
        for l in labs:
            c = l
            for k, vs in VARIANTS.items():
                if l in vs:
                    c = k
            if c not in canon:
                canon.append(c)
        return {"labels": [{"name": c, "description": f"Meldungen zu {c}"} for c in canon]}
    if "Ordne jede" in user:
        themes = dict((name.split(" - ")[0], int(n)) for n, name in
                      re.findall(r"^(\d+): (.*)$", user, flags=re.M))
        return {"assignments": {i: themes.get(topic(t), 0) for i, t in items(user)}}
    m = re.search(r"\n(\w+) A:\n(.*)\n\1 B:\n(.*)", user, flags=re.S)
    if m:
        a, b = m.group(2), m.group(3)
        return {"same": topic(a) is not None and topic(a) == topic(b)}
    if "Benenne das gemeinsame Thema" in user:
        body = re.split(r"\n(?:Meldungen|Antworten):\n", user)[-1]
        tops = [topic(t) for t in body.split("---")]
        tops = [t for t in tops if t] or ["Diverses"]
        name = max(set(tops), key=tops.count)
        return {"cluster_name": name, "description": f"Gemeinsames Thema: {name}"}
    if "Analyze the following cluster" in user:  # naive/cluster_incidents.py
        tops = [topic(t) for t in re.findall(r"^Text: (.*)$", user, flags=re.M)]
        tops = [t for t in tops if t] or ["Diverses"]
        name = max(set(tops), key=tops.count)
        return {"cluster_name": name, "root_cause": "mock", "affected_area": "mock",
                "recommended_measure": "mock", "urgency": "medium", "rationale": "mock",
                "most_relevant_incident_ids": []}
    return {"error": "unknown prompt"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            data = [{"index": i, "embedding": embed(t)} for i, t in enumerate(body["input"])]
            out = {"data": data}
        else:
            user = body["messages"][-1]["content"]
            content = json.dumps(answer(user), ensure_ascii=False)
            if rnd.random() < 0.05:
                content = content[: len(content) // 2]  # broken JSON
            out = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"mock vLLM on :{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
