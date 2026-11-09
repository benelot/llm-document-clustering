#!/usr/bin/env python3
"""Write a synthetic cleaned_incidents.json for smoke tests (no real data).

Usage: python tests/make_synthetic_data.py <out.json> [n]
"""

import sys
import json
import random

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from common.gold import load_gold  # noqa: E402
EXAMPLES = load_gold(__file__.rsplit("/", 2)[0] + "/gold/incidents.json")

# topic key -> (hypothesis category or None, text templates)
TOPICS = {
    "drucker": ("Drucken und Scannen", [
        "Drucker auf Station {s} druckt keine Etiketten mehr",
        "Etikettendrucker {s} meldet Papierstau, Drucker offline",
        "Scanner liest Barcode nicht, Drucker Warteschlange blockiert"]),
    "perfusor": ("Pumpen / Perfusor", [
        "Perfusor Einheit mg statt mcg bei Remifentanil auf {s}",
        "Perfusor Laufrate in EPIC falsch übernommen, Pumpe {s}",
        "Pumpe zeigt andere Perfusor Konzentration als EPIC"]),
    "warnung": ("Warnmeldung", [
        "Warnmeldung Interaktion erscheint bei jeder Verordnung {s}",
        "Unnötige Warnmeldung Doppelverordnung auf {s}",
        "Warnmeldung Allergie kann nicht quittiert werden"]),
    "stammdaten": ("Medikationsstammdaten", [
        "Medikament {s} fehlt in Stammdaten, nicht verordnungsfähig",
        "Stammdaten Medikament falsche Dosierungseinheit hinterlegt",
        "Neues Medikament in Stammdaten erfassen {s}"]),
    "bestellung": ("Bestellung / Verrechnung", [
        "Bestellung Apotheke wird nicht verrechnet {s}",
        "Verrechnung Medikament doppelt auf Fall {s}",
        "Bestellung an Apotheke bleibt hängen"]),
    "ansicht": ("Ansichten und Darstellung", [
        "Ansicht Medikationsliste Spalten fehlen {s}",
        "Darstellung Verlauf in EPIC verschoben, Ansicht leer",
        "Ansicht Kurve zeigt Medikamente nicht an {s}"]),
    "login": (None, [
        "Login EPIC nicht möglich, Passwort abgelaufen {s}",
        "Konto gesperrt nach Login Versuch {s}",
        "Passwort Reset für Login benötigt"]),
    "telefon": (None, [
        "Telefon auf Station {s} ohne Signal",
        "Telefon Weiterleitung funktioniert nicht {s}",
        "Telefon Display defekt"]),
}
CAT_TO_TOPIC = {cat: t for t, (cat, _) in TOPICS.items() if cat}


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    rng = random.Random(0)
    rows, used = [], set()

    def add(iid, topic):
        tpl = rng.choice(TOPICS[topic][1])
        txt = tpl.format(s=rng.choice(["B3", "C1", "IPS", "Notfall", "OPS"]))
        rows.append({
            "incident_id": iid, "priority": rng.choice(["1 - Hoch", "2 - Mittel", "3 - Mässig", "4 - Niedrig"]),
            "status": "Geschlossen", "assignment_group": "APPL_EPIC_WILLOW",
            "department": rng.choice(["Anästhesie", "Kardiologie", "Pflege"]),
            "consolidated_text": f"SHORT_DESC: {txt}\n\nDESCRIPTION: {txt}. Bitte prüfen.",
            "created_at": "2025-01-01 10:00:00",
        })
        used.add(iid)

    for cat, incs in EXAMPLES.items():
        for iid in incs:
            if iid not in used and cat in CAT_TO_TOPIC:
                add(iid, CAT_TO_TOPIC[cat])
    i = 2000000
    while len(rows) < n:
        add(f"INC{i}", rng.choice(list(TOPICS)))
        i += 1
    rng.shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(rows)} synthetic incidents to {out}")


if __name__ == "__main__":
    main()
