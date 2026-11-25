#!/usr/bin/env python3
"""Write a synthetic hospital-wide CIRS export (.xlsx or .csv) for smoke tests.

Usage: python tests/make_synthetic_cirs.py <out.xlsx|out.csv> [n_cases]
"""

import sys
import random

import pandas as pd

# (Hauptkategorie, Subkategorie, Massnahme, title, what happened, what was done, prevention)
CASES = [
    ("Medikation", "Dosierung", "Schulung",
     "Perfusor falsch programmiert", "Perfusor lief mit falscher Rate, mg statt mcg",
     "Arzt informiert, Perfusor neu eingestellt", "Doppelkontrolle bei Perfusoren"),
    ("Medikation", "Verordnung", "Prozessanpassung",
     "Warnmeldung übersehen", "Warnmeldung zur Interaktion wurde weggeklickt",
     "Verordnung angepasst", "Warnmeldungen reduzieren und schärfen"),
    ("Sturz", "Sturz aus dem Bett", "Material",
     "Sturz in der Nacht", "Patient bei Toilettengang gestürzt, Sturz ohne Verletzung",
     "Patient untersucht, Bettgitter hochgestellt", "Sturzrisiko bei Eintritt erfassen"),
    ("Hygiene", "Händehygiene", "Schulung",
     "Hygiene bei Verbandwechsel", "Händedesinfektion vor Verbandwechsel ausgelassen, Hygiene mangelhaft",
     "Team informiert", "Hygiene-Schulung wiederholen"),
    ("Kommunikation", "Übergabe", "Prozessanpassung",
     "Information bei Übergabe fehlte", "Bei der Übergabe wurde die Allergie nicht erwähnt",
     "Allergie nachgetragen", "Strukturierte Übergabe einführen"),
    ("Geräte", "Drucker und Scanner", "Technische Änderung",
     "Etikettendrucker defekt", "Drucker druckte keine Etiketten für Laborproben",
     "Proben von Hand beschriftet", "Ersatzdrucker bereitstellen"),
]


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rng = random.Random(0)
    rows = []
    for i in range(n):
        h, s, m, title, what, done, prev = rng.choice(CASES)
        rows.append({
            "Fallnummer": 70000 + i,
            "Erfassungsdatum": pd.Timestamp("2025-01-01") + pd.Timedelta(days=rng.randint(0, 300)),
            "Abschlussdatum": pd.Timestamp("2025-11-01"),
            "Status": rng.choice(["abgeschlossen", "in Bearbeitung"]),
            "cirs_verwertbar": rng.choice(["ja", "nein"]),
            "Hauptkategorie": h,
            "Subkategorie": s,
            "Sub Subkategorie": rng.choice(["A", "B"]),
            "Falltitel": title,
            "Massnahmenklassifikation": m,
            "Was ist passiert?": f"{what} auf Station {rng.choice(['B3', 'C1', 'IPS'])}.",
            "Was wurde getan?": done if rng.random() < 0.8 else "",
            "Wie könnten ähnliche Ereignisse künftig verhindert werden?": prev,
            "Zusammenfassung / Wichtigstes": rng.choice(["", title]),
            "Beruf": rng.choice(["Pflege", "Ärzteschaft", "MTT"]),
            "Häufigkeit": rng.choice(["selten", "monatlich", "wöchentlich"]),
            "Relevanz": rng.choice(["gering", "mittel", "hoch"]),
            "spital": "Inselspital",
            "medizinbereich": rng.choice(["Chirurgie", "Medizin", "Pädiatrie"]),
            "Meldekreis": rng.choice(["Pflege", "Ärzteschaft"]),
            "bereich": rng.choice(["Station", "Notfall", "OP"]),
        })
    df = pd.DataFrame(rows)
    if out.endswith(".csv"):
        df.to_csv(out, index=False)
    else:
        df.to_excel(out, index=False)
    print(f"Wrote {n} synthetic CIRS cases to {out}")


if __name__ == "__main__":
    main()
