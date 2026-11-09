#!/usr/bin/env python3
"""Write a synthetic EHR survey export (.xlsx or .csv) for smoke tests.

Usage: python tests/make_synthetic_survey.py <out.xlsx|out.csv> [n_respondents]
"""

import sys
import random

import pandas as pd

ANSWERS = [
    "Drucker druckt Etiketten oft nicht, Etikettendrucker langsam",
    "Scanner für Barcode funktioniert schlecht",
    "Perfusor Daten werden nicht korrekt aus der Pumpe übernommen",
    "Zu viele Warnmeldung Popups bei der Verordnung",
    "Warnmeldung Interaktion stört den Arbeitsablauf",
    "Stammdaten der Medikamente sind unvollständig",
    "Ansicht der Medikationsliste ist unübersichtlich",
    "Darstellung auf kleinen Bildschirmen schlecht",
    "Login dauert zu lange, Passwort muss zu oft geändert werden",
    "Bestellung und Verrechnung von Material umständlich",
    "Mehr Schulung zu EPIC für neue Mitarbeitende",
    "KI könnte Berichte und Austrittsberichte vorformulieren",
]
EMPTY = ["", "-", "keine", None]
TEXT_COLS = ["Fix 1", "Fix 2", "Fix 3", "Most valuable improvements", "CommentsGeneral",
             "Disagree IT Seeking to Improve EHR Comments", "IT Department Improve EHR Comments",
             "Useful Training Comment", "AI greatest potential not currently used",
             "Hardware issues description"]


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    rng = random.Random(0)
    rows = []
    for i in range(n):
        r = {
            "RespondentID": 50000 + i,
            "MeasurementName": "EHR Survey 2025",
            "Clinical Background": rng.choice(["Pflege", "Ärzteschaft", "MTT"]),
            "Clinical Background Raw": rng.choice(["Pflegefachperson", "Assistenzarzt", "Physio"]),
            "Average Experience Rating": rng.randint(1, 5),
            "Your Specialty": rng.choice(["Anästhesie", "Kardiologie", "Chirurgie"]),
            "Locations of Use": "Station",
            "EMR": "Epic",
            "PrimaryWork Location": rng.choice(["Inselspital", "Tiefenau"]),
        }
        for c in TEXT_COLS:
            r[c] = rng.choice(ANSWERS) if rng.random() < 0.5 else rng.choice(EMPTY)
        rows.append(r)
    df = pd.DataFrame(rows)
    if out.endswith(".csv"):
        df.to_csv(out, index=False)
    else:
        df.to_excel(out, index=False)
    print(f"Wrote {n} synthetic survey rows to {out}")


if __name__ == "__main__":
    main()
