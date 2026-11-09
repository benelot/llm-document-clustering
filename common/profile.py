"""
Data profile: adapts the LLM prompts to the kind of text being clustered.

  DATA_PROFILE=incidents   (default) IT incident reports, preprocess_incidents.py
  DATA_PROFILE=cirs        CIRS / EHR survey free-text answers, preprocess_cirs.py
"""

import os

PROFILES = {
    "incidents": {
        "system": (
            "Du analysierst IT-Incident-Meldungen eines Spitals (Service Desk, "
            "klinische Systeme wie EPIC, Medikation, Geräte, Drucker usw.). "
            "Antworte ausschliesslich mit gültigem JSON, ohne Markdown."
        ),
        "item": "Meldung",
        "items": "Meldungen",
        "items_long": "Incident-Meldungen",
        "category": "Problemkategorie",
        "granularity_question": (
            "Gehören diese beiden Meldungen zur selben Problemkategorie, d.h. "
            "betreffen sie dasselbe System bzw. dieselbe Funktion und dieselbe Art "
            "von Problem? Unterschiedliche Patienten, Personen oder Zeitpunkte spielen "
            "keine Rolle."
        ),
        "label_en": "Incidents",
    },
    "cirs": {
        "system": (
            "Du analysierst Freitext-Antworten von Spitalmitarbeitenden zum "
            "Klinikinformationssystem (EHR/EPIC), aus CIRS-Meldungen und Befragungen: "
            "Probleme, Verbesserungsvorschläge, Schulung, Hardware, KI-Potenzial. "
            "Die Antworten können Deutsch, Französisch oder Englisch sein. "
            "Antworte ausschliesslich mit gültigem JSON, ohne Markdown."
        ),
        "item": "Antwort",
        "items": "Antworten",
        "items_long": "Freitext-Antworten",
        "category": "Themenkategorie",
        "granularity_question": (
            "Gehören diese beiden Antworten zur selben Themenkategorie, d.h. "
            "sprechen sie dasselbe System bzw. dieselbe Funktion und dieselbe Art "
            "von Problem oder Verbesserungswunsch an? Unterschiedliche Personen, "
            "Fachbereiche oder Formulierungen spielen keine Rolle."
        ),
        "label_en": "Documents",
    },
}

DATA_PROFILE = os.getenv("DATA_PROFILE", "incidents").lower()
if DATA_PROFILE not in PROFILES:
    raise ValueError(f"Unknown DATA_PROFILE: {DATA_PROFILE}. Use one of {list(PROFILES)}.")
P = PROFILES[DATA_PROFILE]
