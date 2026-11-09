"""
Data profile: adapts the LLM prompts to the kind of text being clustered.

  --format incidents   (default) ServiceNow incident reports, preprocess_incidents.py
  --format survey      EHR survey among clinicians, preprocess_survey.py
(environment fallback: DATA_PROFILE)
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
            "Gehören diese beiden Meldungen zum selben übergeordneten Thema, so dass "
            "man sie in einer Übersicht über alle Meldungen in dieselbe Gruppe einordnen "
            "würde? Sie müssen nicht dasselbe Detailproblem beschreiben. Unterschiedliche "
            "Patienten, Personen, Medikamente oder Zeitpunkte spielen keine Rolle."
        ),
        "label_en": "Incidents",
        "embed_instruction": ("Identify the technical topic or problem category of this "
                              "hospital IT incident report"),
    },
    "survey": {
        "system": (
            "Du analysierst Freitext-Antworten aus einer Befragung von Klinikerinnen "
            "und Klinikern zum Klinikinformationssystem (EHR/EPIC): "
            "Probleme, Verbesserungsvorschläge, Schulung, Hardware, KI-Potenzial. "
            "Die Antworten können Deutsch, Französisch oder Englisch sein. "
            "Antworte ausschliesslich mit gültigem JSON, ohne Markdown."
        ),
        "item": "Antwort",
        "items": "Antworten",
        "items_long": "Freitext-Antworten",
        "category": "Themenkategorie",
        "granularity_question": (
            "Gehören diese beiden Antworten zum selben übergeordneten Thema, so dass "
            "man sie in einer Übersicht über alle Antworten in dieselbe Gruppe einordnen "
            "würde? Sie müssen nicht dasselbe Detail ansprechen. Unterschiedliche "
            "Personen, Fachbereiche oder Formulierungen spielen keine Rolle."
        ),
        "label_en": "Documents",
        "embed_instruction": ("Identify the topic of this clinician's feedback on the "
                              "electronic health record system"),
    },
}

P = {}  # the active profile; updated in place, so "from common.profile import P" stays valid


def set_profile(name):
    global DATA_PROFILE
    name = (name or "incidents").lower()
    if name not in PROFILES:
        raise ValueError(f"Unknown format/profile: {name}. Use one of {list(PROFILES)}.")
    DATA_PROFILE = name
    P.clear()
    P.update(PROFILES[name])


DATA_PROFILE = None
set_profile(os.getenv("DATA_PROFILE", "incidents"))
