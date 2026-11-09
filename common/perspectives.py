"""
Clustering perspectives: the viewpoint along which the LLM groups the texts.

The same texts can be grouped correctly along different axes (subject,
place in the workflow, cause, ...). A perspective adds one paragraph to the
LLM prompts and sets the embedding instruction for the UMAP plot. It never
names concrete clusters.

  text   German paragraph for the LLM prompts ("" = no extra text)
  embed  English task for instruction-aware embedding models
         (None = the format's default instruction from common/profile.py)
"""

PERSPECTIVES = {
    "subject": {
        "title": "Thema (bisheriges Verhalten)",
        "text": "",
        "embed": None,
    },
    "process": {
        "title": "Ort im System oder Arbeitsablauf",
        "text": (
            "Gruppiere nach dem Ort des Problems: welche Funktion, welche Ansicht, "
            "welches Gerät oder welcher Schritt im Arbeitsablauf betroffen ist. "
            "Medikament, Station und Fachrichtung sind für die Gruppierung nicht "
            "entscheidend."
        ),
        "embed": ("Identify where in the clinical IT system or workflow the problem "
                  "described in this text occurs"),
    },
    "failure-type": {
        "title": "Art des Problems",
        "text": (
            "Gruppiere nach der Art des Problems: was geht schief, unabhängig davon, in "
            "welchem Modul oder bei welchem Medikament es auftritt. Zwei Texte gehören "
            "zusammen, wenn derselbe Typ von Fehler oder Störung vorliegt."
        ),
        "embed": "Identify the type of failure or malfunction described in this text",
    },
    "root-cause": {
        "title": "Wahrscheinliche Ursache",
        "text": (
            "Gruppiere nach der wahrscheinlichen Ursache: z.B. Konfiguration, Stammdaten, "
            "Softwarefehler, Schnittstelle, Hardware, fehlendes Wissen der Anwendenden "
            "oder Prozess und Organisation. Die beschriebenen Symptome sind weniger "
            "wichtig als ihr Grund."
        ),
        "embed": "Identify the likely root cause of the problem described in this text",
    },
    "action": {
        "title": "Massnahme und Zuständigkeit",
        "text": (
            "Gruppiere danach, welche Massnahme das Problem lösen würde und wer zuständig "
            "wäre: z.B. Stammdatenpflege, Systemkonfiguration, Fehlerbehebung durch den "
            "Hersteller, Hardware-Support, Schulung oder Prozessanpassung."
        ),
        "embed": ("Identify which corrective action and which responsible team would "
                  "resolve the problem described in this text"),
    },
    "med-process-stage": {
        "title": "Schritt im Medikationsprozess",
        "text": (
            "Gruppiere nach dem Schritt im Medikationsprozess, in dem das Problem "
            "auftritt: Verordnung, Validierung durch die Pharmazie, Richten und "
            "Dispensation, Verabreichung, Überwachung und Dokumentation, oder Logistik "
            "und Beschaffung."
        ),
        "embed": ("Identify the stage of the medication use process (prescribing, pharmacy "
                  "validation, dispensing, administration, monitoring and documentation, "
                  "logistics) affected by the problem described in this text"),
    },
    "patient-safety": {
        "title": "Art des Medikationsfehlers",
        "text": (
            "Gruppiere nach der Art des Medikationsfehlers, den das Problem verursacht "
            "oder ermöglicht: z.B. falsches Medikament, falsche Dosis, falscher Weg, "
            "falscher Zeitpunkt, falscher Patient, Auslassung, Doppelverordnung, nicht "
            "erkannte Interaktion, fehlende oder falsche Information."
        ),
        "embed": ("Identify the type of medication error that the problem described in "
                  "this text causes or enables"),
    },
    "contributing-factors": {
        "title": "Beitragender Faktor (London Protocol)",
        "text": (
            "Gruppiere nach dem beitragenden Faktor im System: Aufgabe und Technologie "
            "(Software, Geräte, Usability), Personen (Wissen, Erfahrung), Team und "
            "Kommunikation, Arbeitsumfeld (Zeitdruck, Ressourcen) oder Organisation "
            "(Regeln, Prozesse)."
        ),
        "embed": ("Identify the contributing factor (task and technology, individual, team "
                  "and communication, work environment, organisation) behind the problem "
                  "described in this text"),
    },
}


def perspective_block(text):
    """Paragraph inserted into the LLM prompts; empty for no perspective."""
    if not text:
        return ""
    return f"\nSichtweise für die Gruppierung:\n{text}\n"
