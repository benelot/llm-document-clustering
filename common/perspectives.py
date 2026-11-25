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


# Versions for hospital-wide CIRS reports (--format cirs). The names stay the
# same; the texts are general instead of medication-specific.
CIRS_PERSPECTIVES = {
    "process": {
        "title": "Ort im Behandlungs- oder Arbeitsablauf",
        "text": (
            "Gruppiere nach dem Ort des Ereignisses im Behandlungs- oder Arbeitsablauf: "
            "welcher Arbeitsschritt, welche Tätigkeit oder welche Schnittstelle betroffen "
            "ist (z.B. Aufnahme, Übergabe, Verordnung, Pflege, Eingriff, Transport, "
            "Entlassung). Die einzelne Person und der genaue Ort im Spital sind für die "
            "Gruppierung nicht entscheidend."
        ),
        "embed": ("Identify the step of the care or work process in which the incident "
                  "described in this hospital CIRS report occurred"),
    },
    "failure-type": {
        "title": "Art des Ereignisses",
        "text": (
            "Gruppiere nach der Art des Ereignisses: was ist schiefgegangen oder beinahe "
            "schiefgegangen, unabhängig davon, in welchem Bereich es auftrat. Zwei "
            "Meldungen gehören zusammen, wenn derselbe Typ von Fehler, Versäumnis oder "
            "Störung vorliegt."
        ),
        "embed": "Identify the type of error, omission or failure described in this CIRS report",
    },
    "root-cause": {
        "title": "Wahrscheinliche Ursache",
        "text": (
            "Gruppiere nach der wahrscheinlichen Ursache: z.B. Kommunikation, fehlende oder "
            "unklare Regeln, Arbeitsbelastung, Wissen und Erfahrung, Geräte und Technik, "
            "Material, Dokumentation oder Organisation. Die beschriebenen Folgen sind "
            "weniger wichtig als ihr Grund."
        ),
        "embed": "Identify the likely root cause of the incident described in this CIRS report",
    },
    "action": {
        "title": "Massnahme und Zuständigkeit",
        "text": (
            "Gruppiere danach, welche Massnahme ähnliche Ereignisse verhindern würde und wer "
            "zuständig wäre: z.B. Schulung, Anpassung einer Richtlinie oder eines Standards, "
            "Checkliste, Kommunikationsregel, technische Änderung, Anpassung von Material "
            "oder Geräten, Personalplanung."
        ),
        "embed": ("Identify which preventive measure and which responsible team would "
                  "prevent the incident described in this CIRS report"),
    },
    "med-process-stage": {
        "title": "Schritt im Behandlungsprozess",
        "text": (
            "Gruppiere nach dem Schritt im Behandlungsprozess, in dem das Ereignis auftritt: "
            "Aufnahme und Anamnese, Diagnostik, Verordnung und Planung, Durchführung von "
            "Therapie, Pflege oder Eingriff, Überwachung und Nachsorge, Übergabe und "
            "Verlegung, Entlassung, oder Logistik und Organisation."
        ),
        "embed": ("Identify the stage of the patient care process (admission, diagnostics, "
                  "ordering and planning, treatment or procedure, monitoring, handover, "
                  "discharge, logistics) in which the incident described in this CIRS "
                  "report occurred"),
    },
    "patient-safety": {
        "title": "Art des Patientensicherheitsereignisses",
        "text": (
            "Gruppiere nach der Art des Patientensicherheitsereignisses: z.B. klinische "
            "Administration, klinischer Prozess oder Eingriff, Dokumentation, Infektion, "
            "Medikation, Blut und Blutprodukte, Ernährung, medizinische Geräte, Verhalten, "
            "Unfälle von Patientinnen und Patienten (z.B. Sturz), Infrastruktur, Ressourcen "
            "und Organisation."
        ),
        "embed": ("Identify the patient safety incident type (e.g. clinical process, "
                  "documentation, infection, medication, medical device, patient accident, "
                  "organisation) of the incident described in this CIRS report"),
    },
    "contributing-factors": {
        "title": "Beitragender Faktor (London Protocol)",
        "text": (
            "Gruppiere nach dem wichtigsten beitragenden Faktor: Patientenfaktoren (Zustand, "
            "Kommunikation), Aufgabe und Technologie (Abläufe, Geräte, Hilfsmittel), Personen "
            "(Wissen, Erfahrung, Belastung), Team (Kommunikation, Übergaben, Aufsicht), "
            "Arbeitsumfeld (Personal, Zeitdruck, Ausstattung) oder Organisation und "
            "Management (Regeln, Ressourcen, Sicherheitskultur)."
        ),
        "embed": ("Identify the main contributing factor (patient, task and technology, "
                  "individual, team, work environment, organisation and management) behind "
                  "the incident described in this CIRS report"),
    },
}


def get_perspective(name, fmt="incidents"):
    """Perspective for a format: the CIRS version if there is one, else the default."""
    if fmt == "cirs" and name in CIRS_PERSPECTIVES:
        return CIRS_PERSPECTIVES[name]
    return PERSPECTIVES.get(name, {})


def perspective_block(text):
    """Paragraph inserted into the LLM prompts; empty for no perspective."""
    if not text:
        return ""
    return f"\nSichtweise für die Gruppierung:\n{text}\n"
