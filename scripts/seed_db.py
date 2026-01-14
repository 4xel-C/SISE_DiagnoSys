"""
Database Seed Script.

Initializes the database with dummy data for testing purposes.
Run with: python -m scripts.seed_db
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import db
from app.models import Document, Patient


def seed_patients() -> None:
    """Insert 5 dummy patients."""
    patients_data = [
        {
            "nom": "Rakotomala",
            "prenom": "Ricco",
            "gravite": "vert",
            "type_maladie": "Grippe saisonniere",
            "symptomes_exprimes": "fievre legere, toux, fatigue",
            "fc": 72,
            "fr": 16,
            "spo2": 98,
            "ta_systolique": 120,
            "ta_diastolique": 80,
            "temperature": 37.8,
        },
        {
            "nom": "Dubois",
            "prenom": "Sophie",
            "gravite": "jaune",
            "type_maladie": "Pneumonie",
            "symptomes_exprimes": "toux productive, douleur thoracique, fievre",
            "fc": 95,
            "fr": 22,
            "spo2": 93,
            "ta_systolique": 135,
            "ta_diastolique": 85,
            "temperature": 38.9,
        },
        {
            "nom": "Bernard",
            "prenom": "Michel",
            "gravite": "rouge",
            "type_maladie": "Infarctus du myocarde",
            "symptomes_exprimes": "douleur thoracique irradiante, dyspnee, sueurs",
            "fc": 120,
            "fr": 28,
            "spo2": 88,
            "ta_systolique": 90,
            "ta_diastolique": 60,
            "temperature": 37.2,
        },
        {
            "nom": "Petit",
            "prenom": "Claire",
            "gravite": "vert",
            "type_maladie": "Entorse cheville",
            "symptomes_exprimes": "douleur, gonflement, difficulte a marcher",
            "fc": 68,
            "fr": 14,
            "spo2": 99,
            "ta_systolique": 118,
            "ta_diastolique": 75,
            "temperature": 36.8,
        },
        {
            "nom": "Leroy",
            "prenom": "Pierre",
            "gravite": "jaune",
            "type_maladie": "Diabete decompense",
            "symptomes_exprimes": "polyurie, polydipsie, fatigue intense",
            "fc": 88,
            "fr": 20,
            "spo2": 96,
            "ta_systolique": 145,
            "ta_diastolique": 92,
            "temperature": 37.0,
        },
    ]

    with db.session() as session:
        for data in patients_data:
            patient = Patient(**data)
            session.add(patient)
        session.commit()
        print(f"Inserted {len(patients_data)} patients.")


def seed_documents():
    """Insert 5 dummy documents."""
    documents_data = [
        {
            "titre": "Guidelines for the Management of Acute Coronary Syndromes",
            "contenu": "Les syndromes coronariens aigus necessitent une prise en charge rapide. Le traitement initial comprend l'aspirine, les anticoagulants et la reperfusion coronaire.",
            "url": "https://www.escardio.org/guidelines/acs-2023.pdf",
        },
        {
            "titre": "WHO Guidelines on Pneumonia Treatment",
            "contenu": "La pneumonie communautaire se traite par antibiotherapie. Amoxicilline en premiere intention, macrolides en cas d'allergie.",
            "url": "https://www.who.int/publications/pneumonia-guidelines.pdf",
        },
        {
            "titre": "Diabetes Mellitus Type 2: Diagnosis and Treatment",
            "contenu": "Le diabete de type 2 se caracterise par une resistance a l'insuline. Metformine en premiere ligne, objectif HbA1c < 7%.",
            "url": "https://www.diabetes.org/clinical-guidelines-2024.pdf",
        },
        {
            "titre": "Emergency Triage Protocols - French SAMU Guidelines",
            "contenu": "Classification des urgences: rouge (immediat), jaune (urgent), vert (peut attendre), gris (deces ou soins palliatifs).",
            "url": "https://www.samu.fr/protocols/triage-2024.pdf",
        },
        {
            "titre": "Influenza: Clinical Presentation and Management",
            "contenu": "La grippe se manifeste par fievre, myalgies, toux. Traitement symptomatique, antiviraux si patient a risque.",
            "url": "https://www.cdc.gov/flu/clinical-guidance.pdf",
        },
    ]

    with db.session() as session:
        for data in documents_data:
            document = Document(**data)
            session.add(document)
        session.commit()
        print(f"Inserted {len(documents_data)} documents.")


def main():
    """Initialize database and seed with dummy data."""
    print("Initializing database...")
    db.init_db()

    print("Seeding patients...")
    seed_patients()

    print("Seeding documents...")
    seed_documents()

    print("Done!")


if __name__ == "__main__":
    main()
