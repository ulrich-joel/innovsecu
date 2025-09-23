# Cybersecurity Data Processing Pipeline

This project processes raw cybersecurity logs/events and prepares them for LLM ingestion, following a professional pipeline.

## Pipeline

1. **Preprocessing**: Cleaning and normalization of datasets.
2. **Behavioral Analysis**: Clustering based on behaviors.
3. **MITRE Mapping**: Mapping events to MITRE ATT&CK techniques.
4. **LLM Preparation**: Formatting data for ingestion by Large Language Models.
5. **Saving**: Saving both LLM-ready files and full mapped datasets.

---

Description of the project.

## Project Structure

project/
│
├── data/
│   ├── raw/                # Données brutes
│   ├── processed/          # Données nettoyées et prêtes à l’usage
│   └── llm_ready/          # Données formatées pour le LLM (NOUVEAU)
│
├── mapping/
│   └── mapping_mitre.json  # Fichier de correspondance MITRE
│
├── output/
│   ├── reports/            # (NOUVEAU) Fichiers de rapport : metrics, stats
│   └── models/             # (NOUVEAU) Modèles sauvegardés, vecteurs, clusters
│
├── src/
│   ├── preprocessing/
│   │   ├── processor_static.py   # (NOUVEAU) Prétraitement des données statiques
│   │   └── processor_dynamic.py  # (NOUVEAU) Prétraitement des logs dynamiques
│   │
│   ├── clustering.py
│   ├── mitre_mapper.py
│   ├── llm_preparer.py
│   ├── saver.py
│   ├── evaluator.py         # (NOUVEAU) Pour les métriques et évaluation
│   ├── feature_engineer.py  # (NOUVEAU) Fusion de vecteurs ou encodage avancé
│   └── main.py
│
└── README.md


## Installation

Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the main script:

```
python src/main.py
