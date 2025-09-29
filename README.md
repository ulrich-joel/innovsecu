Cybersecurity Data Processing Pipeline
Description

Ce projet vise à traiter et analyser des journaux et événements de cybersécurité bruts, en les préparant pour une ingestion par des modèles de langage (LLMs). Il inclut des étapes de prétraitement, de clustering, de mappage aux techniques MITRE ATT&CK, ainsi que la préparation des données pour les modèles.

Structure du Projet

La structure des répertoires et des fichiers du projet est organisée comme suit :

innovsecu/
│
├── .gitattributes          # Configuration pour Git LFS (Large File Storage)
├── .gitignore              # Fichiers et dossiers à exclure du suivi Git
├── README.md               # Documentation principale du projet
├── requirements.txt        # Dépendances nécessaires pour exécuter le projet
├── src/                    # Code source principal
├── data/                   # Données brutes, traitées et prêtes pour le LLM
├── output/                 # Résultats générés (rapports, modèles, clusters)
├── logs/                   # Journaux d'exécution et fichiers de configuration
├── mapping/                # Fichiers de correspondance MITRE ATT&CK
├── config/                 # Fichiers de configuration du projet
├── bert_model/             # Modèles et fichiers associés au modèle BERT
├── venv/                   # Environnement virtuel Python
└── temp.pt                 # Fichier temporaire volumineux (à exclure)

Détails des Répertoires
1. src/ - Code Source

Contient les scripts principaux organisés par fonctionnalité :

src/
├── clustering.py           # Implémentation du clustering avec KMeans
├── logger.py               # Gestion des logs
├── main.py                 # Point d'entrée principal du projet
├── mitre_mapper.py         # Mappage des événements aux techniques MITRE ATT&CK
├── saver.py                # Sauvegarde des résultats
├── tokenizer.py            # Gestion des tokenizers pour les modèles
├── fine-tuning.py          # Fine-tuning des modèles
├── setup.py                # Script de configuration
├── preprocessing/          # Scripts de prétraitement des données
├── detection/              # Scripts pour la détection des anomalies
├── evaluation/             # Scripts pour l'évaluation des modèles
└── inference/              # Scripts pour l'inférence

Exemple : clustering.py

Utilise KMeans pour effectuer un clustering sur les données.

Standardise les données avant d'appliquer le clustering.

Retourne un DataFrame avec les clusters assignés.

2. data/ - Données

Contient les données brutes, traitées et prêtes pour l'ingestion par les modèles de langage (LLMs) :

data/
├── raw/                    # Données brutes (ex. fichiers CSV)
├── processed/              # Données nettoyées et transformées
└── llm_ready/              # Données formatées pour les modèles LLM

Exemple : raw/

data_file.csv: Données brutes générales.

RansomwareData.csv: Données spécifiques aux ransomwares.

Exemple : processed/

dynamic_for_llm.jsonl: Données dynamiques prêtes pour le LLM.

processed_dynamic/: Données dynamiques nettoyées et transformées.

3. output/ - Résultats

Contient les résultats générés tels que les rapports, modèles et clusters :

output/
├── cluster_file/           # Résultats du clustering
├── mapped_mittre/          # Fichiers mappés aux techniques MITRE
├── models/                 # Modèles sauvegardés
└── reports/                # Rapports générés

Exemple : cluster_file/

dynamic_clustered.csv: Résultats du clustering dynamique.

static_clustered.csv: Résultats du clustering statique.

Exemple : mapped_mittre/

dynamic_mitre_mapped.csv: Données dynamiques mappées aux techniques MITRE.

dynamic_mitre_mapped_with_labels.json: Données mappées avec des labels.

4. logs/ - Journaux

Contient les journaux d'exécution et les fichiers de configuration :

logs/
├── anomaly_detection.log   # Logs pour la détection d'anomalies
├── pipeline.log            # Logs de la pipeline
├── training_log.txt        # Logs d'entraînement des modèles
└── model.safetensors       # Modèle sauvegardé (volumineux)

5. mapping/ - Correspondance MITRE

Contient les fichiers de correspondance pour mapper les événements aux techniques MITRE ATT&CK :

mapping/
├── attack_enterprise.json  # Données MITRE ATT&CK
├── mapping_mitre.json      # Fichier de correspondance
└── gen.py                  # Script pour générer des mappings

6. config/ - Configuration

Contient les fichiers de configuration globaux :

config/
└── config.py               # Configuration principale du projet

7. bert_model/ - Modèle BERT

Contient les fichiers associés au modèle BERT :

bert_model/
├── config.json             # Configuration du modèle BERT
├── model.safetensors       # Poids du modèle BERT
├── tokenizer_config.json   # Configuration du tokenizer
└── vocab.txt               # Vocabulaire du tokenizer

Installation
Prérequis

Assurez-vous que vous avez Python 3.7 ou supérieur installé. Il est également recommandé d'utiliser un environnement virtuel.

Clonez le dépôt :

git clone https://github.com/ulrich-joel/innovsecu.git
cd innovsecu


Installez les dépendances :

pip install -r requirements.txt

Utilisation

Lancez le script principal :

python src/main.py


Pour exécuter le clustering :

python src/clustering.py

Améliorations Possibles

Gestion des fichiers volumineux : Utilisez Git LFS
 pour gérer les fichiers volumineux comme model.safetensors et les fichiers CSV volumineux.

Documentation : Ajouter des docstrings détaillées dans chaque script pour mieux expliquer leur rôle et leur fonctionnement.

Tests unitaires : Implémenter des tests unitaires pour valider chaque étape du pipeline.

Auteurs

Ce projet a été développé par Ngueyep Ulrich. Pour toute question ou suggestion, n'hésitez pas à me contacter via ulrich.ngueyepl@cybersearchlab.com
.