## Cybersecurity Data Processing Pipeline
# Description

Ce projet vise à traiter et analyser des journaux et événements de cybersécurité bruts, en les préparant pour une ingestion par des modèles de langage (LLMs). Il inclut des étapes de prétraitement, de clustering, de mappage aux techniques MITRE ATT&CK, ainsi que la préparation des données pour les modèles.

# Structure du Projet

La structure des répertoires et des fichiers du projet est organisée comme suit :

<img width="769" height="457" alt="image" src="https://github.com/user-attachments/assets/ad87b2b4-a5c4-4eeb-a8b2-23b5fe782496" />


Détails des Répertoires
1. src/ - Code Source

Contient les scripts principaux organisés par fonctionnalité :

<img width="793" height="385" alt="image" src="https://github.com/user-attachments/assets/72877668-7933-43bc-8ce7-cea07fdbcb9b" />


Exemple : clustering.py

Utilise KMeans pour effectuer un clustering sur les données.

Standardise les données avant d'appliquer le clustering.

Retourne un DataFrame avec les clusters assignés.

# 2. data/ - Données

Contient les données brutes, traitées et prêtes pour l'ingestion par les modèles de langage (LLMs) :

<img width="675" height="120" alt="image" src="https://github.com/user-attachments/assets/1cc7a11f-f017-4ab9-b2d9-503fa02b8814" />


Exemple : raw/

data_file.csv: Données brutes générales.

RansomwareData.csv: Données spécifiques aux ransomwares.

Exemple : processed/

dynamic_for_llm.jsonl: Données dynamiques prêtes pour le LLM.

processed_dynamic/: Données dynamiques nettoyées et transformées.

# 3. output/ - Résultats

Contient les résultats générés tels que les rapports, modèles et clusters :

<img width="652" height="152" alt="image" src="https://github.com/user-attachments/assets/62bd1110-7aa0-439a-ab47-06d55aeb176f" />


Exemple : cluster_file/

dynamic_clustered.csv: Résultats du clustering dynamique.

static_clustered.csv: Résultats du clustering statique.

Exemple : mapped_mittre/

dynamic_mitre_mapped.csv: Données dynamiques mappées aux techniques MITRE.

dynamic_mitre_mapped_with_labels.json: Données mappées avec des labels.

# 4. logs/ - Journaux

Contient les journaux d'exécution et les fichiers de configuration :

<img width="641" height="155" alt="image" src="https://github.com/user-attachments/assets/da6c2b41-7c40-42c2-afe3-e8f8892abaad" />


# 5. mapping/ - Correspondance MITRE

Contient les fichiers de correspondance pour mapper les événements aux techniques MITRE ATT&CK :

<img width="619" height="124" alt="image" src="https://github.com/user-attachments/assets/da3c390f-5d09-4b1c-b7ab-5ab1462ed825" />


# 6. config/ - Configuration

Contient les fichiers de configuration globaux :

<img width="629" height="68" alt="image" src="https://github.com/user-attachments/assets/614b18a6-abe1-4a5a-8e78-3ae33d847622" />


# 7. bert_model/ - Modèle BERT

Contient les fichiers associés au modèle BERT :

bert_model/
<img width="604" height="172" alt="image" src="https://github.com/user-attachments/assets/e9fa2140-3219-4eec-b6cf-024c5f0041bc" />

Installation
Prérequis

Assurez-vous que vous avez Python 3.7 ou supérieur installé. Il est également recommandé d'utiliser un environnement virtuel.

Clonez le dépôt :

git clone https://github.com/ulrich-joel/innovsecu.git
cd innovsecu


Installez les dépendances :
`` 
pip install -r requirements.txt
``
Utilisation

Lancez le script principal :
```
python src/main.py
```

Pour exécuter le clustering :
```
python src/clustering.py
```
# Améliorations Possibles

Gestion des fichiers volumineux : Utilisez Git LFS
 pour gérer les fichiers volumineux comme model.safetensors et les fichiers CSV volumineux.

Documentation : Ajouter des docstrings détaillées dans chaque script pour mieux expliquer leur rôle et leur fonctionnement.

Tests unitaires : Implémenter des tests unitaires pour valider chaque étape du pipeline.

Auteurs

Ce projet a été développé par Ngueyep Ulrich. Pour toute question ou suggestion, n'hésitez pas à me contacter via ulrich.ngueyepl@cybersearchlab.com
