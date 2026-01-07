# 🛡️ Hybrid Static–Dynamic Malware Detection with MITRE ATT&CK Explainability

## 📌 Overview

This repository implements a **multi-stage hybrid malware detection framework**
combining:

- Static analysis  
- Dynamic analysis  
- Unsupervised anomaly detection  
- Supervised meta-classification  
- MITRE ATT&CK–based explainability  

The pipeline is designed for **research, reproducibility, and SOC-oriented
interpretability**, and follows a **post-detection explainability paradigm**,
where tactical context is assigned **only after high-confidence validation**.

---

## 🧠 Key Contributions

- Hybrid **static + dynamic** detection architecture  
- Multi-level anomaly detection (Isolation Forest, K-Means, VAE)  
- Supervised meta-classifiers for confidence-based alert validation  
- Rule-based **MITRE ATT&CK mapping (post-detection)**  
- Automated visual explainability and ATT&CK Navigator layer generation  
- Fully modular, **publication-ready** codebase  

---

## 🏗️ Project Structure

article1/
│
├── data/
│ ├── raw/ # Raw datasets
│
├── src/
│ ├── preprocessing/ # Data preparation
│ ├── detection/ # Training anomaly & supervised models
│ ├── evaluation/ # Model evaluation & alert generation
│ ├── explainability/ # MITRE mapping, visualization, navigator
│ └── logger.py
│
├── outputs/
│ ├── processed/ # Processed datasets
│ ├── models/ # Trained models
│ ├── evaluation/ # Evaluation metrics & curves
│ ├── explainability/ # Alert-level CSVs
│ └── reports/
│ └── mitre/ # MITRE reports & figures
│
├── figures/ # Figures for paper / LaTeX
├── latex/ # LaTeX sources (article, figures)
├── README.md
└── requirements.txt

---

## 🔄 End-to-End Pipeline

### 1️⃣ Data Preparation

**Static data**
```bash
python -m src.preprocessing.prepare_static_models
Dynamic data

```bash
python -m src.preprocessing.prepare_dynamic_data

### 2️⃣ Unsupervised Anomaly Detection
Static models

```bash
python -m src.detection.train_static_models
Isolation Forest (multiple contamination levels)

K-Means clustering

Dynamic models

```bash
python -m src.detection.train_dynamic_models
Variational Autoencoder (VAE)

### 3️⃣ Unsupervised Model Evaluation
Static

```bash
python -m src.evaluation.evaluate_static_models
Outputs:

Precision / Recall / F1 / ROC-AUC

ROC & PR curves

Confusion matrices

Dynamic

```bash
python -m src.evaluation.evaluate_dynamic_models

### 4️⃣ Supervised Meta-Classification
The meta-classifier aggregates anomaly scores to validate alerts.

Static supervised training

```bash
python -m src.detection.train_static_supervised
Dynamic supervised training

```bash
python -m src.detection.train_dynamic_supervised
Best models are selected automatically based on validation performance.

### 5️⃣ Supervised Evaluation & Alert Generation
Static alerts

```bash
python -m src.evaluation.evaluate_static_supervised
Generates:

```bash
outputs/explainability/static_alerts.csv
Dynamic alerts

```bash
python -m src.evaluation.evaluate_dynamic_supervised
Generates:

```bash
outputs/explainability/dynamic_alerts.csv
These files are the single source of truth for explainability.

🧩 MITRE ATT&CK Explainability Layer
### 6️⃣ MITRE Mapping
Rule-based mapping applied only to validated alerts.

```bash
python -m src.explainability.mitre_mapper --mode static
python -m src.explainability.mitre_mapper --mode dynamic
python -m src.explainability.mitre_mapper --mode final
Outputs:

mitre_*_report.csv

mitre_*_report.json

### 7️⃣ MITRE Visual Analytics

```bash
python -m src.explainability.mitre_visualizer
Generates:

Technique frequency plots

Tactic coverage

Heatmaps

Static vs Dynamic vs Final comparison

Saved in:

swift
Copier le code
outputs/reports/mitre/figures/
### 8️⃣ ATT&CK Navigator Layer

```bash
python -m src.explainability.mitre_navigator_layer
Output:

```bash
outputs/reports/mitre/mitre_attack_layer.json
👉 Directly importable into MITRE ATT&CK Navigator.

## MITRE ATT&CK Visualization

This repository provides an official MITRE ATT&CK Navigator layer generated
from the proposed detection framework.

### Files
- `outputs/reports/mitre/mitre_attack_layer.json`  
  Official ATT&CK Navigator layer (importable at https://mitre-attack.github.io/attack-navigator/)
- `outputs/reports/mitre/figures/mitre_attack_heatmap.svg`  
  Exported ATT&CK heatmap image
- `src/explainability/mitre_navigator_layer.py`  
  Script used to generate the ATT&CK Navigator layer

### How to reproduce
1. Open https://mitre-attack.github.io/attack-navigator/
2. Select **Open Existing Layer**
3. Upload `mitre_attack_layer.json`


📊 Explainability Philosophy
No MITRE labeling without detection confidence

Explainability is post-hoc, rule-based, and reproducible

Avoids misleading tactic attribution from raw anomaly scores

Designed for SOC trust and analyst usability

#📄 Publication & Reproducibility
All scripts include detailed docstrings

Figures are LaTeX-ready

Deterministic seeds where applicable

Fully modular pipeline

This repository is suitable for:

Academic papers

Master / PhD theses

Industrial PoCs

SOC-oriented research

#👤 Author
Ngueyep Ulrich
Cybersecurity & Machine Learning Research
📅 2026

#📜 License
This project is released for research and educational purposes.
For commercial use, please contact the author.
