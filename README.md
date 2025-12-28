# Cybersecurity Data Processing Pipeline

## **Description**
This project is designed to process and analyze raw cybersecurity logs and events, preparing them for ingestion by large language models (LLMs). The pipeline includes steps for:
- Data preprocessing,
- Clustering,
- Mapping to MITRE ATT&CK techniques,
- Preparing data for machine learning models.

The goal is to enhance ransomware detection, prediction, and automated response, with a focus on the LockBit ransomware family.

---

## **Project Structure**

The project is organized into the following directories:

### **1. `src/` - Source Code**
Contains the main scripts organized by functionality:
- **`clustering.py`**: Implements clustering using KMeans.
- **`fine-tuning.py`**: Fine-tunes the BERT model for ransomware detection.
- **`logger.py`**: Handles logging for the pipeline.
- **`mitre_mapper.py`**: Maps events to MITRE ATT&CK techniques.
- **`saver.py`**: Saves the generated results.
- **`preprocessing/`**: Scripts for data preprocessing.
- **`detection/`**: Scripts for anomaly detection.
- **`evaluation/`**: Scripts for model evaluation.

---

### **2. `data/` - Data**
Contains raw, processed, and LLM-ready data:
- **`raw/`**: Raw data files (e.g., `data_file.csv`, `RansomwareData.csv`).
- **`processed/`**: Cleaned and transformed data (e.g., `dynamic_for_llm.jsonl`).

---

### **3. `output/` - Results**
Stores generated results such as reports, models, and clusters:
- **`cluster_file/`**: Contains clustering results (e.g., `dynamic_clustered.csv`).
- **`mapped_mittre/`**: Contains MITRE-mapped data (e.g., `dynamic_mitre_mapped.csv`).

---

### **4. `logs/` - Logs**
Stores execution logs and configuration files.

---

### **5. `mapping/` - MITRE Mapping**
Contains files for mapping events to MITRE ATT&CK techniques.

---

### **6. `config/` - Configuration**
Contains global configuration files for the project.

---

### **7. `bert_model/` - BERT Model**
Contains files related to the BERT model:
- **`model.safetensors`**: Pre-trained BERT weights.
- **`tokenizer_config.json`**: Tokenizer configuration.

---

## **Installation**

### **Prerequisites**
- Python 3.7 or higher.
- It is recommended to use a virtual environment.

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/ulrich-joel/innovsecu.git
   cd innovsecu
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

To run the main pipeline:
```bash
python src/main.py
```

To execute clustering separately:
```bash
python src/clustering.py
```

---

## **Future Improvements**
- **Large File Handling**: Use Git LFS for managing large files like `model.safetensors` and large CSV files.
- **Documentation**: Add detailed docstrings in each script for better clarity.
- **Unit Testing**: Implement unit tests to validate each step of the pipeline.

---

## **Authors**
This project was developed by Ngueyep Ulrich. For any questions or suggestions, feel free to contact me at ulrich.ngueyepl@cybersearchlab.com
