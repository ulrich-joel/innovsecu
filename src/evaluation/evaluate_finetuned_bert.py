#src.evaluation/evaluate_finetuned_bert.py
import torch
import os
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import time

# === CONFIGURATION ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "logs"
val_data_path = "output/models/train_llm_bert/validation_data.pt"
batch_size = 4

# === LOG SIMPLE ===
def log_message(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {msg}")

# === DATASET CLASS ===
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# === CHARGEMENT DU MODELE & TOKENIZER ===
log_message("📂 Chargement du modèle et du tokenizer...")
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.to(device)
model.eval()

# === CHARGEMENT DES DONNÉES DE VALIDATION ===
log_message("📥 Chargement des données de validation...")
val_data = torch.load(val_data_path)
val_dataset = CustomDataset(val_data["input_ids"], val_data["attention_mask"], val_data["labels"])
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === ÉVALUATION ===
log_message("🧪 Début de l'évaluation...")
preds, truths = [], []
start_time = time.time()
max_eval_duration = 60 * 60

with torch.no_grad():
    for batch in val_loader:
        if time.time() - start_time > max_eval_duration:
            log_message("⏱️ Temps d'évaluation dépassé.")
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        preds.extend(predictions.cpu().tolist())
        truths.extend(labels.cpu().tolist())

# === RAPPORT FINAL ===
acc = accuracy_score(truths, preds)
log_message(f"✅ Accuracy : {acc:.4f}")
log_message("📊 Rapport de classification :\n" + classification_report(truths, preds, digits=4))
log_message(f"🧩 Matrice de confusion :\n{confusion_matrix(truths, preds)}")


# import torch
# import os
# import argparse
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertForSequenceClassification, BertTokenizer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from datetime import datetime
# import time

# # === ARGUMENTS ===
# parser = argparse.ArgumentParser(description="Évaluation d’un modèle BERT fine-tuné à un shard donné.")
# parser.add_argument("--shard", type=int, required=True, help="Numéro du shard à évaluer (ex: 25 pour epoch_shard_25)")
# parser.add_argument("--model_base_dir", type=str, default="logs", help="Dossier contenant les checkpoints")
# parser.add_argument("--val_data_path", type=str, default="output/models/train_llm_bert/validation_data.pt", help="Chemin vers les données de validation")
# parser.add_argument("--batch_size", type=int, default=4, help="Taille de batch pour l’évaluation")
# args = parser.parse_args()

# # === CONFIGURATION ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_path = os.path.join(args.model_base_dir, "checkpoints", f"epoch_shard_{args.shard}")
# tokenizer_path = args.model_base_dir

# # === LOG SIMPLE ===
# def log_message(msg):
#     timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
#     print(f"{timestamp} {msg}")

# # === DATASET CLASS ===
# class CustomDataset(Dataset):
#     def __init__(self, input_ids, attention_mask, labels):
#         self.input_ids = input_ids
#         self.attention_mask = attention_mask
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return {
#             "input_ids": self.input_ids[idx],
#             "attention_mask": self.attention_mask[idx],
#             "labels": self.labels[idx]
#         }

# # === CHARGEMENT DU MODELE & TOKENIZER ===
# log_message(f"📂 Chargement du modèle depuis {model_path}")
# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
# model.to(device)
# model.eval()

# # === CHARGEMENT DES DONNÉES DE VALIDATION ===
# log_message("📥 Chargement des données de validation...")
# val_data = torch.load(args.val_data_path)
# val_dataset = CustomDataset(val_data["input_ids"], val_data["attention_mask"], val_data["labels"])
# val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# # === ÉVALUATION ===
# log_message("🧪 Début de l'évaluation...")
# preds, truths = [], []
# start_time = time.time()
# max_eval_duration = 60 * 60

# with torch.no_grad():
#     for batch in val_loader:
#         if time.time() - start_time > max_eval_duration:
#             log_message("⏱️ Temps d'évaluation dépassé.")
#             break

#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         predictions = torch.argmax(outputs.logits, dim=1)

#         preds.extend(predictions.cpu().tolist())
#         truths.extend(labels.cpu().tolist())

# # === RAPPORT FINAL ===
# acc = accuracy_score(truths, preds)
# log_message(f"✅ Accuracy shard {args.shard} : {acc:.4f}")
# log_message("📊 Rapport de classification :\n" + classification_report(truths, preds, digits=4))
# log_message(f"🧩 Matrice de confusion :\n{confusion_matrix(truths, preds)}")
