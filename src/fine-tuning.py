import os
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime
import json, time

# === CONFIGURATION ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#shard_dir = "F:/bert_shards"
shard_dir= "output/models/train_llm_bert/bert_full_model/shards"
val_data_path = "output/models/train_llm_bert/validation_data.pt"
model_dir = "logs"
checkpoints_dir = os.path.join(model_dir, "checkpoints")
log_path = os.path.join(model_dir, "training_log.txt")
resume_path = os.path.join(model_dir, "resume_state.json")

os.makedirs(checkpoints_dir, exist_ok=True)

batch_size = 4
epochs = 1
gradient_accumulation_steps = 4

# === SHARDS Déjà TRAITÉS ===
used_shards = {
    "tokenized_data_1.pt", "tokenized_data_2.pt", "tokenized_data_3.pt",
    "tokenized_data_18.pt", "tokenized_data_19.pt", "tokenized_data_20.pt",
    "tokenized_data_21.pt", "tokenized_data_22.pt", "tokenized_data_23.pt",
    "tokenized_data_24.pt", "tokenized_data_25.pt"
}

# === CLASSE DATASET ===
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

# === MÉTHODE POUR NETTOYER LA MÉMOIRE ===
def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# === MÉTHODE POUR LOG ===
def log_message(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")

# === INITIALISATION TOKENIZER ===
if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
else:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

log_message("🧪 Chargement des données de validation...")
val_data = torch.load(val_data_path)
val_dataset = CustomDataset(val_data["input_ids"], val_data["attention_mask"], val_data["labels"])
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === CHARGEMENT OU CRÉATION DU MODÈLE ===
try:
    if os.path.isfile(os.path.join(model_dir, "pytorch_model.bin")) or \
       os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        log_message("📂 Rechargement du modèle existant depuis logs/ ...")
        model = BertForSequenceClassification.from_pretrained(model_dir)
    else:
        raise FileNotFoundError("Aucun fichier de poids trouvé dans logs/")
except Exception as e:
    log_message(f"⚠️ Erreur lors du chargement local : {e}")
    log_message("📥 Téléchargement du modèle BERT de base...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# === CHARGEMENT / INITIALISATION DE L'ÉTAT DE REPRISE ===
def load_resume_state():
    if os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            return json.load(f)
    else:
        return {"last_completed_shard": -1}

def save_resume_state(index):
    with open(resume_path, "w") as f:
        json.dump({"last_completed_shard": index}, f)

resume_state = load_resume_state()
start_index = resume_state["last_completed_shard"] + 1

# === LISTE DES SHARDS FILTRÉS ===
all_shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".pt")])
shard_files = [f for f in all_shard_files if f not in used_shards]
log_message(f"🧲 {len(shard_files)} shard(s) à entraîner restants : {shard_files}")

# === ENTRAÎNEMENT PAR SHARD ===
for i, shard_file in enumerate(shard_files, start=start_index):
    shard_name = f"epoch_shard_{i+1}"
    shard_checkpoint_path = os.path.join(checkpoints_dir, shard_name)

    log_message(f"\n📦 Début de l'entraînement sur shard {i+1}/{len(shard_files) + start_index} : {shard_file}")
    shard_path = os.path.join(shard_dir, shard_file)
    shard_data = torch.load(shard_path)

    train_dataset = CustomDataset(shard_data["input_ids"], shard_data["attention_mask"], shard_data["labels"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.train()
    total_loss = 0
    optimizer.zero_grad()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Shard {i+1}")

    for step, batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            clear_cuda()

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix(loss=loss.item() * gradient_accumulation_steps)

    avg_loss = total_loss / len(train_loader)
    log_message(f"📉 Perte moyenne sur shard {i+1}: {avg_loss:.4f}")

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    os.makedirs(shard_checkpoint_path, exist_ok=True)
    torch.save(optimizer.state_dict(), os.path.join(shard_checkpoint_path, "optimizer.pt"))
    save_resume_state(i)
    log_message(f"🗕️ Modèle et état sauvegardés après shard {i+1}")

# === ÉVALUATION APRÈS ENTRAÎnEMENT ===
do_evaluation = False
if do_evaluation:
    try:
        log_message("🧪 Début de l'évaluation sur les données de validation...")
        clear_cuda()
        model.eval()
        preds, truths = [], []

        start_time = time.time()
        max_eval_duration = 60 * 60

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if time.time() - start_time > max_eval_duration:
                    raise TimeoutError(f"⏱️ Évaluation interrompue automatiquement après {max_eval_duration // 60} min.")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)

                preds.extend(predictions.cpu().tolist())
                truths.extend(labels.cpu().tolist())

        acc = accuracy_score(truths, preds)
        log_message(f"✅ Accuracy après shard {i+1}: {acc:.4f}")

    except TimeoutError as te:
        log_message(str(te))
        log_message(f"⚠️ Passage automatique au shard suivant.")
    except KeyboardInterrupt:
        log_message("⚠️ Évaluation interrompue manuellement après entraînement.")
        log_message("⚠️ Reprise possible au prochain shard sans problème.")
        exit(0)
    except Exception as e:
        log_message(f"❌ Erreur lors de l’évaluation : {str(e)}")
        log_message("⚠️ Passage automatique au shard suivant malgré l'erreur.")

# import os
# import torch
# import gc
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertForSequenceClassification, BertTokenizer, get_scheduler
# from torch.optim import AdamW
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
# from datetime import datetime
# import json, time

# # === CONFIGURATION ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# shard_dir = "F:/bert_shards"
# #shard_dir = "C:/Users/nguey/OneDrive/Documents/UY1/Articles/innovsecu/output/models/train_llm_bert/bert_full_model/shards"
# val_data_path = "output/models/train_llm_bert/validation_data.pt"
# model_dir = "logs"
# checkpoints_dir = os.path.join(model_dir, "checkpoints")
# log_path = os.path.join(model_dir, "training_log.txt")
# resume_path = os.path.join(model_dir, "resume_state.json")

# os.makedirs(checkpoints_dir, exist_ok=True)

# batch_size = 4
# epochs = 1
# gradient_accumulation_steps = 4

# # === CLASSE DATASET ===
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

# # === MÉTHODE POUR NETTOYER LA MÉMOIRE ===
# def clear_cuda():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     gc.collect()

# # === MÉTHODE POUR LOG ===
# def log_message(message):
#     timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
#     full_message = f"{timestamp} {message}"
#     print(full_message)
#     with open(log_path, "a", encoding="utf-8") as f:
#         f.write(full_message + "\n")

# # === INITIALISATION TOKENIZER & VALIDATION ===
# # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# # === INITIALISATION TOKENIZER & VALIDATION ===
# if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
#     tokenizer = BertTokenizer.from_pretrained(model_dir)
# else:
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# log_message("🧪 Chargement des données de validation...")
# val_data = torch.load(val_data_path)
# val_dataset = CustomDataset(val_data["input_ids"], val_data["attention_mask"], val_data["labels"])
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# # === CHARGEMENT OU CRÉATION DU MODÈLE ===
# # if os.path.isfile(os.path.join(model_dir, "pytorch_model.bin")):
# #     log_message("📂 Rechargement du modèle existant...")
# #     model = BertForSequenceClassification.from_pretrained(model_dir)
# # else:
# #     log_message("📥 Chargement d'un nouveau modèle BERT...")
# #     model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# # === CHARGEMENT OU CRÉATION DU MODÈLE ===
# try:
#     # Vérifie présence de *model.safetensors* OU *pytorch_model.bin*
#     if os.path.isfile(os.path.join(model_dir, "pytorch_model.bin")) or \
#        os.path.isfile(os.path.join(model_dir, "model.safetensors")):
#         log_message("📂 Rechargement du modèle existant depuis logs/ ...")
#         model = BertForSequenceClassification.from_pretrained(model_dir)
#     else:
#         raise FileNotFoundError("Aucun fichier de poids trouvé dans logs/")
# except Exception as e:
#     log_message(f"⚠️ Erreur lors du chargement local : {e}")
#     log_message("📥 Téléchargement du modèle BERT de base...")
#     model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# model.to(device)
# optimizer = AdamW(model.parameters(), lr=2e-5)

# # === CHARGEMENT / INITIALISATION DE L'ÉTAT DE REPRISE ===
# def load_resume_state():
#     if os.path.exists(resume_path):
#         with open(resume_path, "r") as f:
#             return json.load(f)
#     else:
#         return {"last_completed_shard": -1}

# def save_resume_state(index):
#     with open(resume_path, "w") as f:
#         json.dump({"last_completed_shard": index}, f)

# resume_state = load_resume_state()
# start_index = resume_state["last_completed_shard"] + 1

# # === LISTE DES SHARDS À ENTRAÎNER ===
# shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".pt")])

# # === ENTRAÎNEMENT PAR SHARD ===
# for i in range(start_index, len(shard_files)):
#     shard_file = shard_files[i]
#     shard_name = f"epoch_shard_{i+1}"
#     shard_checkpoint_path = os.path.join(checkpoints_dir, shard_name)

#     log_message(f"\n📦 Début de l'entraînement sur shard {i+1}/{len(shard_files)} : {shard_file}")
#     shard_path = os.path.join(shard_dir, shard_file)
#     shard_data = torch.load(shard_path)

#     train_dataset = CustomDataset(shard_data["input_ids"], shard_data["attention_mask"], shard_data["labels"])
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     num_training_steps = len(train_loader) * epochs
#     scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

#     model.train()
#     total_loss = 0
#     optimizer.zero_grad()
#     progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Shard {i+1}")

#     for step, batch in progress_bar:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss / gradient_accumulation_steps
#         loss.backward()

#         if (step + 1) % gradient_accumulation_steps == 0:
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#             clear_cuda()

#         total_loss += loss.item() * gradient_accumulation_steps
#         progress_bar.set_postfix(loss=loss.item() * gradient_accumulation_steps)

#     avg_loss = total_loss / len(train_loader)
#     log_message(f"📉 Perte moyenne sur shard {i+1}: {avg_loss:.4f}")

#     # === SAUVEGARDE DU MODÈLE ET DE L'ÉTAT ===
#     model.save_pretrained(model_dir)
#     tokenizer.save_pretrained(model_dir)
#     os.makedirs(shard_checkpoint_path, exist_ok=True)
#     torch.save(optimizer.state_dict(), os.path.join(shard_checkpoint_path, "optimizer.pt"))
#     save_resume_state(i)
#     log_message(f"💾 Modèle et état sauvegardés après shard {i+1}")

# # === ÉVALUATION AVEC GESTION D'ERREURS ET BLOQUAGE ===
# do_evaluation = False  # Mets False si tu veux désactiver les évaluations
# if do_evaluation:
#     try:
#         log_message("🧪 Début de l'évaluation sur les données de validation...")
#         clear_cuda()
#         model.eval()
#         preds, truths = [], []

#         start_time = time.time()
#         max_eval_duration = 60 * 60  # 1h max pour évaluation

#         with torch.no_grad():
#             for batch_idx, batch in enumerate(val_loader):
#                 if time.time() - start_time > max_eval_duration:
#                     raise TimeoutError(f"⏱️ Évaluation interrompue automatiquement après {max_eval_duration // 60} min.")

#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels = batch["labels"].to(device)

#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#                 predictions = torch.argmax(outputs.logits, dim=1)

#                 preds.extend(predictions.cpu().tolist())
#                 truths.extend(labels.cpu().tolist())

#         acc = accuracy_score(truths, preds)
#         log_message(f"✅ Accuracy après shard {i+1}: {acc:.4f}")

#     except TimeoutError as te:
#         log_message(str(te))
#         log_message(f"⚠️ Passage automatique au shard suivant.")
#     except KeyboardInterrupt:
#         log_message("⚠️ Évaluation interrompue manuellement après entraînement.")
#         log_message("⚠️ Reprise possible au prochain shard sans problème.")
#         exit(0)
#     except Exception as e:
#         log_message(f"❌ Erreur lors de l’évaluation : {str(e)}")
#         log_message("⚠️ Passage automatique au shard suivant malgré l'erreur.")



