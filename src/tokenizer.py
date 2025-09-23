#src/tokenizer.py
import torch
import os
import json
from transformers import BertTokenizer

# Preparation
print("🔠 Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("./bert_model")

json_path = os.path.join("output", "dynamic_mitre_mapped_with_labels.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract texts and labels from the JSON structure
texts = [entry["text"] for entry in data["data"]]
labels = [1 if entry["label"] == "malveillant" else 0 for entry in data["data"]]

print("✂️ Tokenizing texts...")
tokenized = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

print("💾 Saving tokenized data...")
torch.save({
    "input_ids": tokenized["input_ids"],
    "attention_mask": tokenized["attention_mask"],
    "labels": torch.tensor(labels)
}, os.path.join("output", "tokenized_data.pt"))

print("✅ Tokenization completed and data saved.")


# #src/shards_tokenizer.py
# import torch
# import os

# # === CONFIGURATION ===
# input_path = "output/models/train_llm_bert/tokenized_data.pt"
# output_dir = "F:/bert_shards"
# os.makedirs(output_dir, exist_ok=True)

# # === CHARGEMENT DES DONNÉES ===
# print("📦 Chargement des données d'entraînement...")
# data = torch.load(input_path)
# total_size = len(data["labels"])
# num_shards = 30
# shard_size = total_size // num_shards

# # === CRÉATION DES MORCEAUX ===
# for i in range(num_shards):
#     start = i * shard_size
#     end = total_size if i == num_shards - 1 else (i + 1) * shard_size
#     shard = {
#         "input_ids": data["input_ids"][start:end],
#         "attention_mask": data["attention_mask"][start:end],
#         "labels": data["labels"][start:end]
#     }
#     shard_path = os.path.join(output_dir, f"tokenized_data_{i}.pt")
#     torch.save(shard, shard_path)
#     print(f"✅ Shard {i+1}/{num_shards} sauvegardé : {shard_path}")

# import os
# import torch
# from math import ceil

# # === CONFIGURATION ===
# full_data_path = "output/models/train_llm_bert/tokenized_data.pt"
# output_dir = "output/models/train_llm_bert/bert_full_model/shards"
# start_shard = 25  # Pour continuer depuis ici
# total_shards = 30  # Nombre total de shards à créer

# # === CRÉATION DU DOSSIER DE SORTIE ===
# os.makedirs(output_dir, exist_ok=True)

# # === CHARGEMENT DES DONNÉES ===
# print("📦 Chargement des données tokenisées complètes...")
# data = torch.load(full_data_path)
# input_ids = data["input_ids"]
# attention_mask = data["attention_mask"]
# labels = data["labels"]

# total_size = len(labels)
# shard_size = ceil(total_size / total_shards)

# print(f"🔁 Reprise à partir du shard {start_shard + 1}/{total_shards}")
# print(f"📊 Total de données : {total_size} — Taille par shard : {shard_size}")

# # === DÉCOUPE DES SHARDS MANQUANTS ===
# for i in range(start_shard, total_shards):
#     start_idx = i * shard_size
#     end_idx = min((i + 1) * shard_size, total_size)

#     shard = {
#         "input_ids": input_ids[start_idx:end_idx],
#         "attention_mask": attention_mask[start_idx:end_idx],
#         "labels": labels[start_idx:end_idx],
#     }

#     shard_path = os.path.join(output_dir, f"tokenized_data_{i}.pt")
#     torch.save(shard, shard_path)
#     print(f"✅ Shard {i+1}/{total_shards} sauvegardé : {shard_path}")

# print("🎉 Découpage repris et terminé avec succès !")


