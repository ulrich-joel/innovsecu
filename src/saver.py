# src/saver.py

# import os
# import pandas as pd

# class FileSaver:
#     def save_text(self, data, path):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "w", encoding="utf-8") as f:
#             f.write("\n".join(data))
#         print(f"[INFO] Text file saved: {path}")

#     def save_csv(self, data, path):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         data.to_csv(path, index=False)
#         print(f"[INFO] CSV file saved: {path}")
# # Sauvegarde

import os
from gtts import gTTS

# Crée un dossier 'audio' s'il n'existe pas
os.makedirs("audio", exist_ok=True)

# Texte du pitch à transformer en audio
speech_text = """
En 2019, ma famille a perdu une entreprise à cause d’un ransomware.
Les hackers exigeaient 100 millions de francs CFA, avec menace de publier toutes les données.
C’était la veille du 9ᵉ anniversaire de l’entreprise. Une semaine plus tard, elle fermait.
Et ce drame, des milliers de PME africaines le vivent chaque année, souvent sans défense.

C’est de cette douleur qu’est né INNOVSECU.
Un système de cybersécurité intelligent et prédictif, conçu ici, pour nos réalités.
INNOVSECU n’attend pas que l’attaque se déclenche. Il surveille, apprend, anticipe.
Il utilise l’intelligence artificielle, des modèles de langage comme ceux derrière ChatGPT,
et le framework MITRE ATT&CK pour comprendre les comportements malveillants, comme ceux des ransomwares.

Imaginons qu’une PME reçoive un fichier Word piégé par LockBit – le ransomware le plus redoutable de 2024.
Dès l’ouverture, INNOVSECU repère l’anomalie.
Il bloque la menace en quelques secondes et affiche cette alerte explicative à l’utilisateur.
Pas besoin d’être expert. L’outil vous dit ce qui se passe, et ce que vous devez faire.
C’est simple, visuel et surtout : proactif.

Nous voulons offrir cette solution aux PME à partir de 10 000 FCFA par mois.
Moins cher qu’un antivirus, mais capable de détecter des attaques ciblées.
Notre ambition : protéger 100 entreprises la première année,
et former leurs équipes grâce à un assistant IA intégré.
Notre devise : anticiper, expliquer, protéger.

INNOVSECU, c’est l’IA au service d’une cybersécurité africaine, accessible à tous.
Parce que dans notre continent aussi, nos données valent de l’or.
"""

# Chemin local pour enregistrer l'audio
audio_path = "audio/innovsecu_demo_pitch.mp3"
tts = gTTS(speech_text, lang='fr')
tts.save(audio_path)

print(f"Audio enregistré avec succès à : {audio_path}")
