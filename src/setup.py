# # src/setup.py
# import win32evtlog
# import time
# import os
# import logging
# import numpy as np
# import joblib
# from config.config import Config

# # === CONFIGURATION ===
# config = Config()
# model_path = config.ISOLATION_FOREST_MODEL_PATH
# scaler_path = os.path.join(config.PROCESSED_DYNAMIC_DIR, "scaler.pkl")

# # === LOGGER ===
# LOG_FILE = os.path.join(config.LOG_DIR, "anomaly_detection.log")
# os.makedirs(config.LOG_DIR, exist_ok=True)
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# logger = logging.getLogger()

# def log_alert(message):
#     print(message)
#     logger.warning(message)

# # === CHARGEMENT MODELE ET SCALER ===
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"❌ Modèle Isolation Forest introuvable à : {model_path}")
# model = joblib.load(model_path)
# logger.info("Modèle Isolation Forest chargé avec succès.")

# if not os.path.exists(scaler_path):
#     raise FileNotFoundError(f"❌ Scaler introuvable à : {scaler_path}")
# scaler = joblib.load(scaler_path)
# logger.info("Scaler chargé avec succès.")

# # === PARAMÈTRES DE SURVEILLANCE ===
# SERVER = 'localhost'
# LOG_TYPE = 'Microsoft-Windows-Sysmon/Operational'

# # Liste des features attendues (doit correspondre à l'ordre dans ton scaler/model)
# FEATURE_ORDER = [
#     "timestamp", "processId", "threadId", "parentProcessId", "userId",
#     "mountNamespace", "eventId", "argsNum", "returnValue", "stack_depth", "behavior_cluster"
# ]

# # === EXTRACTION DES FEATURES DEPUIS event.StringInserts ===
# def extract_sysmon_data(event):
#     try:
#         inserts = event.StringInserts or []
#         logger.debug(f"Event {event.RecordNumber} raw inserts: {inserts}")
#         features = []
#         for i in range(len(FEATURE_ORDER)):
#             if i < len(inserts):
#                 try:
#                     val = float(inserts[i])
#                 except (ValueError, TypeError):
#                     val = 0.0
#             else:
#                 val = 0.0
#             features.append(val)

#         logger.debug(f"Event {event.RecordNumber} features (before scaling): {features}")

#         vector = np.array(features).reshape(1, -1)
#         vector_scaled = scaler.transform(vector)
#         return vector_scaled
#     except Exception as e:
#         logger.error(f"Error extracting features: {e}")
#         return None


# # === DÉTECTION ===
# def detect(vector):
#     try:
#         score = model.decision_function(vector)
#         is_anomaly = model.predict(vector)[0] == -1
#         return is_anomaly, score[0]
#     except Exception as e:
#         logger.error(f"Erreur pendant la détection : {e}")
#         return False, 0

# # === ACTION SUR ANOMALIE ===
# def react_to_anomaly(event, score):
#     alert_msg = f"Anomalie détectée (score={score:.4f}) dans événement ID {event.RecordNumber}"
#     log_alert(alert_msg)
#     # TODO: Ajouter ici des actions (alerte mail, blocage processus, etc.)

# # === BOUCLE PRINCIPALE DE SURVEILLANCE ===
# def monitor_sysmon():
#     hand = win32evtlog.OpenEventLog(SERVER, LOG_TYPE)
#     flags = win32evtlog.EVENTLOG_FORWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
#     seen = set()
#     PURGE_INTERVAL = 3600  # purge toutes les heures
#     last_purge = time.time()

#     logger.info(f"Starting monitoring on {LOG_TYPE}")
#     print(f"🚨 Surveillance active sur {LOG_TYPE} (Ctrl+C pour arrêter)...")

#     while True:
#         events = win32evtlog.ReadEventLog(hand, flags, 0)
#         if events is None:
#             time.sleep(5)
#             continue

#         for event in events:
#             record_id = event.RecordNumber
#             if record_id in seen:
#                 continue
#             seen.add(record_id)

#             vector = extract_sysmon_data(event)
#             if vector is not None:
#                 anomaly, score = detect(vector)
#                 if anomaly:
#                     react_to_anomaly(event, score)

#         # Purge périodique des events vus pour libérer la mémoire
#         if time.time() - last_purge > PURGE_INTERVAL:
#             logger.info("Purge de la mémoire des événements vus.")
#             seen.clear()
#             last_purge = time.time()

#         time.sleep(5)

# # === POINT D'ENTRÉE ===
# if __name__ == "__main__":
#     try:
#         monitor_sysmon()
#     except KeyboardInterrupt:
#         print("\n🛑 Surveillance arrêtée.")
#         logger.info("Monitoring stopped by user.")

import os
import time
import logging
from queue import Queue
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
WATCHED_FOLDERS = [
    r"C:\Users\nguey\OneDrive\Documents",
    r"C:\Users\nguey\OneDrive\Desktop",
    r"C:\Users\nguey\Downloads",  # ou "Téléchargements" selon ton système
]


# Extensions à surveiller
WHITELIST_EXTENSIONS = {
    ".exe", ".dll", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".pdf", ".txt", ".csv",
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".zip", ".rar", ".7z",
}

# Logger setup
logging.basicConfig(
    filename='file_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Queue de fichiers à traiter
file_queue = Queue()

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        print(f"DEBUG: on_created event detected for {filepath} with ext {ext}")  # debug

        if ext in WHITELIST_EXTENSIONS:
            logging.info(f"📁 Nouveau fichier détecté : {filepath}")
            file_queue.put(filepath)
        else:
            logging.info(f"→ Fichier ignoré (extension non supportée) : {filepath}")

    def on_deleted(self, event):
        if event.is_directory:
            return
        logging.info(f"🗑️ Fichier supprimé : {event.src_path}")


    def on_modified(self, event):
        print(f"DEBUG: on_modified event detected for {event.src_path}")  # debug


def process_files():
    while True:
        filepath = file_queue.get()
        try:
            # Ici tu peux appeler ta fonction d'extraction et prétraitement
            logging.info(f"🔎 Traitement du fichier : {filepath}")
            # simulate processing
            time.sleep(1)
            logging.info(f"✅ Traitement terminé pour : {filepath}")
        except Exception as e:
            logging.error(f"❌ Erreur lors du traitement de {filepath} : {e}")
        finally:
            file_queue.task_done()

def main():
    # Vérifier que les dossiers existent
    valid_folders = [os.path.normpath(f) for f in WATCHED_FOLDERS if os.path.isdir(f)]

    if not valid_folders:
        logging.error("🛑 Aucun dossier valide à surveiller.")
        return

    event_handler = FileHandler()
    observer = Observer()
    for folder in valid_folders:
        observer.schedule(event_handler, folder, recursive=True)
        logging.info(f"👁️ Surveillance démarrée sur : {folder}")

    # Thread de traitement des fichiers
    worker_thread = Thread(target=process_files, daemon=True)
    worker_thread.start()

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("🛑 Surveillance arrêtée par l'utilisateur.")
        observer.stop()
    observer.join()
    file_queue.join()

if __name__ == "__main__":
    main()