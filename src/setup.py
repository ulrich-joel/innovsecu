# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# setup.py

# This script implements a file monitoring system to detect ransomware-like behaviors
# (e.g., LockBit). It uses the Watchdog library to monitor filesystem events and applies
# various heuristics to detect suspicious activities such as:
# - File renaming by ransomware,
# - Bulk file modifications,
# - Abnormal file sizes.

# When suspicious activity is detected, the script:
# 1. Moves the affected files to a quarantine directory,
# 2. Sends email alerts (if configured),
# 3. Logs the events to both a file and the console.

# Configuration:
# - Update `WATCHED_FOLDERS`, `SMTP`, `QUARANTINE_DIR`, and other parameters as needed.

# Dependencies:
# - Watchdog for filesystem monitoring,
# - smtplib for email alerts,
# - hashlib for file hashing,
# - logging for event logging.

# Author: Ngueyep Ulrich
# """

# import os
# import time
# import shutil
# import hashlib
# import logging
# import smtplib
# from email.mime.text import MIMEText
# from collections import defaultdict, deque
# from queue import Queue
# from threading import Thread, Event
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# # ---------------------------
# # CONFIGURATION
# # ---------------------------

# # Directories to monitor for suspicious activity
# WATCHED_FOLDERS = [
#     r"C:\Users\nguey\OneDrive\Documents",
#     r"C:\Users\nguey\OneDrive\Desktop",
#     r"C:\Users\nguey\OneDrive\Downloads",
# ]

# # File extensions to monitor (e.g., office files, media, archives, etc.)
# MONITORED_EXTENSIONS = {
#     ".exe", ".dll", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
#     ".pdf", ".txt", ".csv", ".jpg", ".jpeg", ".png", ".bmp", ".gif",
#     ".zip", ".rar", ".7z", ".tar", ".gz", ".sql", ".bak", ".db", ".xml",
#     ".py", ".java", ".jar", ".cs", ".go", ".iso", ".vhd", ".vmdk"
# }

# # Typical ransomware-added extensions to detect
# RANSOM_EXTENSIONS = {
#     ".lockbit", ".locked", ".crypted", ".enc", ".ran", ".fckd"
# }

# # File size threshold for alerts (in MB)
# SIZE_THRESHOLD_MB = 50  # Alert if file size exceeds this value

# # Bulk modification detection parameters
# BULK_TIME_WINDOW = 60      # Time window in seconds
# BULK_COUNT_THRESHOLD = 15  # Number of file modifications in the window to trigger an alert

# # Quarantine directory for suspicious files
# QUARANTINE_DIR = os.path.join(os.path.expanduser("~"), "ransom_quarantine")
# os.makedirs(QUARANTINE_DIR, exist_ok=True)

# # Logging configuration
# LOG_FILE = "file_monitor.log"
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# logging.getLogger('').addHandler(console)

# # Email alert configuration
# ALERT_EMAIL_ENABLED = False
# ALERT_EMAIL_SMTP = "smtp.example.com"
# ALERT_EMAIL_PORT = 587
# ALERT_EMAIL_USER = "user@example.com"
# ALERT_EMAIL_PASS = "CHANGE_ME"
# ALERT_EMAIL_FROM = "alert@example.com"
# ALERT_EMAIL_TO = ["admin@example.com"]

# # Files processing queue
# file_queue = Queue()
# stop_event = Event()

# # Keep track of recent modifications per directory for bulk detection
# recent_mods = defaultdict(lambda: deque())

# # Excluded folders (e.g., system directories)
# EXCLUDED_FOLDERS = {
#     r"C:\Windows", r"C:\Program Files", r"C:\Program Files (x86)"
# }

# # ---------------------------
# # HELPER FUNCTIONS
# # ---------------------------

# def is_path_excluded(path):
#     """
#     Check if a given path is excluded from monitoring.

#     Args:
#         path (str): The file or directory path.

#     Returns:
#         bool: True if the path is excluded, False otherwise.
#     """
#     norm = os.path.normpath(path).lower()
#     for ex in EXCLUDED_FOLDERS:
#         if norm.startswith(os.path.normpath(ex).lower()):
#             return True
#     return False

# def file_hash(path, algo='sha256', max_read=4 * 1024 * 1024):
#     """
#     Compute a quick hash of the first `max_read` bytes of a file.

#     Args:
#         path (str): Path to the file.
#         algo (str): Hashing algorithm (default: sha256).
#         max_read (int): Maximum number of bytes to read.

#     Returns:
#         str: The computed hash, or None if an error occurs.
#     """
#     try:
#         h = hashlib.new(algo)
#         with open(path, 'rb') as f:
#             chunk = f.read(max_read)
#             h.update(chunk)
#         return h.hexdigest()
#     except Exception:
#         return None

# def send_email_alert(subject, body):
#     """
#     Send an email alert.

#     Args:
#         subject (str): Email subject.
#         body (str): Email body.
#     """
#     if not ALERT_EMAIL_ENABLED:
#         logging.info("🔔 Email alert (disabled) - subject: %s", subject)
#         return
#     try:
#         msg = MIMEText(body)
#         msg['Subject'] = subject
#         msg['From'] = ALERT_EMAIL_FROM
#         msg['To'] = ", ".join(ALERT_EMAIL_TO)
#         with smtplib.SMTP(ALERT_EMAIL_SMTP, ALERT_EMAIL_PORT, timeout=10) as s:
#             s.starttls()
#             s.login(ALERT_EMAIL_USER, ALERT_EMAIL_PASS)
#             s.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())
#         logging.info("📧 Email alert sent: %s", subject)
#     except Exception as e:
#         logging.error("Error sending email: %s", e)

# def quarantine_file(src_path):
#     """
#     Move a suspicious file to the quarantine directory.

#     Args:
#         src_path (str): Path to the file to quarantine.

#     Returns:
#         str: Path to the quarantined file, or None if an error occurs.
#     """
#     try:
#         if not os.path.exists(src_path):
#             logging.warning("File no longer exists: %s", src_path)
#             return None
#         basename = os.path.basename(src_path)
#         ts = int(time.time())
#         dest_name = f"{ts}_{basename}"
#         dest = os.path.join(QUARANTINE_DIR, dest_name)
#         shutil.move(src_path, dest)
#         logging.warning("🗄️ File quarantined: %s -> %s", src_path, dest)
#         return dest
#     except Exception as e:
#         logging.error("Error quarantining file (%s): %s", src_path, e)
#         return None

# # ---------------------------
# # FILESYSTEM EVENT HANDLER
# # ---------------------------

# class RansomFileHandler(FileSystemEventHandler):
#     """
#     Custom event handler for monitoring filesystem events.
#     """

#     def on_created(self, event):
#         """
#         Handle file creation events.
#         """
#         if event.is_directory:
#             return
#         path = event.src_path
#         if is_path_excluded(path):
#             return
#         _, ext = os.path.splitext(path)
#         ext = ext.lower()
#         logging.info("File created: %s (ext=%s)", path, ext)
#         if ext in MONITORED_EXTENSIONS or ext in RANSOM_EXTENSIONS:
#             file_queue.put(path)

#     def on_modified(self, event):
#         """
#         Handle file modification events.
#         """
#         if event.is_directory:
#             return
#         path = event.src_path
#         if is_path_excluded(path):
#             return
#         logging.info("File modified: %s", path)
#         file_queue.put(path)

# # ---------------------------
# # MAIN FUNCTION
# # ---------------------------

# def start_monitoring():
#     """
#     Start the file monitoring system.
#     """
#     valid_folders = [os.path.normpath(f) for f in WATCHED_FOLDERS if os.path.isdir(f)]
#     if not valid_folders:
#         logging.error("No valid folders to monitor. Check WATCHED_FOLDERS.")
#         return

#     event_handler = RansomFileHandler()
#     observer = Observer()
#     for folder in valid_folders:
#         observer.schedule(event_handler, folder, recursive=True)
#         logging.info("👁️ Monitoring started on: %s", folder)

#     observer.start()
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         logging.info("🛑 Monitoring stopped by user.")
#     finally:
#         observer.stop()
#         observer.join()

# if __name__ == "__main__":
#     start_monitoring()

import tensorflow as tf

# Vérifie les GPU disponibles
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✔️ GPU disponible")
    # Si tu veux assigner de manière explicite un GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')  # Cela limite TensorFlow à n'utiliser que le premier GPU
else:
    print("❌ Aucune GPU détectée, utilisation du CPU")

