import csv
import time
import os
from datetime import datetime

class ValidationLogger:
    def __init__(self, filename="session_data.csv"):
        self.filename = filename
        # Si le fichier n'existe pas, on crée l'en-tête AVEC LES DÉTAILS
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Time_Relatif", "Etat_Global", "Score_Total", 
                    "Score_Geo", "Score_CNN", "Temperature_Habitacle", "FPS",
                    "Detail_Yeux", "Detail_Sourcils", "Detail_Bouche" # <--- NOUVEAU
                ])
        
        self.start_time = time.time()

    def log_frame(self, state, total_score, geo_score, cnn_score, temp, fps, eyes, brows, mouth):
        """Enregistre une ligne de données complètes."""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, 
                round(elapsed, 2), 
                state, 
                round(total_score, 2),
                round(geo_score, 2),
                round(cnn_score, 2),
                round(temp, 1),
                int(fps),
                eyes,   # <--- NOUVEAU
                brows,  # <--- NOUVEAU
                mouth   # <--- NOUVEAU
            ])