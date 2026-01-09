import cv2
import numpy as np
import time
import sys
import os
from collections import Counter

# --- IMPORTS MODULES LOCAUX ---
try:
    from detection import FaceRegionDetector
    from emotion_cnn import EmotionCNNAnalyzer
    from geometry_analysis import FaceGeometryAnalyzer
except ImportError as e:
    print(f"[ERREUR CRITIQUE] Manque un fichier : {e}")
    sys.exit(1)

# Gestion Son
try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

# Gestion Vêtements (Optionnel)
try:
    from clothing_analysis import ClothingAnalyzer
    HAS_CLOTHING = True
except ImportError:
    HAS_CLOTHING = False
    class ClothingAnalyzer:
        def analyze_attire(self, img, bbox): return []

# --- CHARTE GRAPHIQUE STELLANTIS (PRO & CLEAN) ---
C_WHITE = (255, 255, 255)
C_STELLANTIS_BLUE = (105, 51, 0) 
C_TEXT_DARK = (40, 40, 40)
C_PANEL_BG = (30, 30, 35)       
C_FOOTER_BG = (240, 240, 240)    

# Couleurs Sémantiques
C_OK = (80, 200, 80)        # Vert (Confort)
C_WARN = (0, 165, 255)      # Orange (Attention)
C_ALERT = (50, 50, 220)     # Rouge (Inconfort / Somnolence)
C_NEUTRAL = (160, 160, 160) # Gris (Neutre)

FONT_MAIN = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

class StellantisUltimateSystem:
    def __init__(self):
        print("\n=== STELLANTIS CARE MONITOR (VERSION FINALE 20/20) ===")
        
        # 1. Chargement des IA
        self.detector = FaceRegionDetector() 
        self.geo_engine = FaceGeometryAnalyzer()
        self.cnn_engine = EmotionCNNAnalyzer("models/emotion_resnet18_affectnet.pt")
        self.cloth_engine = ClothingAnalyzer()
        
        # Logo
        self.logo_img = None
        if os.path.exists("logo.png"):
            img = cv2.imread("logo.png")
            if img is not None:
                h_l, w_l = img.shape[:2]
                ratio = 50 / h_l
                self.logo_img = cv2.resize(img, (int(w_l * ratio), 50))

        # 2. Variables Système
        self.state = "INIT"
        self.calib_buffer = []
        self.calib_start_time = 0
        self.frame_count = 0
        
        # 3. Climatisation
        self.target_temp = 21.0
        self.current_temp = 21.0
        self.climate_mode = "AUTO"
        
        # 4. Somnolence
        self.eyes_closed_start = None
        self.is_drowsy = False

        # 5. VLM MEMORY (Historique 30s)
        self.state_history = [] 
        self.stats_percentages = {"CONFORT": 0, "NEUTRE": 0, "INCONFORT": 0}
        
        self.hud_data = {
            "global_state": "EN ATTENTE",
            "geo_details": {},
            "cnn_details": {"label": "-", "score": 0.0},
            "clothes": [],
            "bbox": None
        }

    # --- MOTEUR INTELLIGENT (VLM & FUSION) ---

    def update_history_30s(self, current_state):
        """
        Calcule la moyenne glissante exacte sur les 30 dernières secondes.
        """
        now = time.time()
        # Ajouter l'état courant
        self.state_history.append((now, current_state))
        
        # Filtrer : Garder uniquement les données où (Temps actuel - Temps donnée) <= 30s
        self.state_history = [x for x in self.state_history if (now - x[0]) <= 30.0]
        
        total = len(self.state_history)
        if total > 0:
            counts = Counter([x[1] for x in self.state_history])
            
            # Calcul précis
            p_conf = int((counts["CONFORT"] / total) * 100)
            p_inconf = int((counts["INCONFORT"] / total) * 100)
            p_neutre = int((counts["NEUTRE"] / total) * 100)
            
            # Correction pour que la somme fasse toujours 100% (Gestion des arrondis)
            reste = 100 - (p_conf + p_inconf + p_neutre)
            p_neutre += reste
            
            self.stats_percentages["CONFORT"] = p_conf
            self.stats_percentages["INCONFORT"] = p_inconf
            self.stats_percentages["NEUTRE"] = p_neutre

    def update_climate(self):
        st = self.hud_data["global_state"]
        if st == "INCONFORT":
            self.target_temp = 18.0
            self.climate_mode = "AC MAX"
        elif st == "CONFORT":
            self.target_temp = 21.0
            self.climate_mode = "ECO"
        else:
            self.target_temp = 22.5
            self.climate_mode = "STANDARD"
            
        diff = self.target_temp - self.current_temp
        self.current_temp += diff * 0.02

    def fusion_intelligence(self, geo, cnn_label, cnn_score, clothes_list):
        """
        Algorithme de fusion multi-modale.
        """
        s_geo = 0.0
        s_cnn = 0.0
        s_cloth = 0.0
        
        txt_mouth = geo.get("txt_mouth", "")
        txt_eyes = geo.get("txt_eyes", "")
        txt_brows = geo.get("txt_brows", "")
        
        # --- 1. DETECTION SOMNOLENCE (SECURITE) ---
        is_eyes_closed = "Plisses" in txt_eyes and "Sourire" not in txt_mouth
        if is_eyes_closed:
            if self.eyes_closed_start is None: self.eyes_closed_start = time.time()
            elif (time.time() - self.eyes_closed_start) > 1.5: self.is_drowsy = True
        else:
            self.eyes_closed_start = None
            self.is_drowsy = False

        # --- 2. CALCUL DES SCORES (CONFORT/INCONFORT) ---
        
        # A. Géométrie (Poids forts)
        if "Sourire" in txt_mouth: 
            s_geo += 5.0
        
        # Les sourcils froncés pèsent LOURD (Déclenche Inconfort)
        if "Fronces" in txt_brows: 
            s_geo -= 6.0  
            
        if "Grimace" in txt_mouth or "Tension" in txt_mouth: 
            s_geo -= 4.0 
        elif "Baillement" in txt_mouth: 
            s_geo -= 5.0
        
        # B. Emotion (CNN)
        if cnn_score > 0.6:
            if cnn_label == "happy": s_cnn += 3.0
            elif cnn_label in ["sad", "angry", "fear"]: s_cnn -= 3.0

        # C. Contexte (Vêtements)
        if HAS_CLOTHING:
            if "DEBARDEUR" in clothes_list: s_cloth += 2.0
            elif "MANTEAU" in clothes_list: s_cloth -= 2.0
            if "BONNET" in clothes_list: s_cloth -= 3.0
            if "LUNETTES" in clothes_list: s_cloth += 0.5 

        total_score = s_geo + s_cnn + s_cloth

        # --- 3. DECISION FINALE ---
        if self.is_drowsy: 
            final_state = "SOMNOLENCE"
        elif total_score >= 4.5: 
            final_state = "CONFORT"
        elif total_score <= -4.0: 
            final_state = "INCONFORT"
        else: 
            final_state = "NEUTRE"
        
        return final_state, total_score

    # --- DESIGN UI (PIXEL PERFECT) ---

    def draw_gauge(self, img, x, y, w, h, percent, color, label):
        """Dessine une jauge horizontale propre."""
        # Label
        cv2.putText(img, label, (x, y - 8), FONT_MAIN, 0.4, C_TEXT_DARK, 1, cv2.LINE_AA)
        # Fond
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), -1)
        # Remplissage
        fill_w = int(w * (percent / 100))
        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
        # Pourcentage texte
        cv2.putText(img, f"{percent}%", (x + w + 5, y + h - 1), FONT_MAIN, 0.35, (80,80,80), 1, cv2.LINE_AA)

    def draw_ui_pro(self, img):
        h, w = img.shape[:2]
        
        # ALERTE SOMNOLENCE
        if self.is_drowsy:
            if int(time.time() * 10) % 2 == 0: # Clignotement
                overlay = img.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            txt = "!!! ALERTE SOMNOLENCE !!!"
            sz = cv2.getTextSize(txt, FONT_BOLD, 1.5, 3)[0]
            cx = (w - sz[0]) // 2
            cv2.putText(img, txt, (cx, h//2), FONT_BOLD, 1.5, (0,0,255), 5)
            cv2.putText(img, txt, (cx, h//2), FONT_BOLD, 1.5, (255,255,255), 2)
            if HAS_SOUND: winsound.Beep(2000, 100)
            return

        # 1. HEADER
        header_h = 80
        cv2.rectangle(img, (0, 0), (w, header_h), C_WHITE, -1)
        cv2.line(img, (0, header_h), (w, header_h), (200, 200, 200), 1)

        # Logo
        if self.logo_img is not None:
            h_l, w_l = self.logo_img.shape[:2]
            y_off = (header_h - h_l) // 2
            img[y_off:y_off+h_l, 20:20+w_l] = self.logo_img
        else:
            cv2.putText(img, "STELLANTIS", (30, 52), FONT_BOLD, 1.0, C_STELLANTIS_BLUE, 2, cv2.LINE_AA)
        
        cv2.putText(img, "CARE MONITOR", (250, 52), FONT_MAIN, 0.8, C_TEXT_DARK, 1, cv2.LINE_AA)

        # Bouton Action
        btn_txt = "CALIBRATION..." if self.state == "CALIB" else "RE-CALIBRER (Espace)"
        col_btn = C_WARN if self.state == "CALIB" else C_STELLANTIS_BLUE
        cv2.rectangle(img, (w-260, 20), (w-20, 60), C_WHITE, -1)
        cv2.rectangle(img, (w-260, 20), (w-20, 60), col_btn, 2)
        cv2.putText(img, btn_txt, (w-245, 48), FONT_MAIN, 0.55, col_btn, 1, cv2.LINE_AA)

        # 2. DETAILS (DROITE)
        if self.state == "RUN":
            panel_w = 300
            # Panneau gris foncé à droite
            cv2.rectangle(img, (w - panel_w, header_h), (w, h - 100), C_PANEL_BG, -1)
            
            x_p = w - panel_w + 20
            curr_y = header_h + 40
            
            cv2.putText(img, "ANALYSE TEMPS REEL", (x_p, curr_y), FONT_MAIN, 0.5, (180,180,180), 1)
            curr_y += 30

            # Vêtements
            clothes = self.hud_data["clothes"]
            if clothes:
                for c in clothes[:2]:
                    cv2.rectangle(img, (x_p, curr_y), (x_p+150, curr_y+25), (60,60,60), -1)
                    cv2.putText(img, c, (x_p+10, curr_y+18), FONT_MAIN, 0.45, C_WHITE, 1)
                    curr_y += 35
            
            curr_y += 10
            
            # Emotion
            cnn = self.hud_data["cnn_details"]
            # MODIF: Remplacement de "IMPASSIBLE" par "NEUTRE"
            lbl_display = cnn["label"].upper().replace("HAPPY", "SEREIN").replace("NEUTRAL", "NEUTRE").replace("SAD", "PREOCCUPE")
            
            cv2.putText(img, f"EMOTION: {lbl_display}", (x_p, curr_y), FONT_MAIN, 0.6, C_WHITE, 1)
            curr_y += 10
            
            # Barre confiance
            cv2.rectangle(img, (x_p, curr_y), (x_p+200, curr_y+6), (80,80,80), -1)
            cv2.rectangle(img, (x_p, curr_y), (x_p+int(200*cnn["score"]), curr_y+6), C_WARN, -1)
            curr_y += 40

            # Debug Géométrie
            geo = self.hud_data["geo_details"]
            if geo:
                cv2.putText(img, "BIOMETRIE:", (x_p, curr_y), FONT_MAIN, 0.5, (180,180,180), 1)
                curr_y += 25
                cv2.putText(img, f"Yeux: {geo.get('txt_eyes', '-')}", (x_p, curr_y), FONT_MAIN, 0.5, C_WHITE, 1)
                curr_y += 25
                
                # MODIF: Pas de couleur rouge pour les sourcils froncés (reste Blanc)
                txt_b = geo.get('txt_brows', '-')
                col_b = C_WHITE 
                
                cv2.putText(img, f"Sourcils: {txt_b}", (x_p, curr_y), FONT_MAIN, 0.5, col_b, 1)
                curr_y += 25
                cv2.putText(img, f"Bouche: {geo.get('txt_mouth', '-')}", (x_p, curr_y), FONT_MAIN, 0.5, C_WHITE, 1)

        # 3. FOOTER (STATISTIQUES 30 SECONDES)
        footer_h = 100
        y_f = h - footer_h
        cv2.rectangle(img, (0, y_f), (w, h), C_FOOTER_BG, -1)
        cv2.line(img, (0, y_f), (w, y_f), (200, 200, 200), 1)

        # A. Jauges Moyenne 30s
        stat_x = 30
        stat_y_title = y_f + 30
        stat_y_gauge = y_f + 65
        
        cv2.putText(img, "MOYENNE (30s)", (stat_x, stat_y_title), FONT_BOLD, 0.6, C_TEXT_DARK, 1)

        # --- MODIFICATION: Affichage de l'état dominant ---
        # On cherche la clé avec la plus grande valeur
        dominant_state = max(self.stats_percentages, key=self.stats_percentages.get)
        dom_val = self.stats_percentages[dominant_state]
        
        # Couleur du texte dominant
        c_dom = C_NEUTRAL
        if dominant_state == "CONFORT": c_dom = C_OK
        elif dominant_state == "INCONFORT": c_dom = C_ALERT
        
        # Affichage à côté du titre "MOYENNE (30s)" (Décalé de +170 pixels)
        cv2.putText(img, f"-> {dominant_state} ({dom_val}%)", (stat_x + 170, stat_y_title), FONT_BOLD, 0.6, c_dom, 1)
        # --------------------------------------------------
        
        self.draw_gauge(img, stat_x, stat_y_gauge, 90, 8, self.stats_percentages["CONFORT"], C_OK, "Confort")
        self.draw_gauge(img, stat_x + 150, stat_y_gauge, 90, 8, self.stats_percentages["NEUTRE"], C_NEUTRAL, "Neutre")
        self.draw_gauge(img, stat_x + 300, stat_y_gauge, 90, 8, self.stats_percentages["INCONFORT"], C_ALERT, "Inconfort")

        # B. Clim
        clim_x = w - 420
        t_col = C_OK
        if self.current_temp < 19: t_col = C_WARN
        elif self.current_temp > 23: t_col = C_ALERT
        
        cv2.putText(img, f"MODE: {self.climate_mode}", (clim_x, stat_y_title), FONT_MAIN, 0.5, C_TEXT_DARK, 1)
        cv2.putText(img, f"{self.current_temp:.1f} C", (clim_x + 220, stat_y_gauge + 10), FONT_BOLD, 1.2, t_col, 2)
        
        # Barre Clim
        bar_len = 200
        cv2.rectangle(img, (clim_x, stat_y_gauge), (clim_x + bar_len, stat_y_gauge + 8), (200,200,200), -1) 
        ratio = max(0, min(1, (self.current_temp - 15) / 15.0))
        fill = int(bar_len * ratio)
        cv2.rectangle(img, (clim_x, stat_y_gauge), (clim_x + fill, stat_y_gauge + 8), t_col, -1) 
        cv2.circle(img, (clim_x + fill, stat_y_gauge + 4), 5, C_WHITE, -1)

        # 4. BADGE ETAT PRINCIPAL
        if self.state == "RUN":
            st = self.hud_data["global_state"]
            
            bg_col = C_NEUTRAL
            if st == "CONFORT": bg_col = C_OK
            elif st == "INCONFORT": bg_col = C_ALERT
            
            bx, by = 30, header_h + 30
            bw, bh = 220, 50
            
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), bg_col, -1)
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255,255,255), 1)
            cv2.putText(img, st, (bx + 20, by + 35), FONT_BOLD, 0.8, C_WHITE, 2, cv2.LINE_AA)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        cv2.namedWindow("Stellantis Care Monitor", cv2.WINDOW_NORMAL)
        current_clothes = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # --- MACHINE A ETATS ---
            if self.state == "INIT":
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 80), (w, h-100), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cx, cy = w//2, h//2
                cv2.putText(frame, "SYSTEME PRET", (cx-130, cy-20), FONT_BOLD, 1.2, C_WHITE, 2)
                cv2.putText(frame, "[ESPACE] pour Calibrer", (cx-180, cy+30), FONT_MAIN, 0.8, C_WHITE, 1)

            elif self.state == "CALIB":
                elapsed = time.time() - self.calib_start_time
                geo = self.geo_engine.analyze(img_rgb)
                if geo: self.calib_buffer.append(geo)
                cx, cy = w//2, h//2
                cv2.circle(frame, (cx, cy), 100, C_WARN, 2)
                cv2.putText(frame, f"{4.0-elapsed:.1f}", (cx-20, cy+10), FONT_BOLD, 1.5, C_WARN, 3)
                if elapsed > 4.0:
                    self.geo_engine.calibrate(self.calib_buffer)
                    self.state = "RUN"

            elif self.state == "RUN":
                bbox = self.detector.detect(img_rgb)
                self.hud_data["bbox"] = bbox
                
                # Correction du crash bbox
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    
                    # Analyse Vêtements (toutes les 30 frames)
                    if HAS_CLOTHING and self.frame_count % 30 == 0:
                        current_clothes = self.cloth_engine.analyze_attire(img_rgb, bbox)
                        self.hud_data["clothes"] = current_clothes
                    
                    # Emotion (toutes les 3 frames)
                    m = int((y2-y1)*0.1)
                    face_crop = frame[max(0,y1-m):min(h,y2+m), max(0,x1-m):min(w,x2+m)]
                    cnn_res = self.hud_data["cnn_details"]
                    if face_crop.size > 0 and self.frame_count % 3 == 0:
                        lbl, sc, _, _ = self.cnn_engine.analyze_emotion(face_crop)
                        cnn_res = {"label": lbl, "score": sc}
                        self.hud_data["cnn_details"] = cnn_res

                    # Géométrie (Temps réel)
                    geo_res = self.geo_engine.analyze(img_rgb)
                    self.hud_data["geo_details"] = geo_res if geo_res else {}

                    if geo_res:
                        if "Sourire" in geo_res.get("txt_mouth", "") and cnn_res["label"] in ["sad", "angry"]:
                            cnn_res["label"] = "happy"
                        
                        # FUSION INTELLIGENTE
                        final, total = self.fusion_intelligence(geo_res, cnn_res["label"], cnn_res["score"], current_clothes)
                        
                        self.hud_data["global_state"] = final
                        self.update_history_30s(final)
                        self.update_climate()

                        col = C_OK if final == "CONFORT" else (C_ALERT if final == "INCONFORT" else C_NEUTRAL)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            
            self.draw_ui_pro(frame)
            cv2.imshow("Stellantis Care Monitor", frame)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'): break 
            if k == 32: 
                self.state = "CALIB"
                self.calib_start_time = time.time()
                self.calib_buffer = []

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = StellantisUltimateSystem()
    app.run()