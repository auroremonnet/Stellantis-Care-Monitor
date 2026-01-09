import cv2
import numpy as np
import time
import sys
from collections import Counter

# --- IMPORTS LOCAUX ---
from detection import FaceRegionDetector
from emotion_cnn import EmotionCNNAnalyzer
from geometry_analysis import FaceGeometryAnalyzer
from clothing_analysis import ClothingAnalyzer

# --- CONFIGURATION GRAPHIQUE ---
FONT_MAIN = cv2.FONT_HERSHEY_SIMPLEX
C_DARK_BG = (20, 20, 20)
C_PANEL_BG = (40, 40, 45)
C_BAR_BG = (30, 30, 35)
C_TXT_MAIN = (240, 240, 240)
C_TXT_SUB = (180, 180, 180)

# Couleurs
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_INFO = (255, 255, 0)

class StellantisUltimateSystem:
    def __init__(self):
        print("\n=== STELLANTIS V9 (ZERO CONFLIT) ===")
        
        self.detector = FaceRegionDetector() 
        self.geo_engine = FaceGeometryAnalyzer()
        self.cnn_engine = EmotionCNNAnalyzer("models/emotion_resnet18_affectnet.pt")
        self.cloth_engine = ClothingAnalyzer()
        
        print("Systemes prets.\n")
        
        self.state = "INIT"
        self.calib_buffer = []
        self.calib_start_time = 0
        self.frame_count = 0
        
        self.reporting_buffer = []
        self.last_report_time = time.time()
        
        self.emotion_history = [] 
        self.last_stat_update = time.time()
        self.vlm_cache = {"text": "Collecte...", "p_confort": 0, "p_neutre": 0, "p_inconfort": 0}
        
        self.hud_data = {
            "global_state": "EN ATTENTE",
            "confidence_score": 0,
            "geo_details": {},
            "cnn_details": {"label": "-", "score": 0.0},
            "clothes": [],
            "bbox": None
        }

    def update_vlm_memory(self, cnn_label, geo_mouth):
        if cnn_label in ["-", "N/A", None]: return
        category = "NEUTRE"
        is_smiling = "Sourire" in geo_mouth
        if cnn_label == "happy" or is_smiling: category = "CONFORT"
        elif cnn_label in ["sad", "angry", "fear", "disgust", "surprise"]:
            if is_smiling: category = "CONFORT"
            else: category = "INCONFORT"
        else: category = "NEUTRE"
        now = time.time()
        self.emotion_history.append((now, cnn_label, category))
        self.emotion_history = [x for x in self.emotion_history if (now - x[0]) < 30.0]

    def add_to_report(self, final_score, pts_geo, pts_cnn, pts_cloth, state, clothes):
        self.reporting_buffer.append({
            "total": final_score,
            "p_geo": pts_geo,
            "p_cnn": pts_cnn,
            "p_cloth": pts_cloth,
            "state": state,
            "clothes": clothes
        })

    def print_terminal_report(self):
        now = time.time()
        if now - self.last_report_time < 30.0: return
        self.last_report_time = now
        
        if not self.reporting_buffer: return

        count = len(self.reporting_buffer)
        avg_total = sum(x["total"] for x in self.reporting_buffer) / count
        avg_geo = sum(x["p_geo"] for x in self.reporting_buffer) / count
        avg_cnn = sum(x["p_cnn"] for x in self.reporting_buffer) / count
        avg_cloth = sum(x["p_cloth"] for x in self.reporting_buffer) / count
        
        states = [x["state"] for x in self.reporting_buffer]
        c = Counter(states)
        p_conf = int((c["CONFORT"] / count) * 100)
        p_inconf = int((c["INCONFORT"] / count) * 100)
        
        all_clothes = []
        for x in self.reporting_buffer: all_clothes.extend(x["clothes"])
        unique_clothes = list(set(all_clothes))
        str_clothes = ", ".join(unique_clothes) if unique_clothes else "Neutre"

        print("\n" + "="*70)
        print(f"   ANALYSE V9 - ZERO CONFLIT (Moyenne 30s)")
        print("="*70)
        print(f" ETAT GLOBAL        : {avg_total:+.2f} pts")
        print(f" REPARTITION        : Confort {p_conf}% | Inconfort {p_inconf}%")
        print("-" * 70)
        print(f" [1] EMOTIONS       : {avg_cnn:+.2f} pts")
        print(f" [2] GEOMETRIE      : {avg_geo:+.2f} pts")
        print(f" [3] TENUE          : {avg_cloth:+.2f} pts")
        print("-" * 70)
        print(f" DETECTES           : {str_clothes}")
        print("="*70 + "\n")
        self.reporting_buffer = []

    def refresh_vlm_stats(self):
        now = time.time()
        delta = now - self.last_stat_update
        if delta < 30.0: return 
        self.last_stat_update = now
        if not self.emotion_history: return
        cats = [x[2] for x in self.emotion_history]
        total = len(cats)
        c = Counter(cats)
        p_conf = int((c["CONFORT"] / total) * 100)
        p_neut = int((c["NEUTRE"] / total) * 100)
        p_inconf = int((c["INCONFORT"] / total) * 100)
        somme = p_conf + p_neut + p_inconf
        if somme > 0 and somme != 100:
            diff = 100 - somme
            if p_conf >= p_neut: p_conf += diff
            else: p_neut += diff
        if p_conf >= p_neut and p_conf >= p_inconf: txt = f"CLIMAT POSITIF ({p_conf}%)."
        elif p_inconf >= p_neut: txt = f"TENDANCE NEGATIVE ({p_inconf}%)."
        else: txt = f"CONDUITE STANDARD ({p_neut}%)."
        self.vlm_cache = {"text": txt, "p_confort": p_conf, "p_neutre": p_neut, "p_inconfort": p_inconf}

    def fusion_intelligence(self, geo, cnn_label, cnn_score, clothes_list):
        s_geo = 0.0
        s_cnn = 0.0
        s_cloth = 0.0
        
        # --- 1. GEOMETRIE ---
        txt_mouth = geo["txt_mouth"]
        txt_eyes = geo["txt_eyes"]
        txt_brows = geo["txt_brows"]
        if "Sourire" in txt_mouth:
            s_geo += 6.0 if "Grand" in txt_mouth else 4.0
            if "Joie" in txt_eyes: s_geo += 3.0
        elif "Baillement" in txt_mouth: s_geo -= 5.0
        if "Fatigue" in txt_eyes: s_geo -= 4.0
        if "Fronces" in txt_brows: s_geo -= 3.0

        # --- 2. EMOTION ---
        cnn_weight = 3.0 if cnn_score > 0.75 else (1.5 if cnn_score > 0.55 else 0)
        if cnn_weight > 0:
            if cnn_label == "happy": s_cnn += cnn_weight
            elif cnn_label in ["sad", "angry", "disgust", "fear"]: s_cnn -= cnn_weight
            elif cnn_label == "neutral" and abs(s_geo) < 3: s_cnn = 0.0 

        # --- 3. VETEMENTS V9 (Logique sans conflit) ---
        # Le modèle nous a GARANTI qu'il n'y a qu'un seul haut.
        
        # Haut du corps
        if "DEBARDEUR" in clothes_list: s_cloth += 2.0
        elif "T-SHIRT" in clothes_list: s_cloth += 1.0
        elif "PULL" in clothes_list or "PULL CAPUCHE" in clothes_list: s_cloth -= 1.5
        elif "COSTUME" in clothes_list: s_cloth -= 1.0
        elif "MANTEAU" in clothes_list: s_cloth -= 2.0

        # Tête
        if "BONNET" in clothes_list: s_cloth -= 3.0
        elif "CASQUETTE" in clothes_list: s_cloth += 0.5

        # Accessoires
        if "LUNETTES" in clothes_list: s_cloth += 0.5 # Cool
        if "ECHARPE" in clothes_list: s_cloth -= 2.5  # Chaud
        
        # Accumulation (Seulement si compatible)
        # Ex: Bonnet + Manteau
        if "BONNET" in clothes_list and "MANTEAU" in clothes_list:
            s_cloth -= 1.0 # Malus extra

        total_score = s_geo + s_cnn + s_cloth

        if total_score >= 4.5: final_state = "CONFORT"
        elif total_score <= -4.0: final_state = "INCONFORT"
        else: final_state = "NEUTRE / ATTENTIF"
        
        return final_state, total_score, s_geo, s_cnn, s_cloth

    def draw_text_pro(self, img, text, pos, scale=0.6, color=C_TXT_MAIN, thick=1):
        x, y = pos
        cv2.putText(img, text, (x+1, y+1), FONT_MAIN, scale, C_DARK_BG, thick+1)
        cv2.putText(img, text, (x, y), FONT_MAIN, scale, color, thick)

    def draw_bar_row(self, img, label, percentage, y, color):
        x_col = img.shape[1] - 380
        self.draw_text_pro(img, label, (x_col, y + 10), 0.5, C_TXT_SUB)
        bx = x_col + 100
        bar_w = 200
        cv2.rectangle(img, (bx, y), (bx + bar_w, y + 12), C_BAR_BG, -1)
        fill = int(bar_w * (percentage / 100.0))
        if fill > 0: cv2.rectangle(img, (bx, y), (bx + fill, y + 12), color, -1)
        self.draw_text_pro(img, f"{percentage}%", (bx + bar_w + 10, y + 10), 0.5, C_TXT_MAIN)

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        panel_w = 400
        cv2.rectangle(frame, (w - panel_w, 0), (w, h), C_PANEL_BG, -1)
        cv2.line(frame, (w - panel_w, 0), (w - panel_w, h), (60, 60, 70), 3)
        x_col = w - panel_w + 20
        curr_y = 40

        self.draw_text_pro(frame, "STELLANTIS MONITOR", (x_col, curr_y), 0.8, COLOR_GREEN, 2)
        curr_y += 40

        state = self.hud_data["global_state"]
        col_st = C_TXT_MAIN
        if "CONFORT" in state and "INCONFORT" not in state: col_st = COLOR_GREEN
        elif "INCONFORT" in state: col_st = COLOR_RED 
        elif "NEUTRE" in state: col_st = COLOR_ORANGE
        
        self.draw_text_pro(frame, "ETAT CONDUCTEUR", (x_col, curr_y), 0.5, C_TXT_SUB)
        curr_y += 30
        cv2.rectangle(frame, (x_col-10, curr_y-25), (w-10, curr_y+10), C_DARK_BG, -1)
        self.draw_text_pro(frame, state, (x_col, curr_y), 1.0, col_st, 2)
        curr_y += 45

        clothes = self.hud_data["clothes"]
        self.draw_text_pro(frame, "DETECTEUR (CLIP V9)", (x_col, curr_y), 0.6, C_TXT_MAIN, 2)
        curr_y += 25
        if clothes:
            display_clothes = clothes[:3]
            if len(clothes) > 3: display_clothes.append("...")
            txt_clothes = ", ".join(display_clothes)
            col_clothes = COLOR_INFO
            if any(x in ["BONNET", "ECHARPE", "MANTEAU"] for x in clothes): col_clothes = COLOR_RED
            if "DEBARDEUR" in clothes: col_clothes = COLOR_GREEN
        else:
            txt_clothes = "..."
            col_clothes = C_TXT_SUB
        self.draw_text_pro(frame, f"{txt_clothes}", (x_col, curr_y), 0.5, col_clothes)
        curr_y += 35

        geo = self.hud_data["geo_details"]
        if geo:
            self.draw_text_pro(frame, f"Bouche: {geo.get('txt_mouth','-')}", (x_col, curr_y), 0.45, C_TXT_SUB)
            curr_y += 20
            self.draw_text_pro(frame, f"Yeux: {geo.get('txt_eyes','-')}", (x_col, curr_y), 0.45, C_TXT_SUB)
            curr_y += 20
            self.draw_text_pro(frame, f"Sourcils: {geo.get('txt_brows','-')}", (x_col, curr_y), 0.45, C_TXT_SUB)
            curr_y += 35

        cnn = self.hud_data["cnn_details"]
        if cnn["label"] != "-":
            fr = cnn["label"].upper()
            if fr == "HAPPY": fr = "JOIE"
            self.draw_text_pro(frame, f"EMOTION: {fr} ({int(cnn['score']*100)}%)", (x_col, curr_y), 0.6, COLOR_INFO, 2)
            curr_y += 35

        cv2.line(frame, (x_col, curr_y), (w-20, curr_y), C_TXT_SUB, 1)
        curr_y += 25
        
        self.draw_text_pro(frame, self.vlm_cache["text"], (x_col, curr_y), 0.5, C_TXT_MAIN, 1)
        curr_y += 30
        self.draw_bar_row(frame, "CONFORT", self.vlm_cache["p_confort"], curr_y, COLOR_GREEN)
        curr_y += 25
        self.draw_bar_row(frame, "NEUTRE", self.vlm_cache["p_neutre"], curr_y, C_TXT_SUB)
        curr_y += 25
        self.draw_bar_row(frame, "INCONFORT", self.vlm_cache["p_inconfort"], curr_y, COLOR_RED)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        cv2.namedWindow("Stellantis V9", cv2.WINDOW_NORMAL)

        current_clothes = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frame_count += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            if self.state == "INIT":
                frame[:] = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)[:]
                cx, cy = w//2, h//2
                self.draw_text_pro(frame, "INITIALISATION", (cx-100, cy-50), 1.0, COLOR_ORANGE)
                self.draw_text_pro(frame, "[ESPACE] pour calibrer", (cx-120, cy+20), 0.8, C_TXT_MAIN)

            elif self.state == "CALIB":
                elapsed = time.time() - self.calib_start_time
                geo = self.geo_engine.analyze(img_rgb)
                if geo: self.calib_buffer.append(geo)
                self.draw_text_pro(frame, f"CALIBRATION... {4.0-elapsed:.1f}s", (50, 50), 1.0, COLOR_ORANGE)
                if elapsed > 4.0:
                    self.geo_engine.calibrate(self.calib_buffer)
                    self.state = "RUN"
                    self.last_stat_update = time.time()
                    self.last_report_time = time.time() 

            elif self.state == "RUN":
                bbox = self.detector.detect(img_rgb)
                self.hud_data["bbox"] = bbox
                face_crop = None
                cnn_res = self.hud_data["cnn_details"]
                
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    
                    if self.frame_count % 30 == 0:
                        current_clothes = self.cloth_engine.analyze_attire(img_rgb, bbox)
                        self.hud_data["clothes"] = current_clothes
                    
                    m = int((y2-y1)*0.1)
                    face_crop = frame[max(0,y1-m):min(h,y2+m), max(0,x1-m):min(w,x2+m)]
                    
                    if face_crop.size > 0 and self.frame_count % 3 == 0:
                        lbl, sc, _, _ = self.cnn_engine.analyze_emotion(face_crop)
                        cnn_res = {"label": lbl, "score": sc}
                        self.hud_data["cnn_details"] = cnn_res
                        geo_tmp = self.geo_engine.analyze(img_rgb)
                        mouth = geo_tmp.get("txt_mouth", "") if geo_tmp else ""
                        self.update_vlm_memory(lbl, mouth)

                self.refresh_vlm_stats()
                self.print_terminal_report()

                geo_res = self.geo_engine.analyze(img_rgb)
                if geo_res:
                    if "Sourire" in geo_res.get("txt_mouth", "") and cnn_res["label"] in ["sad", "angry"]:
                        cnn_res["label"] = "happy"
                    
                    final, total, s_geo, s_cnn, s_cloth = self.fusion_intelligence(
                        geo_res, cnn_res["label"], cnn_res["score"], current_clothes
                    )
                    self.hud_data["global_state"] = final
                    self.hud_data["confidence_score"] = total
                    self.hud_data["geo_details"] = geo_res
                    self.add_to_report(total, s_geo, s_cnn, s_cloth, final, current_clothes)
                
                self.draw_hud(frame)

            cv2.imshow("Stellantis V9", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break 
            if k == 32 and self.state == "INIT":
                self.state = "CALIB"
                self.calib_start_time = time.time()
                self.calib_buffer = []

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        sys = StellantisUltimateSystem()
        sys.run()
    except Exception as e:
        print(f"\n!!! ERREUR CRITIQUE !!!\n{e}")
        import traceback
        traceback.print_exc()