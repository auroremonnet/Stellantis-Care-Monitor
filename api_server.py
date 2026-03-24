"""
Stellantis Care Monitor — Flask API Bridge
Connecte le frontend React au pipeline IA Python existant.
Toutes les fonctionnalités originales sont préservées.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
import os
import sys
from collections import Counter

# Ajouter le dossier src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection import FaceRegionDetector
from emotion_cnn import EmotionCNNAnalyzer
from geometry_analysis import FaceGeometryAnalyzer
from data_logger import ValidationLogger

try:
    from clothing_analysis import ClothingAnalyzer
    HAS_CLOTHING = True
except ImportError:
    HAS_CLOTHING = False
    class ClothingAnalyzer:
        def analyze_attire(self, img, bbox): return []

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# SYSTÈME GLOBAL (état persistant entre requêtes)
# ─────────────────────────────────────────────

class StellantisAPISystem:
    """
    Noyau du système de confort — même logique que StellantisUltimateSystem
    mais sans boucle OpenCV (le frontend React gère la caméra).
    """

    def __init__(self):
        print("\n=== STELLANTIS CARE MONITOR — API SERVER ===")
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_resnet18_affectnet.pt")

        self.detector    = FaceRegionDetector()
        self.geo_engine  = FaceGeometryAnalyzer()
        self.cnn_engine  = EmotionCNNAnalyzer(model_path)
        self.cloth_engine = ClothingAnalyzer()

        csv_path = os.path.join(os.path.dirname(__file__), "..", "session_data.csv")
        self.logger = ValidationLogger(csv_path)

        # Machine à états
        self.state          = "INIT"   # INIT | CALIB | RUN
        self.calib_buffer   = []
        self.calib_start_time = 0.0
        self.frame_count    = 0
        self.prev_frame_time = 0.0

        # Climatisation
        self.target_temp  = 21.0
        self.current_temp = 21.0
        self.climate_mode = "AUTO"

        # Historique 30 secondes
        self.state_history = []
        self.stats_percentages = {"CONFORT": 0, "NEUTRE": 0, "INCONFORT": 0}

        # Données HUD courant
        self.hud_data = {
            "global_state": "EN ATTENTE",
            "geo_details":  {},
            "cnn_details":  {"label": "-", "score": 0.0},
            "clothes":      [],
            "bbox":         None
        }

        self.current_clothes = []

        # VLM désactivé côté serveur (réponse OUI/NON gérée par le frontend)
        self.vlm_question    = None
        self.vlm_pending     = False

        print("[API] Système prêt.")

    # ─── ALGORITHME DE FUSION (inchangé) ─────────────────────────────────────

    def fusion_intelligence(self, geo, cnn_label, cnn_score, clothes_list):
        s_geo = s_cnn = s_cloth = 0.0

        txt_mouth = geo.get("txt_mouth", "")
        txt_brows = geo.get("txt_brows", "")

        if "Sourire" in txt_mouth:
            s_geo += 5.0
        if "Fronces" in txt_brows:
            s_geo -= 6.0
        if "Grimace" in txt_mouth or "Tension" in txt_mouth:
            s_geo -= 4.0
        elif "Baillement" in txt_mouth:
            s_geo -= 5.0

        if cnn_score > 0.6:
            if cnn_label == "happy":
                s_cnn += 3.0
            elif cnn_label in ["sad", "angry", "fear"]:
                s_cnn -= 3.0

        if HAS_CLOTHING:
            if "DEBARDEUR" in clothes_list: s_cloth += 2.0
            elif "MANTEAU"  in clothes_list: s_cloth -= 2.0
            if "BONNET"     in clothes_list: s_cloth -= 3.0
            if "LUNETTES"   in clothes_list: s_cloth += 0.5

        total = s_geo + s_cnn + s_cloth

        if   total >= 4.5:  state = "CONFORT"
        elif total <= -4.0: state = "INCONFORT"
        else:               state = "NEUTRE"

        return state, total, s_geo, s_cnn

    # ─── HISTORIQUE 30s (inchangé) ────────────────────────────────────────────

    def update_history_30s(self, current_state):
        now = time.time()
        self.state_history.append((now, current_state))
        self.state_history = [x for x in self.state_history if (now - x[0]) <= 30.0]

        total = len(self.state_history)
        if total > 0:
            counts  = Counter([x[1] for x in self.state_history])
            p_conf  = int((counts["CONFORT"]   / total) * 100)
            p_inconf = int((counts["INCONFORT"] / total) * 100)
            p_neutre = int((counts["NEUTRE"]    / total) * 100)
            reste = 100 - (p_conf + p_inconf + p_neutre)
            p_neutre += reste
            self.stats_percentages["CONFORT"]   = p_conf
            self.stats_percentages["INCONFORT"] = p_inconf
            self.stats_percentages["NEUTRE"]    = p_neutre

    # ─── CLIMATISATION (inchangée) ────────────────────────────────────────────

    def update_climate(self):
        st = self.hud_data["global_state"]
        if st == "INCONFORT":
            self.target_temp  = 18.0
            self.climate_mode = "MODE MAX"
        elif st == "CONFORT":
            self.target_temp  = 21.0
            self.climate_mode = "ECO"
        else:
            self.target_temp  = 22.5
            self.climate_mode = "STANDARD"

        self.current_temp += (self.target_temp - self.current_temp) * 0.02

    # ─── DÉCODAGE IMAGE ───────────────────────────────────────────────────────

    def decode_frame(self, b64_data):
        """Convertit une image base64 (JPEG) en tableau NumPy BGR."""
        if b64_data.startswith("data:"):
            b64_data = b64_data.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ─── ENCODAGE IMAGE ANNOTÉE ────────────────────────────────────────────────

    def encode_frame(self, frame_bgr):
        """Encode en JPEG base64 pour renvoyer au frontend."""
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    # ─── PIPELINE PRINCIPAL ────────────────────────────────────────────────────

    def process_frame(self, frame_bgr):
        """
        Pipeline complet : détection → géométrie → CNN → fusion → clim → log.
        Retourne un dict JSON-sérialisable.
        """
        self.frame_count += 1
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w   = frame_bgr.shape[:2]

        result = {
            "state":          self.state,
            "global_state":   self.hud_data["global_state"],
            "temperature":    round(self.current_temp, 1),
            "climate_mode":   self.climate_mode,
            "stats":          dict(self.stats_percentages),
            "cnn":            dict(self.hud_data["cnn_details"]),
            "geo":            {k: v for k, v in self.hud_data["geo_details"].items() if isinstance(v, str)},
            "clothes":        self.hud_data["clothes"],
            "vlm_question":   self.vlm_question,
            "annotated_image": None,
        }

        # ── INIT ─────────────────────────────────────────────────────────────
        if self.state == "INIT":
            return result

        # ── CALIBRATION ──────────────────────────────────────────────────────
        if self.state == "CALIB":
            elapsed = time.time() - self.calib_start_time
            geo = self.geo_engine.analyze(img_rgb)
            if geo:
                self.calib_buffer.append(geo)

            result["calib_progress"] = min(1.0, elapsed / 4.0)
            result["calib_remaining"] = max(0.0, round(4.0 - elapsed, 1))

            if elapsed > 4.0:
                self.geo_engine.calibrate(self.calib_buffer)
                self.state = "RUN"
                result["state"] = "RUN"

            return result

        # ── RUN ──────────────────────────────────────────────────────────────
        bbox = self.detector.detect(img_rgb)
        self.hud_data["bbox"] = bbox

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            # Vêtements (toutes les 30 frames)
            if HAS_CLOTHING and self.frame_count % 30 == 0:
                self.current_clothes = self.cloth_engine.analyze_attire(img_rgb, bbox)
                self.hud_data["clothes"] = self.current_clothes

            # CNN Émotion (toutes les 3 frames)
            m = int((y2 - y1) * 0.1)
            face_crop = frame_bgr[max(0,y1-m):min(h,y2+m), max(0,x1-m):min(w,x2+m)]
            cnn_res = self.hud_data["cnn_details"]
            if face_crop.size > 0 and self.frame_count % 3 == 0:
                lbl, sc, _, _ = self.cnn_engine.analyze_emotion(face_crop)
                cnn_res = {"label": lbl, "score": sc}
                self.hud_data["cnn_details"] = cnn_res

            # Géométrie (temps réel)
            geo_res = self.geo_engine.analyze(img_rgb)
            self.hud_data["geo_details"] = geo_res if geo_res else {}

            if geo_res:
                # Correction sourire vs CNN
                if "Sourire" in geo_res.get("txt_mouth", "") and cnn_res["label"] in ["sad", "angry"]:
                    cnn_res["label"] = "happy"

                # FUSION
                final, total, s_geo_v, s_cnn_v = self.fusion_intelligence(
                    geo_res, cnn_res["label"], cnn_res["score"], self.current_clothes
                )
                self.hud_data["global_state"] = final
                self.update_history_30s(final)
                self.update_climate()

                # Logging CSV
                new_t = time.time()
                fps   = 1 / (new_t - self.prev_frame_time) if self.prev_frame_time > 0 else 0
                self.prev_frame_time = new_t

                self.logger.log_frame(
                    state=final, total_score=total,
                    geo_score=s_geo_v, cnn_score=s_cnn_v,
                    temp=self.current_temp, fps=fps,
                    eyes=geo_res.get("txt_eyes",  "Neutre"),
                    brows=geo_res.get("txt_brows", "Stable"),
                    mouth=geo_res.get("txt_mouth", "Fermee")
                )

            # Dessin bounding box
            col_map = {"CONFORT": (80,200,80), "INCONFORT": (50,50,220), "NEUTRE": (160,160,160)}
            col = col_map.get(self.hud_data["global_state"], (160,160,160))
            cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), col, 2)

            # Label état sur l'image
            label_txt = self.hud_data["global_state"]
            cv2.rectangle(frame_bgr, (x1, y1-30), (x1+len(label_txt)*12, y1), col, -1)
            cv2.putText(frame_bgr, label_txt, (x1+4, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # Encode l'image annotée
        result["annotated_image"] = self.encode_frame(frame_bgr)
        result["global_state"]    = self.hud_data["global_state"]
        result["temperature"]     = round(self.current_temp, 1)
        result["climate_mode"]    = self.climate_mode
        result["stats"]           = dict(self.stats_percentages)
        result["cnn"]             = dict(self.hud_data["cnn_details"])
        result["geo"]             = {k: v for k, v in self.hud_data["geo_details"].items() if isinstance(v, str)}
        result["clothes"]         = self.hud_data["clothes"]

        return result


# Instance globale
system = StellantisAPISystem()


# ─────────────────────────────────────────────
# ROUTES API
# ─────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def get_status():
    """État courant du système."""
    return jsonify({
        "state":        system.state,
        "global_state": system.hud_data["global_state"],
        "temperature":  round(system.current_temp, 1),
        "climate_mode": system.climate_mode,
        "stats":        dict(system.stats_percentages),
        "vlm_question": system.vlm_question,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Reçoit une frame base64, retourne l'analyse complète + image annotée.
    Body JSON : { "frame": "<base64 JPEG>" }
    """
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Pas de frame fournie"}), 400

    try:
        frame = system.decode_frame(data["frame"])
        if frame is None:
            return jsonify({"error": "Frame invalide"}), 400
        result = system.process_frame(frame)
        return jsonify(result)
    except Exception as e:
        print(f"[ERREUR /analyze] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/calibrate", methods=["POST"])
def start_calibration():
    """Démarre la séquence de calibration (4 secondes)."""
    system.state            = "CALIB"
    system.calib_start_time = time.time()
    system.calib_buffer     = []
    return jsonify({"status": "calibration_started"})


@app.route("/api/temperature", methods=["POST"])
def set_temperature():
    """
    Permet au conducteur de régler manuellement la température.
    Body JSON : { "temperature": 22.5 }
    """
    data = request.get_json(silent=True)
    if not data or "temperature" not in data:
        return jsonify({"error": "Température manquante"}), 400

    temp = float(data["temperature"])
    if not (0 <= temp <= 50):
        return jsonify({"error": "Température hors plage (0–50°C)"}), 400

    system.current_temp  = temp
    system.target_temp   = temp
    system.climate_mode  = "MANUEL"
    return jsonify({"temperature": round(system.current_temp, 1), "climate_mode": system.climate_mode})


@app.route("/api/vlm-response", methods=["POST"])
def vlm_response():
    """
    Reçoit la réponse OUI/NON à la question VLM du frontend.
    Body JSON : { "response": "oui" | "non" }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Body manquant"}), 400

    resp = data.get("response", "").lower()
    system.vlm_pending  = False
    system.vlm_question = None
    print(f"[VLM] Réponse utilisateur : {resp}")
    return jsonify({"acknowledged": True, "response": resp})


@app.route("/api/reset", methods=["POST"])
def reset():
    """Remet le système en état INIT."""
    system.state          = "INIT"
    system.calib_buffer   = []
    system.state_history  = []
    system.stats_percentages = {"CONFORT": 0, "NEUTRE": 0, "INCONFORT": 0}
    system.hud_data["global_state"] = "EN ATTENTE"
    system.vlm_question   = None
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    print("[API] Démarrage sur http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
