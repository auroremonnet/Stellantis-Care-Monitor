"""
server.py — Stellantis CARE Monitor : Pont API FastAPI
=======================================================
Lancement :  python -m uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import threading
import time
from collections import Counter
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from detection import FaceRegionDetector
    from emotion_cnn import EmotionCNNAnalyzer
    from geometry_analysis import FaceGeometryAnalyzer
    from data_logger import ValidationLogger
except ImportError as e:
    raise RuntimeError(f"[FATAL] Module manquant : {e}")

try:
    from vlm_analyzer import VLMAnalyzer
    HAS_VLM = True
except ImportError:
    HAS_VLM = False
    print("[WARN] VLMAnalyzer non disponible — questions VLM désactivées.")

try:
    from clothing_analysis import ClothingAnalyzer
    HAS_CLOTHING = True
except ImportError:
    HAS_CLOTHING = False
    class ClothingAnalyzer:
        def analyze_attire(self, img, bbox): return []

C_OK      = (80,  200, 80)
C_WARN    = (0,   165, 255)
C_ALERT   = (50,   50, 220)
C_NEUTRAL = (160, 160, 160)


class StellantisAPIEngine:

    def __init__(self):
        print("\n=== STELLANTIS CARE MONITOR (MODE API) ===")
        self.detector     = FaceRegionDetector()
        self.geo_engine   = FaceGeometryAnalyzer()
        self.cnn_engine   = EmotionCNNAnalyzer("models/emotion_resnet18_affectnet.pt")
        self.cloth_engine = ClothingAnalyzer()
        self.logger       = ValidationLogger("session_data.csv")
        self.vlm_engine   = VLMAnalyzer() if HAS_VLM else None

        self.state         = "AUTO_CALIB"
        self.calib_buffer  = []
        self.CALIB_FRAMES  = 60

        self.target_temp  = 21.0
        self.current_temp = 21.0
        self.climate_mode = "AUTO"

        self.state_history     = []
        self.stats_percentages = {"CONFORT": 0, "NEUTRE": 0, "INCONFORT": 0}

        self.hud_data = {
            "global_state": "CALIBRATION",
            "geo_details":  {},
            "cnn_details":  {"label": "-", "score": 0.0},
            "clothes":      [],
        }

        self.frame_count     = 0
        self.prev_frame_time = 0.0
        self.current_clothes = []
        self.score_history   = []
        self.SMOOTH_WINDOW   = 10

        self.vlm_question     = None
        self.vlm_running      = False
        self.last_vlm_trigger = 0.0
        self.VLM_INTERVAL     = 15.0

    def update_history_30s(self, current_state: str):
        now = time.time()
        self.state_history.append((now, current_state))
        self.state_history = [x for x in self.state_history if (now - x[0]) <= 30.0]
        total = len(self.state_history)
        if total > 0:
            counts   = Counter([x[1] for x in self.state_history])
            p_conf   = int((counts["CONFORT"]   / total) * 100)
            p_inconf = int((counts["INCONFORT"] / total) * 100)
            p_neutre = int((counts["NEUTRE"]    / total) * 100)
            reste    = 100 - (p_conf + p_inconf + p_neutre)
            p_neutre += reste
            self.stats_percentages["CONFORT"]   = p_conf
            self.stats_percentages["INCONFORT"] = p_inconf
            self.stats_percentages["NEUTRE"]    = p_neutre

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

    def fusion_intelligence(self, geo, cnn_label, cnn_score, clothes_list):
        s_geo = s_cnn = s_cloth = 0.0
        txt_mouth = geo.get("txt_mouth", "")
        txt_brows = geo.get("txt_brows", "")
        txt_eyes  = geo.get("txt_eyes",  "")

        if "Sourire"  in txt_mouth:                            s_geo += 5.0
        if "Fronces"  in txt_brows:                            s_geo -= 6.0
        if "Grimace"  in txt_mouth or "Tension" in txt_mouth:  s_geo -= 4.0
        elif "Baillement" in txt_mouth:                         s_geo -= 5.0

        if "Baillement" in txt_mouth and "Plisses" in txt_eyes:
            return "INCONFORT", s_geo, s_geo, 0.0

        if cnn_score > 0.6:
            if cnn_label == "happy":                      s_cnn += 3.0
            elif cnn_label in ["sad", "angry", "fear"]:   s_cnn -= 3.0

        if HAS_CLOTHING:
            if "DEBARDEUR" in clothes_list: s_cloth += 2.0
            elif "MANTEAU"  in clothes_list: s_cloth -= 2.0
            if "BONNET"    in clothes_list:  s_cloth -= 3.0
            if "LUNETTES"  in clothes_list:  s_cloth += 0.5

        total = s_geo + s_cnn + s_cloth
        self.score_history.append(total)
        if len(self.score_history) > self.SMOOTH_WINDOW:
            self.score_history.pop(0)
        smoothed = float(np.mean(self.score_history))

        if   smoothed >= 4.5:  final = "CONFORT"
        elif smoothed <= -4.0: final = "INCONFORT"
        else:                  final = "NEUTRE"

        return final, total, s_geo, s_cnn

    def _vlm_worker(self, img_rgb: np.ndarray, bbox):
        try:
            crops_data = self.detector.crop_regions(img_rgb, bbox)
            if crops_data and "regions_img" in crops_data:
                result = self.vlm_engine.analyze(crops_data["regions_img"])
                self.vlm_question = self._build_question(result)
        except Exception as e:
            print(f"[VLM] Erreur : {e}")
        finally:
            self.vlm_running = False

    def _build_question(self, vlm_result: dict) -> str:
        b = vlm_result.get("brows", {}).get("etat", "neutre")
        e = vlm_result.get("eyes",  {}).get("etat", "neutre")
        m = vlm_result.get("mouth", {}).get("etat", "neutre")
        if b == "inconfort":
            return "Vos sourcils semblent crispés. Avez-vous trop chaud ou trop froid ?"
        if m == "inconfort":
            return "Vous semblez inconfortable. Souhaitez-vous ajuster la température ?"
        if e == "inconfort":
            return "Vos yeux semblent fatigués. Souhaitez-vous que j'ajuste l'habitacle ?"
        return "Êtes-vous confortable thermiquement ?"

    def maybe_trigger_vlm(self, img_rgb: np.ndarray, bbox):
        if not HAS_VLM or self.vlm_engine is None: return
        if self.vlm_running or self.vlm_question is not None: return
        if self.hud_data["global_state"] != "INCONFORT": return
        if (time.time() - self.last_vlm_trigger) < self.VLM_INTERVAL: return
        self.vlm_running      = True
        self.last_vlm_trigger = time.time()
        threading.Thread(target=self._vlm_worker, args=(img_rgb.copy(), bbox), daemon=True).start()

    def process_frame(self, frame_bgr: np.ndarray):
        self.frame_count += 1
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w    = frame_bgr.shape[:2]

        if self.state == "AUTO_CALIB":
            geo = self.geo_engine.analyze(img_rgb)
            if geo:
                self.calib_buffer.append(geo)
            if len(self.calib_buffer) >= self.CALIB_FRAMES:
                self.geo_engine.calibrate(self.calib_buffer)
                self.state = "RUN"
                print("[SERVER] Calibration automatique terminée → RUN")
            annotated = self._draw_calibration(frame_bgr.copy(), len(self.calib_buffer))
            return annotated, "calibration", "", self.current_temp

        bbox    = self.detector.detect(img_rgb)
        cnn_res = self.hud_data["cnn_details"]

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            if HAS_CLOTHING and self.frame_count % 30 == 0:
                self.current_clothes     = self.cloth_engine.analyze_attire(img_rgb, bbox)
                self.hud_data["clothes"] = self.current_clothes

            if self.frame_count % 3 == 0:
                m = int((y2 - y1) * 0.1)
                face_crop = frame_bgr[max(0,y1-m):min(h,y2+m), max(0,x1-m):min(w,x2+m)]
                if face_crop.size > 0:
                    lbl, sc, _, _ = self.cnn_engine.analyze_emotion(face_crop)
                    cnn_res = {"label": lbl, "score": sc}
                    self.hud_data["cnn_details"] = cnn_res

            geo_res = self.geo_engine.analyze(img_rgb)
            self.hud_data["geo_details"] = geo_res if geo_res else {}

            if geo_res:
                if "Sourire" in geo_res.get("txt_mouth", "") and cnn_res["label"] in ["sad", "angry"]:
                    cnn_res["label"] = "happy"

                final, total, s_geo, s_cnn = self.fusion_intelligence(
                    geo_res, cnn_res["label"], cnn_res["score"], self.current_clothes
                )
                self.hud_data["global_state"] = final
                self.update_history_30s(final)
                self.update_climate()

                now = time.time()
                fps = 1 / (now - self.prev_frame_time) if self.prev_frame_time > 0 else 0
                self.prev_frame_time = now

                self.logger.log_frame(
                    state=final, total_score=total, geo_score=s_geo, cnn_score=s_cnn,
                    temp=self.current_temp, fps=fps,
                    eyes=geo_res.get("txt_eyes","Neutre"),
                    brows=geo_res.get("txt_brows","Stable"),
                    mouth=geo_res.get("txt_mouth","Fermee")
                )
                self.maybe_trigger_vlm(img_rgb, bbox)

        annotated = self._draw_overlay(frame_bgr.copy(), bbox)
        state     = self.hud_data["global_state"]
        primary_emotion = (
            "confortable"   if state == "CONFORT"   else
            "inconfortable" if state == "INCONFORT" else
            "neutre"
        )
        return annotated, cnn_res["label"], primary_emotion, self.current_temp

    def _draw_calibration(self, frame: np.ndarray, progress: int) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        pct   = int((progress / self.CALIB_FRAMES) * 100)
        bar_w = int(w * 0.5)
        bar_x = (w - bar_w) // 2
        bar_y = h // 2 + 30
        cv2.putText(frame, "CALIBRATION EN COURS",
                    (bar_x, h // 2 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Restez neutre face a la camera",
                    (bar_x, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (60,60,60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * pct / 100), bar_y + 10), C_WARN, -1)
        cv2.putText(frame, f"{pct}%", (bar_x + bar_w + 8, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WARN, 1)
        return frame

    def _draw_overlay(self, frame: np.ndarray, bbox) -> np.ndarray:
        state = self.hud_data["global_state"]
        geo   = self.hud_data["geo_details"]
        col   = C_NEUTRAL
        if state == "CONFORT":     col = C_OK
        elif state == "INCONFORT": col = C_ALERT
        if bbox is None:
            return frame
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        if geo and "landmarks" in geo:
            lms = geo["landmarks"]
            for idx in [33, 133, 362, 263, 61, 291, 105, 334, 468, 473]:
                if idx < len(lms):
                    cv2.circle(frame, (int(lms[idx][0]), int(lms[idx][1])), 2, (255,255,255), -1)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Stellantis CARE Monitor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = StellantisAPIEngine()


class FrameRequest(BaseModel):
    frame:       str
    temperature: Optional[float] = None

class VLMResponseRequest(BaseModel):
    response: str


def decode_frame(data_url: str) -> Optional[np.ndarray]:
    try:
        if "," in data_url:
            data_url = data_url.split(",")[1]
        img_bytes = base64.b64decode(data_url)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[decode_frame] Erreur : {e}")
        return None

def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"


@app.post("/analyze")
async def analyze(req: FrameRequest):
    frame = decode_frame(req.frame)
    if frame is None:
        return {"error": "Frame invalide", "emotion": "", "primary_emotion": "",
                "annotated_image": None, "temperature": engine.current_temp}

    annotated, emotion, primary_emotion, temperature = engine.process_frame(frame)

    # Détails FACS pour le dashboard React
    geo = engine.hud_data.get("geo_details", {})
    cnn = engine.hud_data.get("cnn_details", {})
    return {
        "emotion":         emotion,
        "primary_emotion": primary_emotion,
        "annotated_image": encode_frame(annotated),
        "temperature":     round(temperature, 1),
        "state":           engine.hud_data["global_state"],
        "stats":           engine.stats_percentages,
        "climate_mode":    engine.climate_mode,
        "facs": {
            "eyes":  geo.get("txt_eyes",  "—"),
            "brows": geo.get("txt_brows", "—"),
            "mouth": geo.get("txt_mouth", "—"),
        },
        "scores": {
            "geo":   round(engine.score_history[-1] if engine.score_history else 0, 1),
            "cnn":   round(cnn.get("score", 0) * (1 if cnn.get("label") == "happy" else -1), 1),
            "total": round(sum(engine.score_history[-10:]) / max(1, len(engine.score_history[-10:])), 1),
        }
    }


@app.get("/vlm/check")
async def vlm_check():
    return {"question": engine.vlm_question}


@app.post("/vlm/response")
async def vlm_response(req: VLMResponseRequest):
    r = req.response.lower().strip()
    if r == "oui":
        engine.target_temp  = max(16.0, engine.current_temp - 2.0)
        engine.climate_mode = "AJUSTEMENT"
    elif r == "non":
        engine.target_temp  = min(26.0, engine.current_temp + 1.0)
        engine.climate_mode = "AUTO"
        engine.score_history.clear()
    engine.vlm_question = None
    return {"status": "ok", "new_target": round(engine.target_temp, 1)}


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "engine_state": engine.state,
        "temperature":  round(engine.current_temp, 1),
        "climate_mode": engine.climate_mode,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)