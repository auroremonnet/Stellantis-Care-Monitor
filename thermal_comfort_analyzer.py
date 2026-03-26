import os
import sys
import cv2
import webbrowser
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

# ============================================================
# IMPORT MEDIAPIPE ROBUSTE
# ============================================================
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    HAS_MP = True
except Exception:
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        HAS_MP = True
    except Exception:
        HAS_MP = False
        mp_face_mesh = None


# ============================================================
# CONSTANTES LANDMARKS
# ============================================================
MP_ZONE_INDICES = {
    "forehead": [10, 67, 109, 108, 151, 337, 338, 297, 299, 69],
    "nose": [1, 2, 4, 5, 6, 19, 94, 195, 197],
    "cheek_left": [50, 101, 116, 117, 118, 119, 120, 123, 187, 205, 206, 207],
    "cheek_right": [280, 330, 345, 346, 347, 348, 349, 352, 411, 425, 426, 427],
    "chin": [152, 148, 149, 150, 169, 170, 171, 175, 176, 377, 378, 379, 394, 395, 396, 400],
}

INNER_CANTHUS_LEFT = 133
INNER_CANTHUS_RIGHT = 362
NOSE_TIP = 1


# ============================================================
# DATACLASSES
# ============================================================
@dataclass
class ZoneStats:
    mean: float = 0.0
    std: float = 0.0
    n: int = 0


@dataclass
class ComfortResult:
    verdict: str
    direction: str
    score: float
    confidence: float
    action: str
    temperatures: Dict[str, float]
    score_parts: Dict[str, float]
    notes: List[str]


# ============================================================
# ANALYSEUR PRINCIPAL
# ============================================================
class ThermalImageAnalyzer:
    def __init__(self, temp_min: float = 18.0, temp_max: float = 36.2):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.face_mesh = None

        if HAS_MP:
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.35,
            )

    # --------------------------------------------------------
    # 1) Chargement
    # --------------------------------------------------------
    def load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")
        return img

    # --------------------------------------------------------
    # 2) Détection écran HIKMICRO
    # --------------------------------------------------------
    def auto_crop_thermal_screen(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        if w <= 900 and h <= 900:
            return img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 15, 245)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img.copy()

        best = None
        best_score = -1.0

        for c in contours:
            x, y, rw, rh = cv2.boundingRect(c)
            area = rw * rh
            aspect = rw / max(rh, 1)

            score = float(area)
            if 0.55 <= aspect <= 1.9:
                score *= 1.3

            if score > best_score:
                best_score = score
                best = (x, y, rw, rh)

        if best is None:
            return img.copy()

        x, y, rw, rh = best
        pad = 8

        x1 = max(0, x + pad)
        y1 = max(0, y + pad)
        x2 = min(w, x + rw - pad)
        y2 = min(h, y + rh - pad)

        cropped = img[y1:y2, x1:x2].copy()
        return cropped if cropped.size > 0 else img.copy()

    # --------------------------------------------------------
    # 3) Suppression UI HIKMICRO
    # --------------------------------------------------------
    def remove_hikmicro_ui_margins(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x1 = int(w * 0.15)
        y1 = int(h * 0.08)
        cropped = img[y1:h, x1:w].copy()
        return cropped if cropped.size > 0 else img.copy()

    # --------------------------------------------------------
    # 4) Palette -> température
    # --------------------------------------------------------
    def colormap_to_temperature(self, img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        score_h = np.zeros_like(H, dtype=np.float32)

        mask_cold = H > 80
        score_h[mask_cold] = np.clip((130 - H[mask_cold]) / 130.0, 0.0, 0.35)

        mask_mid = (H > 35) & (H <= 80)
        score_h[mask_mid] = 0.30 + (80 - H[mask_mid]) / 80.0 * 0.25

        mask_warm = (H > 12) & (H <= 35)
        score_h[mask_warm] = 0.55 + (35 - H[mask_warm]) / 35.0 * 0.20

        mask_hot = (H <= 12) | (H >= 170)
        score_h[mask_hot] = 0.90

        mask_white = (V > 220) & (S < 60)
        score_h[mask_white] = 1.0

        thermal_index = 0.72 * score_h + 0.28 * (V / 255.0)
        thermal = self.temp_min + thermal_index * (self.temp_max - self.temp_min)
        return thermal.astype(np.float32)

    # --------------------------------------------------------
    # 5) Préparation image pour MediaPipe
    # --------------------------------------------------------
    def prepare_image_for_mediapipe(self, img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return enhanced

    # --------------------------------------------------------
    # 6) Landmarks hybrides
    # --------------------------------------------------------
    def detect_face_landmarks_hybrid(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.face_mesh is None:
            return None

        enhanced = self.prepare_image_for_mediapipe(img_bgr)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        h, w = img_bgr.shape[:2]
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lms = results.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    # --------------------------------------------------------
    # 7) Détection thermique tête / visage
    # --------------------------------------------------------
    def detect_head_bbox_from_thermal(self, thermal: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = thermal.shape[:2]

        p55 = np.percentile(thermal, 55)
        p72 = np.percentile(thermal, 72)
        thresh = 0.45 * p55 + 0.55 * p72

        mask = (thermal >= thresh).astype(np.uint8) * 255

        kernel_close = np.ones((11, 11), np.uint8)
        kernel_open = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = -1.0

        for c in contours:
            x, y, rw, rh = cv2.boundingRect(c)
            area = rw * rh
            if area < 0.03 * w * h:
                continue

            aspect = rw / max(rh, 1)
            if not (0.45 <= aspect <= 1.25):
                continue

            cx = x + rw / 2
            cy = y + rh / 2

            center_bonus = 1.0 - abs(cx - w / 2) / (w / 2)
            upper_bonus = 1.0 - abs(cy - h * 0.43) / max(h * 0.43, 1)

            score = area * (1 + 0.40 * max(0, center_bonus)) * (1 + 0.30 * max(0, upper_bonus))

            if score > best_score:
                best_score = score
                best = (x, y, rw, rh)

        if best is None:
            return None

        x, y, rw, rh = best
        pad_x = int(rw * 0.12)
        pad_y = int(rh * 0.10)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + rw + pad_x)
        y2 = min(h, y + rh + pad_y)

        return (x1, y1, x2, y2)

    # --------------------------------------------------------
    # 8) ROI depuis landmarks MediaPipe
    # --------------------------------------------------------
    def build_face_rois_from_landmarks(self, landmarks: np.ndarray, img_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
        h, w = img_shape[:2]
        rois: Dict[str, np.ndarray] = {}

        for name, idxs in MP_ZONE_INDICES.items():
            pts = landmarks[idxs].astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            rois[name] = cv2.convexHull(pts)

        inter_eye = float(np.linalg.norm(landmarks[33] - landmarks[263])) + 1e-6
        r = max(4, int(inter_eye * 0.04))

        for name, idx in [
            ("inner_canthus_left", INNER_CANTHUS_LEFT),
            ("inner_canthus_right", INNER_CANTHUS_RIGHT),
            ("nose_tip", NOSE_TIP),
        ]:
            x, y = landmarks[idx].astype(int)
            poly = cv2.ellipse2Poly((int(x), int(y)), (r, r), 0, 0, 360, 30)
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            rois[name] = poly.astype(np.int32)

        return rois

    # --------------------------------------------------------
    # 9) ROI fallback géométriques
    # --------------------------------------------------------
    def build_face_rois_from_bbox(self, bbox: Tuple[int, int, int, int]) -> Dict[str, np.ndarray]:
        x1, y1, x2, y2 = bbox
        fw = x2 - x1
        fh = y2 - y1
        cx = x1 + fw // 2

        rois = {
            "forehead": np.array([
                [int(x1 + fw * 0.28), int(y1 + fh * 0.08)],
                [int(x1 + fw * 0.72), int(y1 + fh * 0.08)],
                [int(x1 + fw * 0.68), int(y1 + fh * 0.28)],
                [int(x1 + fw * 0.32), int(y1 + fh * 0.28)],
            ], dtype=np.int32),

            "nose": np.array([
                [int(cx - fw * 0.10), int(y1 + fh * 0.34)],
                [int(cx + fw * 0.10), int(y1 + fh * 0.34)],
                [int(cx + fw * 0.08), int(y1 + fh * 0.58)],
                [int(cx - fw * 0.08), int(y1 + fh * 0.58)],
            ], dtype=np.int32),

            "nose_tip": cv2.ellipse2Poly(
                (int(cx), int(y1 + fh * 0.52)),
                (max(5, int(fw * 0.04)), max(5, int(fw * 0.04))),
                0, 0, 360, 30
            ).astype(np.int32),

            "cheek_left": np.array([
                [int(x1 + fw * 0.12), int(y1 + fh * 0.34)],
                [int(x1 + fw * 0.32), int(y1 + fh * 0.30)],
                [int(x1 + fw * 0.36), int(y1 + fh * 0.62)],
                [int(x1 + fw * 0.14), int(y1 + fh * 0.66)],
            ], dtype=np.int32),

            "cheek_right": np.array([
                [int(x1 + fw * 0.68), int(y1 + fh * 0.30)],
                [int(x1 + fw * 0.88), int(y1 + fh * 0.34)],
                [int(x1 + fw * 0.86), int(y1 + fh * 0.66)],
                [int(x1 + fw * 0.64), int(y1 + fh * 0.62)],
            ], dtype=np.int32),

            "chin": np.array([
                [int(cx - fw * 0.16), int(y1 + fh * 0.68)],
                [int(cx + fw * 0.16), int(y1 + fh * 0.68)],
                [int(cx + fw * 0.14), int(y1 + fh * 0.92)],
                [int(cx - fw * 0.14), int(y1 + fh * 0.92)],
            ], dtype=np.int32),

            "inner_canthus_left": cv2.ellipse2Poly(
                (int(x1 + fw * 0.42), int(y1 + fh * 0.34)),
                (max(4, int(fw * 0.03)), max(4, int(fw * 0.03))),
                0, 0, 360, 30
            ).astype(np.int32),

            "inner_canthus_right": cv2.ellipse2Poly(
                (int(x1 + fw * 0.58), int(y1 + fh * 0.34)),
                (max(4, int(fw * 0.03)), max(4, int(fw * 0.03))),
                0, 0, 360, 30
            ).astype(np.int32),
        }

        return rois

    # --------------------------------------------------------
    # 10) Masques / extraction
    # --------------------------------------------------------
    def polygon_mask(self, shape: Tuple[int, int], poly: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        poly = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillConvexPoly(mask, poly, 255)
        return mask

    def robust_pixels(self, thermal: np.ndarray, mask: np.ndarray, floor: float = 24.0) -> np.ndarray:
        vals = thermal[mask == 255]
        vals = vals[np.isfinite(vals)]
        vals = vals[vals >= floor]

        if len(vals) == 0:
            return vals

        p10, p90 = np.percentile(vals, [10, 90])
        vals = vals[(vals >= p10) & (vals <= p90)]
        return vals

    def extract_zone_stats(self, thermal: np.ndarray, rois: Dict[str, np.ndarray]) -> Dict[str, ZoneStats]:
        stats = {}
        for name, poly in rois.items():
            mask = self.polygon_mask(thermal.shape, poly)
            vals = self.robust_pixels(thermal, mask)

            if len(vals) == 0:
                stats[name] = ZoneStats()
                continue

            stats[name] = ZoneStats(
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                n=int(len(vals)),
            )
        return stats

    # --------------------------------------------------------
    # 11) Classification
    # --------------------------------------------------------
    def classify(self, zs: Dict[str, ZoneStats]) -> ComfortResult:
        def T(name: str, default=32.0):
            return zs[name].mean if name in zs and zs[name].n > 0 else default

        t_front = T("forehead", 33.0)
        t_nez = T("nose", 32.0)
        t_nez_tip = T("nose_tip", t_nez)
        t_jg = T("cheek_left", 32.0)
        t_jd = T("cheek_right", 32.0)
        t_joues = (t_jg + t_jd) / 2.0
        t_menton = T("chin", 32.0)
        t_cg = T("inner_canthus_left", 34.0)
        t_cd = T("inner_canthus_right", 34.0)
        t_canthus = (t_cg + t_cd) / 2.0

        g_front_nez = t_front - t_nez
        g_canthus_nez = t_canthus - t_nez
        asym_joues = abs(t_jg - t_jd)

        score_parts = {
            "gradient_front_nez": 0.0,
            "gradient_canthus_nez": 0.0,
            "temperature_nez": 0.0,
            "temperature_front": 0.0,
            "asymetrie_joues": 0.0,
        }

        notes = []
        conf = []

        if g_front_nez >= 2.2:
            score_parts["gradient_front_nez"] = -1.8
            notes.append(f"Gradient front/nez élevé ({g_front_nez:.2f}°C)")
            conf.append(0.85)
        elif g_front_nez >= 1.2:
            score_parts["gradient_front_nez"] = -0.8
            conf.append(0.70)
        else:
            conf.append(0.60)

        if g_canthus_nez >= 2.5:
            score_parts["gradient_canthus_nez"] = -1.0
            notes.append(f"Gradient canthus/nez élevé ({g_canthus_nez:.2f}°C)")
            conf.append(0.80)
        elif g_canthus_nez <= 1.0:
            score_parts["gradient_canthus_nez"] = +0.4
            conf.append(0.60)

        if t_nez < 30.0:
            score_parts["temperature_nez"] = -1.4
            notes.append(f"Nez froid ({t_nez:.2f}°C)")
            conf.append(0.85)
        elif t_nez > 34.0:
            score_parts["temperature_nez"] = +0.8
            conf.append(0.70)

        if t_front > 35.6:
            score_parts["temperature_front"] = +0.8
            notes.append(f"Front chaud ({t_front:.2f}°C)")
            conf.append(0.70)
        elif t_front < 32.0:
            score_parts["temperature_front"] = -0.5
            conf.append(0.65)

        if asym_joues > 1.2:
            score_parts["asymetrie_joues"] = -0.2
            notes.append(f"Asymétrie joues ({asym_joues:.2f}°C)")
            conf.append(0.50)

        score = float(np.clip(sum(score_parts.values()), -5.0, 5.0))
        confidence = float(np.mean(conf)) if conf else 0.5

        if score <= -1.6:
            verdict = "INCONFORT"
            direction = "froid"
            action = "Augmenter légèrement la température ou réduire le flux d'air froid."
        elif score >= 1.6:
            verdict = "INCONFORT"
            direction = "chaud"
            action = "Renforcer légèrement la ventilation ou le refroidissement."
        elif abs(score) <= 0.8:
            verdict = "CONFORT"
            direction = "neutre"
            action = "Maintenir les conditions actuelles."
        else:
            verdict = "NEUTRE"
            direction = "froid" if score < 0 else "chaud"
            action = "Ajustement léger recommandé."

        temperatures = {
            "front": round(t_front, 2),
            "nez": round(t_nez, 2),
            "pointe_nez": round(t_nez_tip, 2),
            "joue_gauche": round(t_jg, 2),
            "joue_droite": round(t_jd, 2),
            "joues_moyenne": round(t_joues, 2),
            "menton": round(t_menton, 2),
            "canthus_moyenne": round(t_canthus, 2),
        }

        return ComfortResult(
            verdict=verdict,
            direction=direction,
            score=round(score, 2),
            confidence=round(confidence, 2),
            action=action,
            temperatures=temperatures,
            score_parts={k: round(v, 2) for k, v in score_parts.items()},
            notes=notes,
        )

    # --------------------------------------------------------
    # 12) Dessin
    # --------------------------------------------------------
    def draw_result(
        self,
        img_bgr: np.ndarray,
        rois: Dict[str, np.ndarray],
        stats: Dict[str, ZoneStats],
        result: ComfortResult
    ) -> np.ndarray:
            vis = img_bgr.copy()

            colors = {
                "forehead": (255, 200, 80),
                "nose": (80, 180, 255),
                "chin": (220, 140, 255),
                "cheek_left": (100, 255, 100),
                "cheek_right": (100, 255, 100),
                "inner_canthus_left": (255, 255, 255),
                "inner_canthus_right": (255, 255, 255),
                "nose_tip": (0, 220, 255),
            }

            labels = {
                "forehead": "Front",
                "nose": "Nez",
                "cheek_left": "Joue G",
                "cheek_right": "Joue D",
                "chin": "Menton",
                "nose_tip": "Pointe nez",
            }

            for name, poly in rois.items():
                color = colors.get(name, (200, 200, 200))
                poly_draw = np.asarray(poly, dtype=np.int32).reshape(-1, 2)

                overlay = vis.copy()
                cv2.fillConvexPoly(overlay, poly_draw, color)
                vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
                cv2.polylines(vis, [poly_draw], True, color, 2, cv2.LINE_AA)

                if name in stats and stats[name].n > 0 and name in labels:
                    cx = int(np.mean(poly_draw[:, 0]))
                    cy = int(np.mean(poly_draw[:, 1]))
                    txt = f"{labels[name]} {stats[name].mean:.1f}C"

                    cv2.putText(
                        vis, txt, (cx - 36, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
                    )

            return vis
    def build_ai_summary(self, result: ComfortResult) -> Tuple[str, str]:
        """
        Génère un résumé court et professionnel pour le dashboard HTML.
        Retourne :
        - un titre de synthèse
        - un paragraphe explicatif
        """
        score = result.score
        verdict = result.verdict
        direction = result.direction

        t = result.temperatures
        sp = result.score_parts

        joue_g = t.get("joue_gauche", 0.0)
        joue_d = t.get("joue_droite", 0.0)
        nez = t.get("nez", 0.0)
        front = t.get("front", 0.0)
        canthus = t.get("canthus_moyenne", 0.0)

        g_fn_part = sp.get("gradient_front_nez", 0.0)
        g_cn_part = sp.get("gradient_canthus_nez", 0.0)
        nose_part = sp.get("temperature_nez", 0.0)
        front_part = sp.get("temperature_front", 0.0)
        asym_part = sp.get("asymetrie_joues", 0.0)

        if verdict == "INCONFORT" and direction == "froid":
            title = "Résumé IA — inconfort thermique à tendance froide"
            text = (
                f"Le système détecte un profil compatible avec un inconfort froid. "
                f"Le nez ({nez:.1f}°C) apparaît relativement plus froid que les zones de référence, "
                f"notamment le front ({front:.1f}°C) et le canthus moyen ({canthus:.1f}°C). "
                f"Les gradients thermiques faciaux contribuent négativement au score "
                f"(front/nez {g_fn_part:+.2f}, canthus/nez {g_cn_part:+.2f}), ce qui suggère "
                f"un refroidissement périphérique du visage. "
                f"La situation justifie une correction thermique vers plus de chaleur."
            )
            return title, text

        if verdict == "INCONFORT" and direction == "chaud":
            title = "Résumé IA — inconfort thermique à tendance chaude"
            text = (
                f"Le système détecte un profil compatible avec un inconfort chaud. "
                f"La perfusion thermique faciale est globalement élevée, avec un nez à {nez:.1f}°C "
                f"et un front à {front:.1f}°C, ce qui contribue à un score orienté vers le chaud "
                f"(nez {nose_part:+.2f}, front {front_part:+.2f}). "
                f"L’ensemble suggère une charge thermique faciale élevée et une nécessité "
                f"d’augmenter la ventilation ou le refroidissement."
            )
            return title, text

        if verdict == "NEUTRE":
            tendance = "froide" if score < 0 else "chaude"
            title = f"Résumé IA — état neutre avec légère tendance {tendance}"
            text = (
                f"Le système ne détecte pas d’inconfort thermique franc. "
                f"Les signaux restent modérés et aucun critère ne domine fortement la décision finale. "
                f"Le score global ({score:+.2f}) indique un état intermédiaire encore acceptable, "
                f"avec une légère tendance {tendance}. "
                f"Les températures faciales mesurées — joue gauche {joue_g:.1f}°C, "
                f"joue droite {joue_d:.1f}°C, nez {nez:.1f}°C, front {front:.1f}°C — "
                f"restent globalement cohérentes avec une situation stable, mais à surveiller."
            )
            return title, text

        title = "Résumé IA — confort thermique global"
        text = (
            f"Le système détecte un état de confort thermique. "
            f"La distribution de température sur le visage reste globalement équilibrée, "
            f"sans écart critique entre les zones centrales et périphériques. "
            f"Les températures principales — joue gauche {joue_g:.1f}°C, joue droite {joue_d:.1f}°C, "
            f"nez {nez:.1f}°C et front {front:.1f}°C — sont compatibles avec un état neutre et stable. "
            f"Aucune correction thermique immédiate n’est nécessaire."
        )
        return title, text    

    # --------------------------------------------------------
    # 13) Rapport HTML
    # --------------------------------------------------------
    def generate_html_report(
        self,
        image_path: str,
        annotated_path: str,
        result: ComfortResult,
        output_html: Optional[str] = None
    ) -> str:
        if output_html is None:
            base = os.path.splitext(image_path)[0]
            output_html = f"{base}_thermal_report.html"

        state_color = {
            "CONFORT": "#16a34a",
            "NEUTRE": "#6b7280",
            "INCONFORT": "#dc2626"
        }.get(result.verdict, "#6b7280")

        score_color = "#16a34a" if result.score > 0 else "#dc2626" if result.score < 0 else "#6b7280"

        summary_title, summary_text = self.build_ai_summary(result)

        cursor_left = min(98, max(2, ((max(-5, min(5, result.score)) + 5) / 10) * 100))

        notes_html = "".join(f"<li>{n}</li>" for n in result.notes) or "<li>Aucune remarque particulière.</li>"

        temp_cards = f"""
        <div class="metric-card">
        <div class="metric-label">Joue gauche</div>
        <div class="metric-value">{result.temperatures['joue_gauche']:.2f} °C</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">Joue droite</div>
        <div class="metric-value">{result.temperatures['joue_droite']:.2f} °C</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">Nez</div>
        <div class="metric-value">{result.temperatures['nez']:.2f} °C</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">Front</div>
        <div class="metric-value">{result.temperatures['front']:.2f} °C</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">Canthus moyen</div>
        <div class="metric-value">{result.temperatures['canthus_moyenne']:.2f} °C</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">Menton</div>
        <div class="metric-value">{result.temperatures['menton']:.2f} °C</div>
        </div>
        """

        score_rows_left = f"""
        <div class="detail-row">
        <div class="detail-name">Part gradient front / nez</div>
        <div class="detail-value">{result.score_parts['gradient_front_nez']:+.2f}</div>
        </div>
        <div class="detail-row">
        <div class="detail-name">Part gradient canthus / nez</div>
        <div class="detail-value">{result.score_parts['gradient_canthus_nez']:+.2f}</div>
        </div>
        <div class="detail-row">
        <div class="detail-name">Part température nez</div>
        <div class="detail-value">{result.score_parts['temperature_nez']:+.2f}</div>
        </div>
        """

        score_rows_right = f"""
        <div class="detail-row">
        <div class="detail-name">Part température front</div>
        <div class="detail-value">{result.score_parts['temperature_front']:+.2f}</div>
        </div>
        <div class="detail-row">
        <div class="detail-name">Part asymétrie joues</div>
        <div class="detail-value">{result.score_parts['asymetrie_joues']:+.2f}</div>
        </div>
        <div class="detail-row">
        <div class="detail-name">Score final</div>
        <div class="detail-value">{result.score:+.2f}</div>
        </div>
        """

        html = f"""<!DOCTYPE html>
    <html lang="fr">
    <head>
    <meta charset="UTF-8">
    <title>CARE — Rapport thermique</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

    *{{margin:0;padding:0;box-sizing:border-box}}

    :root{{
    --bg:#f0f2f5;
    --white:#ffffff;
    --black:#0a0a0a;
    --border:#e2e5ea;
    --text:#1a1d23;
    --muted:#6b7280;
    --radius:14px;
    --shadow:0 2px 16px rgba(0,0,0,.06), 0 1px 4px rgba(0,0,0,.04);
    }}

    body{{
    font-family:'DM Sans',sans-serif;
    background:var(--bg);
    color:var(--text);
    min-height:100vh;
    }}

    .app-header{{
    background:var(--white);
    border-bottom:1px solid var(--border);
    padding:0 28px;
    height:60px;
    position:sticky;
    top:0;
    z-index:100;
    box-shadow:0 1px 6px rgba(0,0,0,.05);
    }}

    .header-content{{
    max-width:1080px;
    margin:0 auto;
    height:100%;
    display:grid;
    grid-template-columns:200px 1fr 200px;
    align-items:center;
    }}

    .logo-space{{
    display:flex;
    align-items:center;
    }}

    .logo-space img{{
    height:32px;
    object-fit:contain;
    }}

    .app-title{{
    font-family:'DM Mono',monospace;
    font-size:18px;
    font-weight:500;
    letter-spacing:12px;
    color:var(--black);
    text-align:center;
    }}

    .header-right{{
    display:flex;
    justify-content:flex-end;
    }}

    .hdr-pills{{
    display:flex;
    gap:8px;
    align-items:center;
    }}

    .hdr-pill{{
    display:flex;
    align-items:center;
    gap:6px;
    padding:5px 12px;
    border-radius:20px;
    background:var(--bg);
    border:1px solid var(--border);
    font-family:'DM Mono',monospace;
    font-size:11px;
    color:var(--muted);
    }}

    .hdr-pill-dot{{
    width:7px;
    height:7px;
    border-radius:50%;
    flex-shrink:0;
    background:{state_color};
    animation:pulse 2s ease-in-out infinite;
    }}

    @keyframes pulse {{
    0%,100%{{opacity:1}}
    50%{{opacity:0.4}}
    }}

    .hdr-pill-bold{{font-weight:600;color:var(--text)}}
    .hdr-pill-sep{{color:var(--border)}}
    .hdr-pill-icon{{font-size:13px}}

    .wrap{{
    max-width:1080px;
    margin:0 auto;
    padding:20px 28px 40px;
    }}

    .panel{{
    background:#fff;
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:20px 24px;
    box-shadow:var(--shadow);
    margin-bottom:16px;
    }}

    .hero{{
    display:flex;
    justify-content:center;
    align-items:center;
    padding:18px;
    min-height:320px;
    }}

    .hero img{{
    width:auto;
    max-width:100%;
    max-height:68vh;
    height:auto;
    object-fit:contain;
    border-radius:12px;
    display:block;
    margin:0 auto;
    box-shadow:0 2px 10px rgba(0,0,0,.08);
    }}

    .title{{
    font-family:'DM Mono',monospace;
    font-size:11px;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:6px;
    }}

    .sub{{
    font-size:12px;
    color:var(--muted);
    margin-bottom:16px;
    }}

    .top{{
    display:grid;
    grid-template-columns:1.2fr 1fr 1fr;
    gap:14px;
    }}

    .big{{
    border:2px solid {state_color};
    border-radius:12px;
    padding:18px;
    background:#fafbfc;
    }}

    .big-label{{
    font-family:'DM Mono',monospace;
    font-size:10px;
    color:var(--muted);
    text-transform:uppercase;
    }}

    .big-value{{
    font-size:28px;
    font-weight:700;
    color:{state_color};
    margin-top:6px;
    }}

    .big-meta{{
    font-size:13px;
    color:var(--muted);
    margin-top:6px;
    line-height:1.5;
    }}

    .mini{{
    border:1px solid var(--border);
    border-radius:12px;
    padding:18px;
    background:#fafbfc;
    }}

    .mini-value{{
    font-family:'DM Mono',monospace;
    font-size:26px;
    font-weight:700;
    margin-top:8px;
    }}

    .summary-box{{
    margin-top:14px;
    border:1px solid var(--border);
    border-left:4px solid {state_color};
    border-radius:12px;
    background:#fafbfc;
    padding:16px 18px;
    }}

    .summary-title{{
    font-size:16px;
    font-weight:700;
    color:{state_color};
    margin-bottom:8px;
    }}

    .summary-text{{
    font-size:14px;
    line-height:1.7;
    color:var(--text);
    }}

    .metrics{{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:12px;
    }}

    .metric-card{{
    border:1px solid var(--border);
    border-radius:10px;
    background:#fafbfc;
    padding:14px;
    }}

    .metric-label{{
    font-family:'DM Mono',monospace;
    font-size:10px;
    color:var(--muted);
    text-transform:uppercase;
    }}

    .metric-value{{
    font-size:22px;
    font-weight:700;
    margin-top:8px;
    }}

    .two{{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:14px;
    }}

    .detail-row{{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 0;
    border-bottom:1px solid var(--border);
    }}

    .detail-row:last-child{{border-bottom:none}}

    .detail-name{{font-size:14px}}

    .detail-value{{
    font-family:'DM Mono',monospace;
    font-size:14px;
    font-weight:700;
    color:{score_color};
    }}

    .bar{{
    display:grid;
    grid-template-columns:1fr 2fr 1fr;
    height:28px;
    border-radius:6px;
    overflow:hidden;
    border:1px solid var(--border);
    margin-top:12px;
    }}

    .zone{{
    display:flex;
    align-items:center;
    justify-content:center;
    font-family:'DM Mono',monospace;
    font-size:9px;
    font-weight:700;
    text-align:center;
    }}

    .bad{{background:#fee2e2;color:#dc2626}}
    .neutral{{background:#f3f4f6;color:#6b7280}}
    .good{{background:#dcfce7;color:#16a34a}}

    .cursor-wrap{{
    position:relative;
    height:12px;
    margin-top:4px;
    }}

    .cursor{{
    position:absolute;
    left:{cursor_left}%;
    transform:translateX(-50%);
    width:12px;
    height:12px;
    border-radius:50%;
    border:2px solid #fff;
    background:{state_color};
    box-shadow:0 1px 4px rgba(0,0,0,.25);
    }}

    ul{{
    padding-left:20px;
    line-height:1.8;
    }}

    @media (max-width:900px){{
    .header-content{{grid-template-columns:1fr;gap:8px;padding:10px 0;height:auto}}
    .app-header{{height:auto;padding:12px 18px}}
    .top,.metrics,.two{{grid-template-columns:1fr}}
    }}
    </style>
    </head>
    <body>
    <header class="app-header">
    <div class="header-content">
        <div class="logo-space">
        <img src="Stellantis.png" alt="Stellantis">
        </div>
        <h1 class="app-title">CARE</h1>
        <div class="header-right">
        <div class="hdr-pills">
            <div class="hdr-pill">
            <span class="hdr-pill-dot"></span>
            <span>Caméra thermique</span>
            </div>
            <div class="hdr-pill">
            <span class="hdr-pill-icon">🌡️</span>
            <span class="hdr-pill-bold">{result.score:+.2f}</span>
            <span class="hdr-pill-sep">·</span>
            <span>{result.verdict}</span>
            </div>
        </div>
        </div>
    </div>
    </header>

    <div class="wrap">
    <div class="panel hero">
        <img src="{Path(annotated_path).name}" alt="Analyse thermique">
    </div>

    <div class="panel">
        <div class="title">Fusion et résultat</div>
        <div class="sub">Décision thermique globale sur l’image analysée</div>

        <div class="top">
        <div class="big">
            <div class="big-label">Verdict</div>
            <div class="big-value">{result.verdict}</div>
            <div class="big-meta">Direction : {result.direction}<br>Action : {result.action}</div>
        </div>
        <div class="mini">
            <div class="big-label">Score global</div>
            <div class="mini-value">{result.score:+.2f}</div>
        </div>
        <div class="mini">
            <div class="big-label">Confiance</div>
            <div class="mini-value">{result.confidence:.2f}</div>
        </div>
        </div>

        <div class="summary-box">
        <div class="summary-title">{summary_title}</div>
        <div class="summary-text">{summary_text}</div>
        </div>

        <div class="bar">
        <div class="zone bad">INCONFORT</div>
        <div class="zone neutral">NEUTRE</div>
        <div class="zone good">CONFORT</div>
        </div>
        <div class="cursor-wrap"><div class="cursor"></div></div>
    </div>

    <div class="panel">
        <div class="title">Températures essentielles</div>
        <div class="sub">Zones les plus utiles pour la décision thermique</div>
        <div class="metrics">
        {temp_cards}
        </div>
    </div>

    <div class="panel">
        <div class="title">Détail du score</div>
        <div class="sub">Contribution des critères principaux</div>
        <div class="two">
        <div>{score_rows_left}</div>
        <div>{score_rows_right}</div>
        </div>
    </div>

    <div class="panel">
        <div class="title">Interprétation</div>
        <div class="sub">Facteurs détectés et justification technique</div>
        <ul>{notes_html}</ul>
    </div>
    </div>
    </body>
    </html>
    """
        output_html = os.path.abspath(output_html)
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)
        return output_html

        # --------------------------------------------------------
        # 14) Pipeline complet
        # --------------------------------------------------------
    def analyze_image(
            self,
            image_path: str,
            output_path: Optional[str] = None,
            html_path: Optional[str] = None,
            open_browser: bool = True
        ) -> ComfortResult:
            original = self.load_image(image_path)

            screen = self.auto_crop_thermal_screen(original)
            thermal_zone = self.remove_hikmicro_ui_margins(screen)
            thermal = self.colormap_to_temperature(thermal_zone)

            bbox = self.detect_head_bbox_from_thermal(thermal)
            if bbox is None:
                h, w = thermal.shape[:2]
                bbox = (
                    int(w * 0.20),
                    int(h * 0.08),
                    int(w * 0.80),
                    int(h * 0.92),
                )

            x1, y1, x2, y2 = bbox
            face_crop = thermal_zone[y1:y2, x1:x2].copy()

            landmarks_local = self.detect_face_landmarks_hybrid(face_crop)

            if landmarks_local is not None:
                landmarks_global = landmarks_local.copy()
                landmarks_global[:, 0] += x1
                landmarks_global[:, 1] += y1
                rois = self.build_face_rois_from_landmarks(landmarks_global, thermal_zone.shape)
                detection_mode = "landmarks"
            else:
                rois = self.build_face_rois_from_bbox(bbox)
                detection_mode = "bbox"

            stats = self.extract_zone_stats(thermal, rois)
            result = self.classify(stats)

            if detection_mode == "bbox":
                result.notes.append("Détection visage en mode géométrique de secours.")
            else:
                result.notes.append("Détection visage affinée par landmarks.")

            vis = self.draw_result(thermal_zone, rois, stats, result)

            if output_path is None:
                base, _ = os.path.splitext(image_path)
                output_path = f"{base}_thermal_analysis.jpg"

            output_path = os.path.abspath(output_path)
            cv2.imwrite(output_path, vis)

            html_report = self.generate_html_report(
                image_path=image_path,
                annotated_path=output_path,
                result=result,
                output_html=html_path
            )

            print("\n=== ANALYSE THERMIQUE ===")
            print(f"Verdict      : {result.verdict}")
            print(f"Direction    : {result.direction}")
            print(f"Score        : {result.score}")
            print(f"Confiance    : {result.confidence}")
            print(f"Action       : {result.action}")
            print(f"Mode visage  : {detection_mode}")
            print(f"Joue gauche  : {result.temperatures['joue_gauche']:.2f} °C")
            print(f"Joue droite  : {result.temperatures['joue_droite']:.2f} °C")
            print(f"Nez          : {result.temperatures['nez']:.2f} °C")
            print(f"Front        : {result.temperatures['front']:.2f} °C")
            print(f"Rapport HTML : {html_report}")

            if open_browser:
                webbrowser.open(f"file:///{html_report.replace(os.sep, '/')}")

            return result
    

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Analyse thermique faciale à partir d'une image")
    parser.add_argument("image", help="Chemin vers l'image thermique")
    parser.add_argument("--temp-min", type=float, default=18.0, help="Température min palette")
    parser.add_argument("--temp-max", type=float, default=36.2, help="Température max palette")
    parser.add_argument("--output", type=str, default=None, help="Chemin image annotée")
    parser.add_argument("--html", type=str, default=None, help="Chemin rapport HTML")
    parser.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir automatiquement le navigateur")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERREUR] Fichier introuvable : {args.image}")
        sys.exit(1)

    analyzer = ThermalImageAnalyzer(temp_min=args.temp_min, temp_max=args.temp_max)
    result = analyzer.analyze_image(
        image_path=args.image,
        output_path=args.output,
        html_path=args.html,
        open_browser=not args.no_browser
    )

    if result.verdict == "CONFORT":
        sys.exit(0)
    elif result.verdict == "INCONFORT":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()