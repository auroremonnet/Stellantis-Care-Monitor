from __future__ import annotations

import os
import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import mediapipe as mp
    HAS_MP = True
except Exception:
    HAS_MP = False
    mp_face_mesh = None

# ==============================
# CONFIG
# ==============================

ROI_INDICES = {
    "forehead": [10, 67, 109, 108, 151, 337, 338, 297, 299, 69],
    "nose": [1, 2, 4, 5, 6, 19, 94, 195, 197],
    "chin": [152, 148, 149, 150, 169, 170, 171, 175, 176, 377, 378, 379, 394, 395, 396, 400],
    "cheek_left": [50, 101, 116, 117, 118, 119, 120, 123, 187, 205, 206, 207],
    "cheek_right": [280, 330, 345, 346, 347, 348, 349, 352, 411, 425, 426, 427],
    # Inner canthi are intentionally small, point-centered ROIs built later around landmarks.
}

INNER_CANTHUS_LEFT = 133
INNER_CANTHUS_RIGHT = 362
NOSE_TIP = 1


@dataclass
class ZoneStats:
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    tmin: float = 0.0
    tmax: float = 0.0
    p10: float = 0.0
    p90: float = 0.0
    n: int = 0


@dataclass
class ComfortResult:
    label: str
    direction: str
    score: float
    confidence: float
    action: str
    temperatures: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class ThermalFaceAnalyzer:
    def __init__(self, temp_min: float = 18.0, temp_max: float = 36.2):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.face_mesh = None
        if HAS_MP:
            
            self.face_mesh_engine = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    # ------------------------------
    # I/O + thermal preparation
    # ------------------------------
    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def thermal_to_celsius(self, img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        score_h = np.zeros_like(h, dtype=np.float32)
        mask_cold = h > 80
        score_h[mask_cold] = np.clip((130.0 - h[mask_cold]) / 130.0, 0.0, 0.35)

        mask_mid = (h > 35) & (h <= 80)
        score_h[mask_mid] = 0.3 + (80.0 - h[mask_mid]) / 80.0 * 0.25

        mask_warm = (h > 12) & (h <= 35)
        score_h[mask_warm] = 0.55 + (35.0 - h[mask_warm]) / 35.0 * 0.20

        mask_hot = (h <= 12) | (h >= 170)
        score_h[mask_hot] = 0.90

        mask_white = (v > 220) & (s < 60)
        score_h[mask_white] = 1.0

        thermal_index = 0.7 * score_h + 0.3 * (v / 255.0)
        celsius = self.temp_min + thermal_index * (self.temp_max - self.temp_min)
        return celsius.astype(np.float32)

    # ------------------------------
    # Face / ROI detection
    # ------------------------------
    def detect_landmarks(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.face_mesh is None:
            return None
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = self.face_mesh.process(rgb)
        if not out.multi_face_landmarks:
            return None
        lms = out.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    def build_rois(self, img_shape: Tuple[int, int], landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = img_shape[:2]
        rois: Dict[str, np.ndarray] = {}
        for name, idxs in ROI_INDICES.items():
            pts = landmarks[idxs].astype(np.int32)
            rois[name] = cv2.convexHull(pts)

        inter_eye = float(np.linalg.norm(landmarks[33] - landmarks[263])) + 1e-6
        radius = max(3, int(inter_eye * 0.04))

        for name, idx in [("inner_canthus_left", INNER_CANTHUS_LEFT), ("inner_canthus_right", INNER_CANTHUS_RIGHT), ("nose_tip", NOSE_TIP)]:
            x, y = landmarks[idx].astype(int)
            mask_poly = cv2.ellipse2Poly((int(x), int(y)), (radius, radius), 0, 0, 360, 30)
            mask_poly[:, 0] = np.clip(mask_poly[:, 0], 0, w - 1)
            mask_poly[:, 1] = np.clip(mask_poly[:, 1], 0, h - 1)
            rois[name] = mask_poly.astype(np.int32)

        return rois

    # ------------------------------
    # Robust temperature extraction
    # ------------------------------
    def _polygon_mask(self, shape: Tuple[int, int], poly: np.ndarray) -> np.ndarray:
        m = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(m, poly, 255)
        return m

    def _robust_pixels(self, thermal: np.ndarray, mask: np.ndarray, floor: float = 24.0) -> np.ndarray:
        vals = thermal[mask == 255]
        vals = vals[np.isfinite(vals)]
        vals = vals[vals >= floor]
        if len(vals) == 0:
            return vals
        p05, p95 = np.percentile(vals, [5, 95])
        vals = vals[(vals >= p05) & (vals <= p95)]
        return vals

    def extract_zone_stats(self, thermal: np.ndarray, rois: Dict[str, np.ndarray]) -> Dict[str, ZoneStats]:
        stats: Dict[str, ZoneStats] = {}
        for name, poly in rois.items():
            mask = self._polygon_mask(thermal.shape, poly)
            vals = self._robust_pixels(thermal, mask)
            if len(vals) == 0:
                stats[name] = ZoneStats()
                continue
            stats[name] = ZoneStats(
                mean=float(np.mean(vals)),
                median=float(np.median(vals)),
                std=float(np.std(vals)),
                tmin=float(np.min(vals)),
                tmax=float(np.max(vals)),
                p10=float(np.percentile(vals, 10)),
                p90=float(np.percentile(vals, 90)),
                n=int(len(vals)),
            )
        return stats

    # ------------------------------
    # Features based on literature
    # ------------------------------
    def build_features(self, zs: Dict[str, ZoneStats]) -> Dict[str, float]:
        def t(name: str, fallback: float = np.nan) -> float:
            return zs[name].mean if name in zs and zs[name].n > 0 else fallback

        forehead = t("forehead")
        nose = t("nose")
        cheek_l = t("cheek_left")
        cheek_r = t("cheek_right")
        cheeks = np.nanmean([cheek_l, cheek_r])
        chin = t("chin")
        canthus_l = t("inner_canthus_left")
        canthus_r = t("inner_canthus_right")
        canthus = np.nanmean([canthus_l, canthus_r])
        nose_tip = t("nose_tip", nose)

        feats = {
            "t_forehead": forehead,
            "t_nose": nose,
            "t_nose_tip": nose_tip,
            "t_cheeks": cheeks,
            "t_chin": chin,
            "t_canthus": canthus,
            "g_forehead_nose": forehead - nose,
            "g_forehead_cheeks": forehead - cheeks,
            "g_canthus_nose": canthus - nose,
            "g_canthus_nose_tip": canthus - nose_tip,
            "asym_cheeks": abs(cheek_l - cheek_r),
            "face_uniformity": np.nanstd([forehead, nose, cheeks, chin]),
            "mean_face": np.nanmean([forehead, nose, cheeks, chin]),
        }
        return {k: float(v) for k, v in feats.items() if np.isfinite(v)}

    # ------------------------------
    # Decision logic
    # ------------------------------
    def classify(self, feats: Dict[str, float]) -> ComfortResult:
        score = 0.0
        notes: List[str] = []
        conf = []

        g_fn = feats.get("g_forehead_nose", 0.0)
        g_cn = feats.get("g_canthus_nose", 0.0)
        t_nose = feats.get("t_nose", 32.0)
        t_fore = feats.get("t_forehead", 33.5)
        t_canthus = feats.get("t_canthus", 34.0)
        asym = feats.get("asym_cheeks", 0.0)
        uniform = feats.get("face_uniformity", 0.0)

        # Strong cold evidence: nose cooler than forehead / inner canthus.
        if g_fn >= 2.5:
            score -= min(4.0, 1.2 + 0.9 * (g_fn - 2.5))
            notes.append(f"Gradient front→nez élevé ({g_fn:.2f}°C): compatible avec refroidissement périphérique.")
            conf.append(0.9)
        elif g_fn >= 1.3:
            score -= 1.4
            notes.append(f"Gradient front→nez modéré ({g_fn:.2f}°C): tendance au froid.")
            conf.append(0.75)
        else:
            conf.append(0.65)

        if g_cn >= 3.0:
            score -= 2.0
            notes.append(f"Gradient canthus→nez élevé ({g_cn:.2f}°C): nez nettement plus froid que le coin interne de l’œil.")
            conf.append(0.85)
        elif g_cn <= 1.0:
            score += 0.8
            notes.append(f"Gradient canthus→nez faible ({g_cn:.2f}°C): visage plus uniformément chaud.")
            conf.append(0.65)

        if t_nose < 30.0:
            score -= 2.0
            notes.append(f"Nez froid ({t_nose:.2f}°C).")
            conf.append(0.85)
        elif t_nose > 34.0:
            score += 1.2
            notes.append(f"Nez très chaud ({t_nose:.2f}°C).")
            conf.append(0.7)

        if t_fore > 35.8:
            score += 1.5
            notes.append(f"Front chaud ({t_fore:.2f}°C): charge thermique globale possible.")
            conf.append(0.7)
        elif t_fore < 32.0:
            score -= 0.8
            notes.append(f"Front relativement frais ({t_fore:.2f}°C).")
            conf.append(0.7)

        if t_canthus > 36.0 and t_nose > 33.5:
            score += 1.2
            notes.append("Coin interne de l’œil et nez élevés: profil compatible avec sensation de chaud.")
            conf.append(0.65)

        if asym > 1.0:
            notes.append(f"Asymétrie joues ({asym:.2f}°C): possible jet d’air latéral ou exposition non uniforme.")
            conf.append(0.5)

        if uniform < 0.8:
            score += 0.4
            notes.append("Distribution thermique faciale homogène.")
            conf.append(0.6)

        confidence = float(np.mean(conf)) if conf else 0.5
        score = float(np.clip(score, -5.0, 5.0))

        if score <= -1.8:
            label = "INCONFORT"
            direction = "froid"
            action = "Augmenter légèrement la température cabine ou réduire le flux d’air froid direct."
        elif score >= 1.8:
            label = "INCONFORT"
            direction = "chaud"
            action = "Renforcer légèrement le refroidissement ou la ventilation."
        elif abs(score) <= 0.8:
            label = "CONFORT"
            direction = "neutre"
            action = "Maintenir la stratégie HVAC actuelle."
        else:
            label = "NEUTRE"
            direction = "froid" if score < 0 else "chaud"
            action = "Ajustement doux recommandé après confirmation temporelle."

        return ComfortResult(
            label=label,
            direction=direction,
            score=round(score, 2),
            confidence=round(confidence, 2),
            action=action,
            temperatures={k: round(v.mean, 2) for k, v in zs.items()} if False else {},
            features={k: round(v, 2) for k, v in feats.items()},
            notes=notes,
        )

    # ------------------------------
    # End-to-end one frame
    # ------------------------------
    def analyze_frame(self, img_bgr: np.ndarray):
        thermal = self.thermal_to_celsius(img_bgr)
        landmarks = self.detect_landmarks(img_bgr)
        if landmarks is None:
            raise RuntimeError("Visage non détecté sur l'image thermique.")
        rois = self.build_rois(img_bgr.shape, landmarks)
        stats = self.extract_zone_stats(thermal, rois)
        feats = self.build_features(stats)
        result = self.classify(feats)
        result.temperatures = {k: round(v.mean, 2) for k, v in stats.items() if v.n > 0}
        return thermal, landmarks, rois, stats, result

    def draw(self, img_bgr: np.ndarray, rois: Dict[str, np.ndarray], result: ComfortResult) -> np.ndarray:
        vis = img_bgr.copy()
        color_map = {
            "forehead": (255, 200, 80),
            "nose": (80, 180, 255),
            "chin": (220, 140, 255),
            "cheek_left": (100, 255, 100),
            "cheek_right": (100, 255, 100),
            "inner_canthus_left": (255, 255, 255),
            "inner_canthus_right": (255, 255, 255),
            "nose_tip": (0, 220, 255),
        }
        for name, poly in rois.items():
            c = color_map.get(name, (200, 200, 200))
            overlay = vis.copy()
            cv2.fillConvexPoly(overlay, poly, c)
            vis = cv2.addWeighted(overlay, 0.22, vis, 0.78, 0)
            cv2.polylines(vis, [poly], True, c, 1, cv2.LINE_AA)

        h, w = vis.shape[:2]
        band = (40, 180, 40) if result.label == "CONFORT" else ((30, 30, 230) if result.label == "INCONFORT" else (180, 180, 40))
        cv2.rectangle(vis, (0, 0), (w, 58), (20, 20, 20), -1)
        cv2.putText(vis, f"{result.label} | score {result.score:+.2f} | conf {result.confidence:.2f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, band, 2, cv2.LINE_AA)
        cv2.putText(vis, f"Direction: {result.direction} | {result.action}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        return vis


def demo(image_path: str, temp_min: float = 18.0, temp_max: float = 36.2, save_path: Optional[str] = None):
    analyzer = ThermalFaceAnalyzer(temp_min=temp_min, temp_max=temp_max)
    img = analyzer.load_image(image_path)
    thermal, landmarks, rois, stats, result = analyzer.analyze_frame(img)
    vis = analyzer.draw(img, rois, result)

    if save_path is None:
        root, ext = os.path.splitext(image_path)
        save_path = root + "_v2_analysis.jpg"
    cv2.imwrite(save_path, vis)

    print("\n=== THERMAL COMFORT RESULT ===")
    print(result.label, result.direction, result.score, result.confidence)
    print("Action:", result.action)
    print("Temperatures:")
    for k, v in result.temperatures.items():
        print(f"  - {k}: {v:.2f}°C")
    print("Features:")
    for k, v in result.features.items():
        print(f"  - {k}: {v:.2f}")
    print("Notes:")
    for n in result.notes:
        print("  -", n)
    print("Saved:", save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thermal comfort facial analyzer V2")
    parser.add_argument("image", help="Chemin vers l'image thermique")
    parser.add_argument("--temp-min", type=float, default=18.0, help="Température min palette")
    parser.add_argument("--temp-max", type=float, default=36.2, help="Température max palette")
    parser.add_argument("--output", type=str, default=None, help="Chemin de sortie image annotée")
    args = parser.parse_args()

    demo(
        image_path=args.image,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        save_path=args.output,
    )