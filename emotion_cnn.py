import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from collections import deque


class EmotionCNNAnalyzer:
    """
    Analyseur d'émotion ResNet18 — Version améliorée.
    
    Améliorations vs version initiale :
    ─────────────────────────────────────
    1. Lissage temporel sur fenêtre glissante (15 frames)
       → Empêche les flashs d'émotion sur 1 frame
    
    2. Preprocessing renforcé
       → Égalisation CLAHE sur luminance (Y) en espace YCrCb
       → Correction gamma pour les conditions de faible lumière
       → Padding carré avant resize (évite la déformation)
    
    3. Seuil de confiance minimum (0.35)
       → Si le modèle est incertain → retourne "neutral"
       → Évite les labels aberrants sur visage flou/de profil
    
    4. Suppression de la correction CNN forcée dans server.py
       → Le lissage temporel remplace avantageusement cette heuristique
    
    5. Retour enrichi : scores_dict complet pour affichage dans le dashboard
    """

    # Fenêtre de lissage temporel (en frames)
    SMOOTH_WINDOW    = 15
    # Seuil minimum de confiance pour valider une prédiction
    CONFIDENCE_THRESHOLD = 0.35

    def __init__(self, model_path="models/emotion_resnet18_affectnet.pt", device=None):

        # ── Chargement checkpoint ────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location="cpu")
        self.classes = checkpoint["classes"]
        print(f"[EmotionCNN] Classes chargées : {self.classes}")

        # ── Device ───────────────────────────────────────────────────────────
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[EmotionCNN] Device : {self.device}")

        # ── Modèle ───────────────────────────────────────────────────────────
        num_classes = len(self.classes)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

        # ── Transformations (identiques à l'entraînement) ───────────────────
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

        # ── Lissage temporel ─────────────────────────────────────────────────
        # Stocke les vecteurs de probabilités des N dernières frames
        self.prob_history = deque(maxlen=self.SMOOTH_WINDOW)

        # CLAHE pour améliorer le contraste local du visage
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # ────────────────────────────────────────────────────────────────────────
    # PREPROCESSING
    # ────────────────────────────────────────────────────────────────────────

    def _preprocess(self, face_bgr: np.ndarray) -> Image.Image:
        """
        Pipeline de preprocessing robuste :
        1. Crop carré centré (évite la distorsion horizontale)
        2. Correction CLAHE sur la luminance (Y)
        3. Correction gamma légère si image sombre
        """
        h, w = face_bgr.shape[:2]

        # 1. Crop carré centré
        size = min(h, w)
        y0   = (h - size) // 2
        x0   = (w - size) // 2
        face = face_bgr[y0:y0+size, x0:x0+size]

        # 2. CLAHE sur le canal Y (luminance) en espace YCrCb
        ycrcb    = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq     = self.clahe.apply(y)
        face_eq  = cv2.merge([y_eq, cr, cb])
        face_eq  = cv2.cvtColor(face_eq, cv2.COLOR_YCrCb2BGR)

        # 3. Correction gamma si image globalement sombre (mean < 80)
        mean_lum = float(np.mean(y_eq))
        if mean_lum < 80:
            gamma = 1.5
            lut   = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                              for i in range(256)], dtype=np.uint8)
            face_eq = cv2.LUT(face_eq, lut)

        # 4. BGR → RGB → PIL
        face_rgb = cv2.cvtColor(face_eq, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)

    # ────────────────────────────────────────────────────────────────────────
    # MAPPING CONFORT
    # ────────────────────────────────────────────────────────────────────────

    def _map_emotion_to_confort(self, emo_label: str) -> str:
        emo = emo_label.lower()
        if emo in ["happy", "surprise"]:
            return "confort"
        elif emo in ["neutral"]:
            return "neutre"
        else:
            return "inconfort"

    # ────────────────────────────────────────────────────────────────────────
    # ANALYSE PRINCIPALE
    # ────────────────────────────────────────────────────────────────────────

    def analyze_emotion(self, face_bgr: np.ndarray):
        """
        Analyse un visage BGR et retourne l'émotion lissée temporellement.

        Retourne :
          emo_label    : str   — émotion dominante (après lissage)
          emo_score    : float — probabilité lissée
          confort_state: str   — "confort" / "neutre" / "inconfort"
          scores_dict  : dict  — {émotion: proba lissée}
        """
        # Vérification taille minimale
        if face_bgr is None or face_bgr.size == 0:
            return "neutral", 0.0, "neutre", {}

        h, w = face_bgr.shape[:2]
        if h < 32 or w < 32:
            return "neutral", 0.0, "neutre", {}

        # ── Preprocessing ────────────────────────────────────────────────────
        pil_img = self._preprocess(face_bgr)

        # ── Inférence ────────────────────────────────────────────────────────
        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        # ── Lissage temporel ─────────────────────────────────────────────────
        self.prob_history.append(probs.copy())

        # Moyenne sur la fenêtre glissante
        smoothed = np.mean(np.stack(list(self.prob_history), axis=0), axis=0)

        # ── Décision avec seuil de confiance ─────────────────────────────────
        idx       = int(np.argmax(smoothed))
        emo_score = float(smoothed[idx])

        # Si confiance insuffisante → neutral (évite les labels aberrants)
        if emo_score < self.CONFIDENCE_THRESHOLD:
            emo_label = "neutral"
            emo_score = float(smoothed[self.classes.index("neutral")]) \
                        if "neutral" in self.classes else emo_score
        else:
            emo_label = self.classes[idx]

        # ── Résultats ────────────────────────────────────────────────────────
        scores_dict   = {cls: float(p) for cls, p in zip(self.classes, smoothed)}
        confort_state = self._map_emotion_to_confort(emo_label)

        return emo_label, emo_score, confort_state, scores_dict

    def reset_history(self):
        """Remet à zéro l'historique de lissage (utile après calibration)."""
        self.prob_history.clear()