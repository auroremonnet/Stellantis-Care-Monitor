import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


class EmotionCNNAnalyzer:
    """
    Analyseur d'émotion basé sur un ResNet18 fine-tuné.
    Charge un modèle entraîné avec train_emotion_cnn.py.
    """

    def __init__(self, model_path="models/emotion_resnet18_affectnet.pt", device=None):
        # Charge checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        # Récupération des classes (ordre identique à ImageFolder)
        self.classes = checkpoint["classes"]

        # Sélection du device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Reconstruction du modèle
        num_classes = len(self.classes)
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

        # Chargement des poids du modèle
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model

        # Transformations d'entrée
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _map_emotion_to_confort(self, emo_label: str):
        """
        Regroupe les émotions en Confort / Neutre / Inconfort
        """
        emo = emo_label.lower()

        if emo in ["happy", "surprise"]:
            return "confort"
        elif emo in ["neutral"]:
            return "neutre"
        else:
            return "inconfort"

    def analyze_emotion(self, face_bgr: np.ndarray):
        """
        Analyse une image de visage (OpenCV BGR).

        Retourne :
          - emo_label : nom de l'émotion dominante
          - emo_score : probabilité associée (float)
          - confort_state : confort / neutre / inconfort
          - scores_dict : {emotion: proba}
        """

        # BGR -> RGB -> PIL
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        # Prétraitement
        x = self.tf(pil_img).unsqueeze(0).to(self.device)

        # Inférence
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Emotion dominante
        idx = int(np.argmax(probs))
        emo_label = self.classes[idx]
        emo_score = float(probs[idx])

        # Conversion en dictionnaire
        scores_dict = {
            cls: float(p) for cls, p in zip(self.classes, probs)
        }

        # Mapping confort
        confort_state = self._map_emotion_to_confort(emo_label)

        return emo_label, emo_score, confort_state, scores_dict
