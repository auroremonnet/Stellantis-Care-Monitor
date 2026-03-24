import cv2
import numpy as np
import os

# --- FIX CRITIQUE : Importation explicite ---
# Au lieu de "import mediapipe as mp", on va chercher directement le sous-module
# Cela contourne le bug "module has no attribute solutions"
from mediapipe.python.solutions import face_mesh

# Désactiver les logs inutiles de TensorFlow/MediaPipe
os.environ['GLOG_minloglevel'] = '2' 

class FaceRegionDetector:
    """
    Détecteur basé sur MediaPipe avec correctif d'importation.
    """
    def __init__(self, model_path=None):
        print("[FaceRegionDetector] Chargement du moteur MediaPipe (Mode Robuste)...")
        
        # On utilise l'objet importé explicitement
        self.face_mesh_engine = face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[FaceRegionDetector] Moteur de précision chargé.")

    def detect(self, img_rgb):
        """
        Méthode standard appelée par main.py.
        Retourne la BBOX [x1, y1, x2, y2] ou None.
        """
        h, w = img_rgb.shape[:2]
        
        # Utilisation du moteur chargé
        results = self.face_mesh_engine.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        # Récupération des points pour calculer la boîte englobante
        landmarks = results.multi_face_landmarks[0].landmark
        x_all = [int(pt.x * w) for pt in landmarks]
        y_all = [int(pt.y * h) for pt in landmarks]
        
        # Calcul de la Bbox avec une petite marge de sécurité
        x1, y1 = min(x_all), min(y_all)
        x2, y2 = max(x_all), max(y_all)
        
        # Retourne format numpy array comme YOLO le ferait
        return np.array([x1, y1, x2, y2])

    def crop_regions(self, img_rgb, bbox=None):
        """
        Découpe les régions (yeux, bouche) pour l'analyse fine.
        Si bbox est fournie, on peut optimiser, mais MediaPipe recalcule souvent tout.
        """
        h, w = img_rgb.shape[:2]
        results = self.face_mesh_engine.process(img_rgb)

        if not results.multi_face_landmarks:
            return {}, None

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Indices spécifiques (Yeux, Bouche, Sourcils)
        indices = {
            "brows": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300], 
            "eyes": [33, 133, 157, 158, 159, 160, 161, 246, 362, 263, 384, 385, 386, 387, 388, 466],
            "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        }

        crops = {}
        coords = {}
        
        for region_name, idx_list in indices.items():
            pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in idx_list])
            
            # Limites avec marge
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            pad_x = int((x_max - x_min) * 0.3)
            pad_y = int((y_max - y_min) * 0.3)
            
            c_x1 = max(0, x_min - pad_x)
            c_y1 = max(0, y_min - pad_y)
            c_x2 = min(w, x_max + pad_x)
            c_y2 = min(h, y_max + pad_y)
            
            # Extraction
            crop = img_rgb[c_y1:c_y2, c_x1:c_x2]
            
            # Conversion RGB -> BGR pour affichage OpenCV correct
            crops[region_name] = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            coords[region_name] = (c_x1, c_y1, c_x2, c_y2)

        return {"regions_img": crops, "coords": coords}