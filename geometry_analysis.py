import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# ==========================================
# INDICES MEDIAPIPE (CONSTANTES)
# ==========================================
# Contour des yeux
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Milieu des sourcils
LEFT_BROW_MID = 105
RIGHT_BROW_MID = 334

# Pupilles (Iris landmarks)
LEFT_PUPIL_IDX = 468 
RIGHT_PUPIL_IDX = 473

# Bouche (Coins et Lèvres)
MOUTH_CORNER_L = 61
MOUTH_CORNER_R = 291
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14

class FaceGeometryAnalyzer:
    """
    Moteur d'analyse géométrique faciale V10 (Professionnel).
    Spécialisé dans la distinction subtile entre Sourire (Joie) et Grimace (Inconfort).
    """

    def __init__(self):
        # Initialisation robuste de MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True, # Indispensable pour les pupilles
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Valeurs de référence (Moyennes humaines par défaut)
        # Ces valeurs seront affinées lors de la calibration.
        self.baseline = {
            "ear": 0.30,         # Eye Aspect Ratio
            "mar": 0.05,         # Mouth Aspect Ratio
            "brow_dist": 0.10,   # Distance Sourcil-Oeil
            "smile_lift": 0.0,   # Verticalité du sourire
            "mouth_width": 0.40  # Largeur horizontale de la bouche
        }
        
        self.is_calibrated = False
        self.inter_ocular_distance = 1.0 # Facteur de normalisation (Zoom)

    def get_landmarks(self, img_rgb):
        """Récupère les points 3D du visage."""
        h, w, _ = img_rgb.shape
        results = self.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        mesh = results.multi_face_landmarks[0]
        # Conversion optimisée en numpy array
        points = np.array([(p.x * w, p.y * h, p.z) for p in mesh.landmark], dtype=np.float32)
        return points

    # ==========================================
    # MÉTRIQUES SCIENTIFIQUES (Normalisées)
    # ==========================================

    def _get_inter_ocular(self, lms):
        """Distance entre les deux pupilles (référence de taille)."""
        return dist.euclidean(lms[LEFT_PUPIL_IDX][:2], lms[RIGHT_PUPIL_IDX][:2]) + 1e-6

    def _calculate_ear(self, points):
        """Eye Aspect Ratio : Ouverture de l'oeil."""
        A = dist.euclidean(points[1][:2], points[5][:2])
        B = dist.euclidean(points[2][:2], points[4][:2])
        C = dist.euclidean(points[0][:2], points[3][:2])
        return (A + B) / (2.0 * (C + 1e-6))

    def _calculate_mar(self, lms):
        """Mouth Aspect Ratio : Ouverture verticale de la bouche."""
        V = dist.euclidean(lms[UPPER_LIP_TOP][:2], lms[LOWER_LIP_BOTTOM][:2])
        H = dist.euclidean(lms[MOUTH_CORNER_L][:2], lms[MOUTH_CORNER_R][:2])
        return V / (H + 1e-6)

    def _calculate_smile_lift(self, lms):
        """
        Indice de Sourire Vertical.
        Mesure si les coins de la bouche sont plus hauts que le centre de la lèvre.
        Positif = Sourire / Négatif = Triste ou Neutre.
        """
        avg_corner_y = (lms[MOUTH_CORNER_L][1] + lms[MOUTH_CORNER_R][1]) / 2.0
        center_lip_y = lms[UPPER_LIP_TOP][1]
        
        # En image, Y augmente vers le bas, donc (Centre - Coins) est positif si les coins montent.
        return (center_lip_y - avg_corner_y) / self.inter_ocular_distance

    def _calculate_mouth_width_norm(self, lms):
        """Largeur horizontale de la bouche normalisée."""
        width_px = dist.euclidean(lms[MOUTH_CORNER_L][:2], lms[MOUTH_CORNER_R][:2])
        return width_px / self.inter_ocular_distance

    def _calculate_brow_dist(self, lms):
        """Distance Sourcil - Pupille normalisée."""
        dist_l = abs(lms[LEFT_PUPIL_IDX][1] - lms[LEFT_BROW_MID][1])
        dist_r = abs(lms[RIGHT_PUPIL_IDX][1] - lms[RIGHT_BROW_MID][1])
        return ((dist_l + dist_r) / 2.0) / self.inter_ocular_distance

    # ==========================================
    # LOGIQUE D'ANALYSE
    # ==========================================

    def analyze(self, img_rgb):
        """Pipeline principal d'analyse."""
        lms = self.get_landmarks(img_rgb)
        if lms is None: return None

        # Mise à jour du facteur de zoom (distance visage-caméra)
        self.inter_ocular_distance = self._get_inter_ocular(lms)

        # Calcul des métriques brutes
        metrics = {
            "ear": (self._calculate_ear(lms[LEFT_EYE]) + self._calculate_ear(lms[RIGHT_EYE])) / 2.0,
            "mar": self._calculate_mar(lms),
            "smile_lift": self._calculate_smile_lift(lms),
            "mouth_width": self._calculate_mouth_width_norm(lms),
            "brow_dist": self._calculate_brow_dist(lms),
            "landmarks": lms 
        }
        
        # Interprétation sémantique (Texte)
        analysis = self._interpret_metrics(metrics)
        
        return {**metrics, **analysis}

    def calibrate(self, buffer):
        """Définit le visage 'Neutre' de l'utilisateur."""
        if not buffer: return False
        
        print(f"[CALIB] Calibration sur {len(buffer)} frames...")
        self.baseline["ear"] = np.mean([m["ear"] for m in buffer])
        self.baseline["mar"] = np.mean([m["mar"] for m in buffer])
        self.baseline["brow_dist"] = np.mean([m["brow_dist"] for m in buffer])
        self.baseline["smile_lift"] = np.mean([m["smile_lift"] for m in buffer])
        self.baseline["mouth_width"] = np.mean([m["mouth_width"] for m in buffer])
        
        self.is_calibrated = True
        return True

    def _interpret_metrics(self, m):
        """
        Le cœur de l'intelligence : Transforme les chiffres en états humains.
        """
        res = {"txt_eyes": "Neutre", "txt_mouth": "Fermee", "txt_brows": "Stable"}
        if not self.is_calibrated: return res

        # --- 1. ANALYSE BOUCHE (Sourire vs Grimace) ---
        
        # Condition A : Les coins montent vraiment (Lift)
        # Seuil strict pour éviter les faux positifs
        is_lifting = m["smile_lift"] > (self.baseline["smile_lift"] + 0.02)
        
        # Condition B : La bouche s'élargit horizontalement
        is_widening = m["mouth_width"] > (self.baseline["mouth_width"] * 1.12) # +12% largeur

        # Condition C : La bouche est ouverte verticalement
        is_vert_open = m["mar"] > (self.baseline["mar"] + 0.10)

        # ARBRE DE DÉCISION
        if is_lifting:
            # Si ça monte -> C'est un SOURIRE (indépendamment de la largeur)
            if is_vert_open:
                res["txt_mouth"] = "Grand Sourire"
            else:
                res["txt_mouth"] = "Sourire leger"
        
        elif is_widening:
            # Si ça s'élargit MAIS que ça ne monte pas -> C'est une GRIMACE
            # C'est ici qu'on corrige ton problème
            res["txt_mouth"] = "Grimace / Tension"
            
        elif m["mar"] > (self.baseline["mar"] + 0.25):
            res["txt_mouth"] = "Baillement"
            
        elif is_vert_open:
            res["txt_mouth"] = "Parle"

        # --- 2. ANALYSE YEUX ---
        # Si les yeux se ferment (Fatigue ou Rire)
        if m["ear"] < (self.baseline["ear"] * 0.80):
            res["txt_eyes"] = "Plisses"
        # Si les yeux s'ouvrent grand (Surprise/Alerte)
        elif m["ear"] > (self.baseline["ear"] * 1.15):
            res["txt_eyes"] = "Ecarquilles"

        # --- 3. ANALYSE SOURCILS ---
        if m["brow_dist"] < (self.baseline["brow_dist"] * 0.92):
            res["txt_brows"] = "Fronces"
        elif m["brow_dist"] > (self.baseline["brow_dist"] * 1.08):
            res["txt_brows"] = "Releves"

        return res