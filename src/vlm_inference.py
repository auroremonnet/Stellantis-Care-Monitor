import cv2
import numpy as np
import torch
import re
from PIL import Image
from typing import Dict, Any
from transformers import AutoProcessor, AutoModelForImageTextToText

class VLMAnalyzer:
    """
    Expert en analyse FACS (Facial Action Coding System) pour le confort thermique.
    Analyse les micro-expressions musculaires.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cpu"
        print(f"[VLMAnalyzer] Chargement du Moteur FACS sur {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float32
        ).to(self.device)
        print("[VLMAnalyzer] Expert prêt.")

    def _stack_images(self, crops: Dict[str, np.ndarray]) -> Image.Image:
        """Fusionne les 3 zones (Haut/Milieu/Bas) en une bande verticale."""
        images = []
        for part in ["brows", "eyes", "mouth"]:
            img = crops.get(part)
            if img is None or img.size == 0:
                img = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Standardisation à 384px de large pour l'IA
            h, w = img.shape[:2]
            new_w = 384
            new_h = int(h * (new_w / w))
            img_resized = cv2.resize(img, (new_w, new_h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)

        # Collage vertical
        total_h = sum(img.shape[0] for img in images)
        combined = np.zeros((total_h, 384, 3), dtype=np.uint8)
        
        y_offset = 0
        for img in images:
            h = img.shape[0]
            combined[y_offset:y_offset+h, :] = img
            y_offset += h
            
        return Image.fromarray(combined)

    def _parse_combined_response(self, text: str) -> Dict[str, Any]:
        results = {}
        # Nettoyage technique
        text = text.replace("*", "").replace("#", "")
        text_lower = text.lower()
        
        # Mapping des zones
        keys = ["sourcils", "yeux", "bouche"]
        
        for part_en, part_fr in zip(["brows", "eyes", "mouth"], keys):
            state = "neutre"
            reason = "Analyse indisponible"
            
            try:
                if part_fr in text_lower:
                    # Isoler la section concernée
                    segment = text_lower.split(part_fr)[1]
                    for k in keys: # Couper si on rencontre le mot clé suivant
                        if k != part_fr and k in segment:
                            segment = segment.split(k)[0]
                    
                    # --- LOGIQUE DE DÉCISION RENFORCÉE ---
                    # Inconfort si : Tension, Froncement, Pincement, Stress
                    if any(x in segment for x in ["inconfort", "tendu", "crisp", "fronc", "pinc", "serr"]):
                        state = "inconfort"
                    # Confort si : Détendu, Sourire, Relaché, Calme, Plissé (yeux)
                    elif any(x in segment for x in ["confort", "sour", "détend", "relach", "paisible", "pliss"]):
                        state = "confort"
                    
                    # Extraction de la "Raison" (Phrase après "car")
                    # On utilise le texte original (avec majuscules) pour la lisibilité
                    start_idx = text.lower().find(part_fr) + len(part_fr)
                    raw_segment = text[start_idx:].split("\n")[0] # Prend la ligne
                    
                    # Nettoyage : On enlève "Etat:...", on cherche le texte descriptif
                    clean_reason = raw_segment
                    if "car" in clean_reason.lower():
                        clean_reason = clean_reason.split("car", 1)[1]
                    
                    # Nettoyage final ponctuation
                    clean_reason = clean_reason.replace(":", "").replace("-", "").strip()
                    clean_reason = clean_reason.split(".")[0] # Garder la 1ère phrase
                    
                    reason = clean_reason.capitalize()
                    if len(reason) < 5: reason = "Expression détectée."

            except Exception as e:
                print(f"Parsing error ({part_fr}): {e}")
            
            results[part_en] = {"etat": state, "raison": reason}
            
        return results

    def analyze(self, crops: Dict[str, np.ndarray]) -> Dict[str, Any]:
        pil_img = self._stack_images(crops)
        
        # --- PROMPT CHIRURGICAL (Le secret de la perfection) ---
        prompt = (
            "Analyse cette image composée de 3 parties verticales pour un diagnostic de confort thermique.\n\n"
            "1. HAUT (Sourcils): Regarde la glabelle (entre les sourcils). "
            "Rides verticales ou sourcils bas = INCONFORT (Tension). "
            "Sourcils plats et zone lisse = CONFORT (Détente).\n"
            "2. MILIEU (Yeux): Regarde les coins externes. "
            "Plissement 'pattes d'oie' (Sourire Duchenne) = CONFORT. "
            "Yeux grands ouverts/fixes ou paupières lourdes = INCONFORT.\n"
            "3. BAS (Bouche): Regarde les commissures (coins). "
            "Relevés vers le haut = CONFORT. "
            "Horizontaux, serrés ou vers le bas = INCONFORT/NEUTRE.\n\n"
            "Réponds STRICTEMENT selon ce modèle :\n"
            "Sourcils: [Etat] car [Description anatomique précise].\n"
            "Yeux: [Etat] car [Description anatomique précise].\n"
            "Bouche: [Etat] car [Description anatomique précise]."
        )

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_in = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_in], images=[pil_img], return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Temperature 0.1 pour être factuel et analytique
            gen = self.model.generate(**inputs, max_new_tokens=120, temperature=0.1)
        
        out_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        response = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        
        # Debug dans la console pour vous prouver que l'IA réfléchit
        print("\n--- ANALYSE BRUTE ---")
        print(response)
        print("------------------------\n")
        
        return self._parse_combined_response(response)