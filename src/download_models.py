import os
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration de la sécurité PyTorch
torch.serialization.add_safe_globals([DetectionModel])

# URLs des modèles
MODELS = {
    # YOLOv8
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    
    # YOLOv3
    'yolov3-tiny.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3-tiny.pt',
    'yolov3.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3.pt',
    'yolov3-spp.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3-spp.pt',
    
    # Modèles spécialisés
    'yolov8n-face.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt',
    'yolov8n-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt',
    'yolov8n-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt'
}

def verify_model(model_path: Path) -> bool:
    """
    Vérifie si le modèle peut être chargé correctement avec gestion de la sécurité PyTorch 2.2.0
    """
    try:
        # Première tentative avec weights_only=True (par défaut)
        try:
            model = YOLO(str(model_path))
        except Exception as e1:
            logging.warning(f"Tentative de chargement alternatif pour {model_path}")
            # Deuxième tentative avec weights_only=False
            model = torch.load(str(model_path), weights_only=False)
            model = YOLO(model)

        # Vérification simple de l'existence du fichier
        if not model_path.exists():
            logging.error(f"Le fichier {model_path} n'existe pas")
            return False

        # Vérification de la taille du fichier
        if model_path.stat().st_size < 1000:  # moins de 1KB
            logging.error(f"Le fichier {model_path} semble corrompu (trop petit)")
            return False

        logging.info(f"✅ Modèle {model_path.name} vérifié avec succès")
        return True

    except Exception as e:
        logging.error(f"Erreur lors de la vérification de {model_path}: {str(e)}")
        return False

def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """
    Télécharge un fichier avec barre de progression
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=desc,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        # Vérification du modèle après téléchargement
        if verify_model(dest_path):
            return True
        else:
            logging.error(f"Le modèle {dest_path} n'a pas passé la vérification")
            os.remove(dest_path)
            return False
            
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement de {url}: {str(e)}")
        if dest_path.exists():
            os.remove(dest_path)
        return False

def main():
    """
    Fonction principale pour télécharger tous les modèles
    """
    # Création des dossiers nécessaires
    models_dir = Path('src/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Téléchargement des modèles
    success_count = 0
    for model_name, url in MODELS.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            logging.info(f"Téléchargement de {model_name}...")
            if download_file(url, model_path, desc=f"Téléchargement {model_name}"):
                success_count += 1
                logging.info(f"✅ {model_name} téléchargé et vérifié avec succès")
            else:
                logging.error(f"❌ Échec du téléchargement ou de la vérification de {model_name}")
        else:
            if verify_model(model_path):
                logging.info(f"✅ {model_name} existe déjà et est valide")
                success_count += 1
            else:
                logging.warning(f"⚠️ {model_name} existe mais n'est pas valide, retéléchargement...")
                os.remove(model_path)
                if download_file(url, model_path, desc=f"Retéléchargement {model_name}"):
                    success_count += 1
                    logging.info(f"✅ {model_name} retéléchargé et vérifié avec succès")
    
    # Rapport final
    total_models = len(MODELS)
    logging.info(f"\nRésumé du téléchargement:")
    logging.info(f"✅ {success_count}/{total_models} modèles disponibles et vérifiés")
    if success_count < total_models:
        logging.warning(f"⚠️ {total_models - success_count} modèles n'ont pas pu être téléchargés ou vérifiés")
    else:
        logging.info("🎉 Tous les modèles sont prêts!")

if __name__ == "__main__":
    main() 