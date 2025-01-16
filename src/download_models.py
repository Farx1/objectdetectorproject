import os
import requests
from pathlib import Path
import logging
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        return True
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement de {url}: {str(e)}")
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
                logging.info(f"✅ {model_name} téléchargé avec succès")
            else:
                logging.error(f"❌ Échec du téléchargement de {model_name}")
        else:
            logging.info(f"ℹ️ {model_name} existe déjà")
            success_count += 1
    
    # Rapport final
    total_models = len(MODELS)
    logging.info(f"\nRésumé du téléchargement:")
    logging.info(f"✅ {success_count}/{total_models} modèles disponibles")
    if success_count < total_models:
        logging.warning(f"⚠️ {total_models - success_count} modèles n'ont pas pu être téléchargés")
    else:
        logging.info("🎉 Tous les modèles sont prêts!")

if __name__ == "__main__":
    main() 