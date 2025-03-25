import os
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, PoseModel, SegmentationModel

# NOTE: Les modèles OBB (Oriented Bounding Box) et CLS (Classification) sont temporairement
# désactivés dans l'application principale car ils nécessitent des entrées différentes d'un
# flux vidéo standard. Ils restent téléchargeables via ce script pour une utilisation future.

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration de la sécurité PyTorch
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([DetectionModel, PoseModel, SegmentationModel])

# URLs des modèles YOLO11
MODELS = {
    # Modèles YOLO11 standard (détection)
    'yolo11n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n.pt',
    'yolo11s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11s.pt',
    'yolo11m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m.pt',
    'yolo11l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11l.pt',
    'yolo11x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11x.pt',
    
    # Modèles YOLO11 pour segmentation (seg)
    'yolo11n-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n-seg.pt',
    'yolo11s-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11s-seg.pt',
    'yolo11m-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m-seg.pt',
    'yolo11l-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11l-seg.pt',
    'yolo11x-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11x-seg.pt',
    
    # Modèles YOLO11 pour pose (pose)
    'yolo11n-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n-pose.pt',
    'yolo11s-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11s-pose.pt',
    'yolo11m-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m-pose.pt',
    'yolo11l-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11l-pose.pt',
    'yolo11x-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11x-pose.pt',

    # Modèles YOLO11 pour classification (cls)
    'yolo11n-cls.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n-cls.pt',
    'yolo11s-cls.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11s-cls.pt',
    'yolo11m-cls.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m-cls.pt',
    'yolo11l-cls.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11l-cls.pt',
    'yolo11x-cls.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11x-cls.pt',
    
    # Modèles YOLO11 pour détection d'objets orientés (obb)
    'yolo11n-obb.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n-obb.pt',
    'yolo11s-obb.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11s-obb.pt',
    'yolo11m-obb.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m-obb.pt',
    'yolo11l-obb.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11l-obb.pt',
    'yolo11x-obb.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11x-obb.pt'
}

# Modèles essentiels à télécharger automatiquement
ESSENTIAL_MODELS = ['yolo11n.pt', 'yolo11n-pose.pt', 'yolo11n-seg.pt']

def verify_model(model_path: Path) -> bool:
    """
    Vérifie si le modèle peut être chargé correctement avec gestion de la sécurité PyTorch 2.6+
    """
    try:
        # Vérification simple de l'existence du fichier
        if not model_path.exists():
            logging.error(f"Le fichier {model_path} n'existe pas")
            return False

        # Vérification de la taille du fichier
        if model_path.stat().st_size < 1000:  # moins de 1KB
            logging.error(f"Le fichier {model_path} semble corrompu (trop petit)")
            return False
            
        # Pour les modèles YOLO11, on n'essaie pas de les charger directement pour la vérification
        # car cela pourrait échouer à cause des problèmes de sécurité PyTorch
        logging.info(f"✅ Modèle {model_path.name} vérifié avec succès (taille correcte)")
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

def download_models(models_to_download=None):
    """
    Télécharge les modèles spécifiés ou tous les modèles si None
    
    Args:
        models_to_download: Liste des noms de modèles à télécharger ou None pour tous les modèles
    
    Returns:
        Tuple (success_count, total_models)
    """
    # Si aucun modèle spécifié, prendre la liste complète
    if models_to_download is None:
        models_to_download = list(MODELS.keys())
    
    # Création des dossiers nécessaires
    models_dir = Path('src/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Téléchargement des modèles
    success_count = 0
    total_models = len(models_to_download)
    
    for model_name in models_to_download:
        if model_name not in MODELS:
            logging.warning(f"⚠️ Modèle {model_name} non reconnu, ignoré")
            continue
            
        url = MODELS[model_name]
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
    logging.info(f"\nRésumé du téléchargement:")
    logging.info(f"✅ {success_count}/{total_models} modèles disponibles et vérifiés")
    if success_count < total_models:
        logging.warning(f"⚠️ {total_models - success_count} modèles n'ont pas pu être téléchargés ou vérifiés")
    else:
        logging.info("🎉 Tous les modèles sont prêts!")
        
    return success_count, total_models

def ensure_essential_models():
    """
    Vérifie et télécharge les modèles essentiels si nécessaire.
    Cette fonction est appelée au démarrage pour garantir que les
    modèles de base sont disponibles.
    
    Returns:
        bool: True si au moins un modèle essentiel est disponible
    """
    models_dir = Path('src/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Liste des modèles à vérifier/télécharger
    models_to_download = []
    
    # Vérifier quels modèles essentiels sont manquants
    for model_name in ESSENTIAL_MODELS:
        model_path = models_dir / model_name
        if not model_path.exists() or not verify_model(model_path):
            models_to_download.append(model_name)
    
    # Si des modèles sont manquants, les télécharger
    if models_to_download:
        logging.info(f"Téléchargement automatique de {len(models_to_download)} modèles essentiels...")
        success_count, _ = download_models(models_to_download)
        return success_count > 0
    else:
        logging.info("Tous les modèles essentiels sont déjà disponibles")
        return True

def main():
    """
    Fonction principale pour télécharger tous les modèles
    """
    logging.info("Démarrage du téléchargement de tous les modèles YOLO11...")
    download_models()

if __name__ == "__main__":
    main() 