import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
import shutil
import requests
import logging
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Configuration des modeles specialises
SPECIALIZED_MODELS = {
    "General": {
        "path": "src/models/yolov8x.pt",
        "confidence": 0.3
    },
    "Visages": {
        "path": "src/models/yolov8n-face.pt",
        "confidence": 0.5
    },
    "Vehicules": {
        "path": "src/models/yolov8n-seg.pt",
        "confidence": 0.4
    }
}

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Detecteur d'Objets en Temps Reel",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Detecteur d'objets en temps reel avec YOLOv8"
    }
)

# Ajustement de la taille de la barre latérale avec CSS personnalisé
st.markdown("""
    <style>
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 400px;
            max-width: 400px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False

if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None

# Configuration du logging
logging.basicConfig(
    filename='src/logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Classes pour la gestion des modeles et labels
class ModelManager:
    def __init__(self, model_type, model_path, confidence):
        self.model_type = model_type
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.labels = {}
        
    def load(self):
        try:
            self.model = YOLO(self.model_path)
            self.model.to('cpu')
            if hasattr(self.model, 'model'):
                self.model.model.float()
                torch.set_num_threads(os.cpu_count())
                torch.set_grad_enabled(False)
            self._setup_labels()
            return True
        except Exception as e:
            st.error(f"Erreur chargement {self.model_type}: {str(e)}")
            return False
    
    def _setup_labels(self):
        if self.model_type == "General":
            self.labels = DETAILED_LABELS
        elif self.model_type == "Visages":
            self.labels = {"0": "Visage"}
        elif self.model_type == "Vehicules":
            self.labels = {
                "0": "Voiture",
                "1": "Moto",
                "2": "Camion",
                "3": "Bus",
                "4": "Velo"
            }
        
    def get_label(self, cls_id):
        if str(cls_id) in self.labels:
            return self.labels[str(cls_id)]
        return f"Classe_{cls_id}"

# File d'attente pour chaque modele
class ModelQueues:
    def __init__(self):
        self.frame_queues = {}
        self.result_queues = {}
        
    def setup_queues(self, model_type):
        self.frame_queues[model_type] = queue.Queue(maxsize=4)
        self.result_queues[model_type] = queue.Queue(maxsize=4)
        
    def get_queues(self, model_type):
        return self.frame_queues[model_type], self.result_queues[model_type]

# Thread de capture ameliore
def capture_thread(cap, queues, model_types):
    while cap.isOpened() and st.session_state.detection_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        
        # Distribuer le frame à tous les modèles
        for model_type in model_types:
            if not queues.frame_queues[model_type].full():
                queues.frame_queues[model_type].put(frame.copy())
    cap.release()

# Thread de traitement ameliore
def process_thread(model_manager, queues):
    frame_queue, result_queue = queues.get_queues(model_manager.model_type)
    while st.session_state.detection_active:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                results = model_manager.model(frame, conf=model_manager.confidence)[0]
                detections = []
                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls)
                        conf = float(box.conf)
                        label = model_manager.get_label(cls)
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'label': label,
                            'confidence': conf,
                            'model_type': model_manager.model_type
                        })
                result_queue.put((frame, detections))
            except Exception as e:
                logging.error(f"Erreur traitement {model_manager.model_type}: {str(e)}")
                continue

# Fonction de fusion des resultats
def merge_detections(frame, all_results):
    frame_with_results = frame.copy()
    all_detections = []
    
    for model_type, (_, detections) in all_results.items():
        if detections:
            all_detections.extend(detections)
    
    # Dessin des detections
    for det in all_detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['confidence']
        model_type = det['model_type']
        
        # Couleur différente par type de modèle
        color = (0, 255, 0)  # General
        if model_type == "Visages":
            color = (255, 0, 0)  # Rouge pour visages
        elif model_type == "Vehicules":
            color = (0, 0, 255)  # Bleu pour véhicules
        
        cv2.rectangle(frame_with_results, (x1, y1), (x2, y2), color, 2)
        
        if show_labels:
            display_label = f"{label}"
            if show_conf:
                display_label += f" {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_with_results, (x1, y1-20), (x1 + label_w, y1), (0, 0, 0), -1)
            cv2.putText(frame_with_results, display_label, (x1, y1-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame_with_results, all_detections

# Modification du chargement des modeles
@st.cache_resource
def load_models():
    try:
        model = YOLO(model_path)
        model.to('cpu')
        if hasattr(model, 'model'):
            model.model.float()
            torch.set_num_threads(os.cpu_count())
            torch.set_grad_enabled(False)
        return {"main": model}
    except Exception as e:
        st.error(f"Erreur chargement modèle: {str(e)}")
        return None

# Configuration optimisee pour CPU
def setup_processing():
    try:
        # Optimisations CPU
        torch.set_num_threads(os.cpu_count())
        torch.backends.cpu.enabled = True
        
        # Informations systeme
        st.sidebar.info(f"""
        Configuration systeme:
        - Processeurs: {os.cpu_count()}
        - PyTorch threads: {torch.get_num_threads()}
        - Mode: CPU optimise
        """)
        return "cpu"
    except Exception as e:
        st.error(f"Erreur de configuration: {str(e)}")
        return "cpu"

# Utilisation du CPU optimise
device = setup_processing()

# Configuration du device avec plus de details
if device == "cpu":
    if 'show_warning' not in st.session_state:
        st.session_state.show_warning = True
    if st.session_state.show_warning:
        col_warn, col_close = st.sidebar.columns([8,1])
        with col_warn:
            st.warning("""
            Mode CPU - Performance reduite
            Causes possibles:
            - Pas de GPU NVIDIA detecte
            - Drivers NVIDIA non installes
            - CUDA non installe
            - PyTorch CPU-only installe
            """)
        with col_close:
            if st.button("❌", key="close_warning", help="Fermer"):
                st.session_state.show_warning = False
                st.experimental_rerun()
    
    # Suggestion d'installation
    if 'show_info' not in st.session_state:
        st.session_state.show_info = True
    if st.session_state.show_info:
        col_info, col_close = st.sidebar.columns([8,1])
        with col_info:
            st.info("""
            Pour utiliser le GPU:
            1. Installez les drivers NVIDIA
            2. Installez CUDA Toolkit
            3. Reinstallez PyTorch avec CUDA:
            pip3 install torch torchvision --index-url
            https://download.pytorch.org/whl/
            cu118
            """)
        with col_close:
            if st.button("❌", key="close_info", help="Fermer"):
                st.session_state.show_info = False
                st.experimental_rerun()

# Fonction de téléchargement YOLOv3 (déplacée au début)
def download_yolov3():
    st.info("Téléchargement des modèles YOLOv3...")
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs("src/models", exist_ok=True)
    
    # Liste des modèles YOLOv3 avec leurs URLs
    yolov3_models = {
        "yolov3-tiny.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3-tiny.pt",
        "yolov3.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3.pt",
        "yolov3-spp.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3-spp.pt"
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (model_file, url) in enumerate(yolov3_models.items()):
        model_path = os.path.join("src/models", model_file)
        if not os.path.exists(model_path):
            try:
                status_text.text(f"Téléchargement de {model_file}...")
                
                # Téléchargement avec barre de progression
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                # Écriture du fichier avec progression
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                st.success(f"✅ {model_file} téléchargé avec succès!")
            except Exception as e:
                st.error(f"❌ Erreur lors du téléchargement de {model_file}: {str(e)}")
        else:
            st.info(f"ℹ️ {model_file} existe déjà")
        
        # Mise à jour de la barre de progression
        progress_bar.progress((i + 1) / len(yolov3_models))
    
    status_text.text("Téléchargement des modèles YOLOv3 terminé!")
    return True

# Vérification et création des dossiers nécessaires
required_dirs = ['src/models', 'src/output', 'src/logs']
for dir_path in required_dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Dossier vérifié/créé : {dir_path}")

# Titre de l'application
st.title("Détecteur d'Objets en Temps Réel")

# Configuration de la barre latérale
st.sidebar.title("Configuration")

# 1. Source d'entrée
st.sidebar.header("📹 Source")
source_option = st.sidebar.selectbox(
    "Sélectionner la source",
    ["Webcam", "Fichier Vidéo"]
)

# 2. Sélection des modèles
st.sidebar.header("🤖 Modèles de détection")
model_type = st.sidebar.selectbox(
    "Type de modèle",
    ["YOLO (Détection générale)", "YOLOv3 (Léger)", "YOLO-Face (Visages)", "YOLO-Pose (Poses)", "YOLO-Seg (Segmentation)"]
)

# Mapping des types de modèles et leurs versions
model_options = {
    "YOLO (Détection générale)": {
        "YOLOv8n (Rapide)": "src/models/yolov8n.pt",
        "YOLOv8s (Équilibré)": "src/models/yolov8s.pt",
        "YOLOv8m (Précis)": "src/models/yolov8m.pt",
        "YOLOv8l (Très précis)": "src/models/yolov8l.pt",
        "YOLOv8x (Ultra précis)": "src/models/yolov8x.pt"
    },
    "YOLOv3 (Léger)": {
        "YOLOv3-tiny (Très rapide)": "src/models/yolov3-tiny.pt",
        "YOLOv3 (Standard)": "src/models/yolov3.pt",
        "YOLOv3-SPP (Amélioré)": "src/models/yolov3-spp.pt"
    },
    "YOLO-Face (Visages)": {
        "YOLOv8n-Face (Rapide)": "src/models/yolov8n-face.pt",
        "YOLOv8s-Face (Équilibré)": "src/models/yolov8s-face.pt",
        "YOLOv8m-Face (Précis)": "src/models/yolov8m-face.pt"
    },
    "YOLO-Pose (Poses)": {
        "YOLOv8n-Pose (Rapide)": "src/models/yolov8n-pose.pt",
        "YOLOv8s-Pose (Équilibré)": "src/models/yolov8s-pose.pt",
        "YOLOv8m-Pose (Précis)": "src/models/yolov8m-pose.pt"
    },
    "YOLO-Seg (Segmentation)": {
        "YOLOv8n-Seg (Rapide)": "src/models/yolov8n-seg.pt",
        "YOLOv8s-Seg (Équilibré)": "src/models/yolov8s-seg.pt",
        "YOLOv8m-Seg (Précis)": "src/models/yolov8m-seg.pt"
    }
}

# Sélection du modèle spécifique
model_option = st.sidebar.selectbox(
    "Version du modèle",
    list(model_options[model_type].keys())
)

# Obtention du chemin du modèle
model_path = model_options[model_type][model_option]

# Information sur le modèle
st.sidebar.info(f"""
Modèle sélectionné : {model_option}
Type : {model_type}
Chemin : {model_path}
""")

# 3. Paramètres de détection
st.sidebar.header("⚙️ Paramètres")
confidence_threshold = st.sidebar.slider(
    "Seuil de confiance global",
    min_value=0.0,
    max_value=1.0,
    value=0.3
)

# 4. Options d'affichage
st.sidebar.header("🎯 Affichage")
show_labels = st.sidebar.checkbox("Afficher les etiquettes", value=True)
show_conf = st.sidebar.checkbox("Afficher les scores de confiance", value=True)
show_fps = st.sidebar.checkbox("Afficher FPS", value=True)

# 5. Options de capture
st.sidebar.header("📸 Capture")
enable_recording = st.sidebar.checkbox("Activer l'enregistrement", value=False)
enable_screenshot = st.sidebar.checkbox("Activer la capture d'image", value=False)
if enable_screenshot:
    screenshot_key = st.sidebar.text_input("Touche de capture (par défaut: 's')", value='s')

# 6. Options avancées
st.sidebar.header("🔧 Options avancées")
track_objects = st.sidebar.checkbox("Activer le tracking d'objets", value=False)
enable_analytics = st.sidebar.checkbox("Activer les analyses", value=True)

# 7. Réinitialisation
st.sidebar.header("🔄 Réinitialisation")
if st.sidebar.button("Reinitialiser les analyses", key="reset_analytics_sidebar"):
    st.session_state.detection_history = []

# 8. Informations système (si en mode CPU)
if device == "cpu":
    st.sidebar.header("💻 Système")
    if 'show_warning' not in st.session_state:
        st.session_state.show_warning = True
    if st.session_state.show_warning:
        col_warn, col_close = st.sidebar.columns([8,1])
        with col_warn:
            st.warning("""
            Mode CPU - Performance reduite
            Causes possibles:
            - Pas de GPU NVIDIA detecte
            - Drivers NVIDIA non installes
            - CUDA non installe
            - PyTorch CPU-only installe
            """)
        with col_close:
            if st.button("❌", key="close_warning_cpu", help="Fermer"):
                st.session_state.show_warning = False
                st.experimental_rerun()
    
    if 'show_info' not in st.session_state:
        st.session_state.show_info = True
    if st.session_state.show_info:
        col_info, col_close = st.sidebar.columns([8,1])
        with col_info:
            st.info("""
            Pour utiliser le GPU:
            1. Installez les drivers NVIDIA
            2. Installez CUDA Toolkit
            3. Reinstallez PyTorch avec CUDA:
            pip3 install torch torchvision --index-url
            https://download.pytorch.org/whl/
            cu118
            """)
        with col_close:
            if st.button("❌", key="close_info_gpu", help="Fermer"):
                st.session_state.show_info = False
                st.experimental_rerun()

# Après les imports, ajoutons un dictionnaire de labels détaillés
#Vérifie les indentations pour les labels !!!
DETAILED_LABELS = {
    'person': 'Personne',
    'bicycle': 'Velo',
    'car': 'Voiture',
    'motorcycle': 'Moto',
    'airplane': 'Avion',
    'bus': 'Bus',
    'train': 'Train',
    'truck': 'Camion',
    'boat': 'Bateau',
    'traffic light': 'Feu de circulation',
    'fire hydrant': 'Borne incendie',
    'stop sign': 'Panneau stop',
    'parking meter': 'Parcmetre',
    'bench': 'Banc',
    'bird': 'Oiseau',
    'cat': 'Chat',
    'dog': 'Chien',
    'horse': 'Cheval',
    'sheep': 'Mouton',
    'cow': 'Vache',
    'elephant': 'Elephant',
    'bear': 'Ours',
    'zebra': 'Zebre',
    'giraffe': 'Girafe',
    'backpack': 'Sac a dos',
    'umbrella': 'Parapluie',
    'handbag': 'Sac a main',
    'tie': 'Cravate',
    'suitcase': 'Valise',
    'frisbee': 'Frisbee',
    'skis': 'Skis',
    'snowboard': 'Snowboard',
    'sports ball': 'Ballon de sport',
    'kite': 'Cerf-volant',
    'baseball bat': 'Batte de baseball',
    'baseball glove': 'Gant de baseball',
    'skateboard': 'Skateboard',
    'surfboard': 'Planche de surf',
    'tennis racket': 'Raquette de tennis',
    'bottle': 'Bouteille',
    'wine glass': 'Verre de vin',
    'cup': 'Tasse',
    'fork': 'Fourchette',
    'knife': 'Couteau',
    'spoon': 'Cuillere',
    'bowl': 'Bol',
    'banana': 'Banane',
    'apple': 'Pomme',
    'sandwich': 'Sandwich',
    'orange': 'Orange',
    'broccoli': 'Brocoli',
    'carrot': 'Carotte',
    'hot dog': 'Hot-dog',
    'pizza': 'Pizza',
    'donut': 'Donut',
    'cake': 'Gateau',
    'chair': 'Chaise',
    'couch': 'Canape',
    'potted plant': 'Plante en pot',
    'bed': 'Lit',
    'dining table': 'Table a manger',
    'toilet': 'Toilettes',
    'tv': 'Television',
    'laptop': 'Ordinateur portable',
    'mouse': 'Souris',
    'remote': 'Telecommande',
    'keyboard': 'Clavier',
    'cell phone': 'Telephone portable',
    'microwave': 'Four micro-ondes',
    'oven': 'Four',
    'toaster': 'Grille-pain',
    'sink': 'Evier',
    'refrigerator': 'Refrigerateur',
    'book': 'Livre',
    'clock': 'Horloge',
    'vase': 'Vase',
    'scissors': 'Ciseaux',
    'teddy bear': 'Ours en peluche',
    'hair drier': 'Seche-cheveux',
    'toothbrush': 'Brosse a dents',
    'sunglasses': 'Lunettes de soleil'
}

# Dictionnaire etendu de labels
EXTENDED_LABELS = {
    # Labels standard YOLO
    **DETAILED_LABELS,
    
    # Vehicules supplementaires
    'motorcycle_sport': 'Moto de sport',
    'motorcycle_cruiser': 'Moto cruiser',
    'bicycle_mountain': 'VTT',
    'bicycle_road': 'Velo de route',
    'truck_pickup': 'Pickup',
    'truck_delivery': 'Camion de livraison',
    'bus_school': 'Bus scolaire',
    'bus_transit': 'Bus de ville',
    
    # Mobilier supplementaire
    'desk': 'Bureau',
    'shelf': 'Etagere',
    'cabinet': 'Armoire',
    'lamp': 'Lampe',
    'mirror': 'Miroir',
    'carpet': 'Tapis',
    'curtain': 'Rideau',
    
    # Electronique supplementaire
    'printer': 'Imprimante',
    'scanner': 'Scanner',
    'router': 'Routeur',
    'speaker': 'Enceinte',
    'headphones': 'Casque audio',
    'tablet': 'Tablette',
    'smartwatch': 'Montre connectee',
    
    # Vetements
    'shirt': 'Chemise',
    'pants': 'Pantalon',
    'dress': 'Robe',
    'jacket': 'Veste',
    'shoes': 'Chaussures',
    'hat': 'Chapeau',
    'gloves': 'Gants',
    
    # Aliments supplementaires
    'coffee': 'Cafe',
    'tea': 'The',
    'juice': 'Jus',
    'water_bottle': 'Bouteille d eau',
    'fruit_basket': 'Corbeille de fruits',
    'vegetables': 'Legumes',
    'bread': 'Pain',
    
    # Animaux supplementaires
    'fish': 'Poisson',
    'hamster': 'Hamster',
    'rabbit': 'Lapin',
    'parrot': 'Perroquet',
    'turtle': 'Tortue',
    'lizard': 'Lezard',
    
    # Sports et loisirs
    'basketball': 'Ballon de basket',
    'football': 'Ballon de foot',
    'tennis_ball': 'Balle de tennis',
    'golf_club': 'Club de golf',
    'yoga_mat': 'Tapis de yoga',
    'dumbbell': 'Haltere',
    
    # Instruments de musique
    'guitar': 'Guitare',
    'piano': 'Piano',
    'violin': 'Violon',
    'drums': 'Batterie',
    'flute': 'Flute',
    
    # Outils
    'hammer': 'Marteau',
    'screwdriver': 'Tournevis',
    'wrench': 'Cle a molette',
    'drill': 'Perceuse',
    'saw': 'Scie',
    
    # Materiel medical
    'stethoscope': 'Stethoscope',
    'wheelchair': 'Fauteuil roulant',
    'crutches': 'Bequilles',
    'bandage': 'Bandage',
    'thermometer': 'Thermometre'
}

# Dictionnaire des noms de classes COCO
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Dictionnaire des expressions faciales avec emojis et scores
FACIAL_EXPRESSIONS = {
    0: {"label": "Neutre 😐", "color": (255, 255, 255)},
    1: {"label": "Heureux 😊", "color": (0, 255, 0)},
    2: {"label": "Triste 😢", "color": (255, 0, 0)},
    3: {"label": "Surpris 😲", "color": (0, 255, 255)},
    4: {"label": "En colère 😠", "color": (0, 0, 255)},
    5: {"label": "Dégoûté 🤢", "color": (128, 0, 128)},
    6: {"label": "Effrayé 😨", "color": (255, 165, 0)}
}

# Chargement des modeles
models = load_models()

# Verification des modeles charges
if not models:
    if 'show_error' not in st.session_state:
        st.session_state.show_error = True
    if st.session_state.show_error:
        col_error, col_close = st.columns([8,1])
        with col_error:
            st.error("Aucun modele n'a pu etre charge. Verifiez que les fichiers des modeles sont presents dans le dossier src/models/")
        with col_close:
            if st.button("❌", key="close_error", help="Fermer"):
                st.session_state.show_error = False
                st.experimental_rerun()
    st.stop()

# Interface principale avec gestion des erreurs de la webcam
col1, col2 = st.columns([2, 1])

with col1:
    if source_option == "Webcam":
        video_placeholder = st.empty()
        
        # Boutons de controle
        col_start, col_screenshot = st.columns(2)
        
        # Bouton demarrer/arreter
        with col_start:
            button_text = "🔴 Arreter" if st.session_state.detection_active else "▶️ Demarrer"
            if st.button(button_text, key="detection_button"):
                st.session_state.detection_active = not st.session_state.detection_active
                st.experimental_rerun()
        
        # Bouton capture
        with col_screenshot:
            if enable_screenshot:
                if st.button("📸 Capturer", key="capture_button"):
                    if 'last_frame' in st.session_state:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join("src", "output", f"capture_{timestamp}.jpg")
                        cv2.imwrite(save_path, cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_RGB2BGR))
                        st.success(f"Image sauvegardee: {save_path}")
        
        # Affichage de l'etat
        if st.session_state.detection_active:
            if 'show_status' not in st.session_state:
                st.session_state.show_status = True
            if st.session_state.show_status:
                col_status, col_close = st.columns([8,1])
                with col_status:
                    st.info("Detection en cours...")
                with col_close:
                    if st.button("❌", key="close_status", help="Fermer"):
                        st.session_state.show_status = False
                        st.experimental_rerun()
            
    else:
        # Gestion des fichiers video
        video_file = st.file_uploader("Choisir un fichier video", type=['mp4', 'avi', 'mov'])
        
        if video_file is not None:
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                if not cap.isOpened():
                    st.error("Impossible d'ouvrir le fichier video")
                else:
                    video_placeholder = st.empty()
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame, fps, detections = process_frame_multi_model(
                            frame, 
                            models,
                            {"main": confidence_threshold}
                        )
                        video_placeholder.image(processed_frame)
                    
                    cap.release()
                
                os.unlink(tfile.name)
                
            except Exception as e:
                st.error(f"Erreur lors du traitement de la video: {str(e)}")
                logging.error(f"Erreur video: {str(e)}")

with col2:
    if enable_analytics and st.session_state.detection_history:
        # Affichage des statistiques
        st.subheader("Analyse des detections")
        df = pd.DataFrame(st.session_state.detection_history)
        
        # Graphique des detections par classe
        fig_counts = px.histogram(df, x='label', title="Nombre de detections par classe")
        st.plotly_chart(fig_counts)
        
        # Graphique de confiance moyenne par classe
        conf_by_class = df.groupby('label')['confidence'].mean().reset_index()
        fig_conf = px.bar(conf_by_class, x='label', y='confidence',
                         title="Confiance moyenne par classe")
        st.plotly_chart(fig_conf)

# Fonction de traitement multi-modeles
def process_frame_multi_model(frame, models, confidence_thresholds):
    if not models:
        return frame, 0, None
    
    try:
        start_time = time.time()
        frame_resized = cv2.resize(frame, (640, 480))
        all_detections = []
        
        # Traitement des détections avec le modèle sélectionné
        model = models["main"]
        results = model(frame_resized, conf=confidence_threshold)[0]
        
        frame_with_results = frame.copy()
        
        # Dessin des détections
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                
                # Gestion des labels selon le type de modèle
                if "Face" in model_type:
                    # Analyse simple de l'expression basée sur les caractéristiques du visage
                    face_region = frame[y1:y2, x1:x2]
                    if face_region.size > 0:
                        # Convertir en niveaux de gris pour l'analyse
                        gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
                        
                        # Calculer des métriques simples
                        avg_brightness = np.mean(gray_face)
                        contrast = np.std(gray_face)
                        
                        # Logique simple pour déterminer l'expression
                        if contrast > 50:  # Beaucoup de variations = expression prononcée
                            if avg_brightness > 130:  # Plus lumineux = expression positive
                                label = "Heureux 😊"
                                color = (0, 255, 0)
                            else:  # Plus sombre = expression négative
                                label = "Sérieux 😐"
                                color = (255, 0, 0)
                        else:  # Peu de variations = expression neutre
                            label = "Neutre 😐"
                            color = (255, 255, 255)
                    else:
                        label = "Visage"
                        color = (255, 0, 0)
                else:
                    # Labels standards COCO
                    if cls < len(COCO_CLASSES):
                        class_name = COCO_CLASSES[cls]
                        label = DETAILED_LABELS.get(class_name, class_name)
                    else:
                        label = f"Classe_{cls}"
                    color = (0, 255, 0)
                
                # Dessin du rectangle
                cv2.rectangle(frame_with_results, (x1, y1), (x2, y2), color, 2)
                
                # Affichage du label
                if show_labels:
                    display_label = f"{label}"
                    if show_conf:
                        display_label += f" {conf:.2f}"
                    
                    # Ajustement de la position du label
                    font_scale = 0.6
                    font_thickness = 2
                    (label_w, label_h), baseline = cv2.getTextSize(
                        display_label, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        font_thickness
                    )
                    
                    # Fond noir pour le texte
                    cv2.rectangle(
                        frame_with_results, 
                        (x1, y1-label_h-baseline-5), 
                        (x1 + label_w, y1), 
                        (0, 0, 0), 
                        -1
                    )
                    
                    # Texte blanc
                    cv2.putText(
                        frame_with_results, 
                        display_label, 
                        (x1, y1-baseline-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        (255, 255, 255), 
                        font_thickness
                    )
                
                all_detections.append(box)
        
        # Affichage FPS
        fps = 1.0 / (time.time() - start_time)
        if show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame_with_results, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_with_results, "CPU", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Sauvegarde du dernier frame pour la capture
        if enable_screenshot:
            st.session_state.last_frame = frame_with_results.copy()
        
        return frame_with_results, fps, all_detections
        
    except Exception as e:
        logging.error(f"Erreur de traitement multi-modele: {str(e)}")
        return frame, 0, None

# Boucle de detection webcam
if source_option == "Webcam" and st.session_state.detection_active:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Impossible d'acceder a la webcam")
            st.session_state.detection_active = False
            st.experimental_rerun()
        else:
            video_placeholder = st.empty()
            
            while cap.isOpened() and st.session_state.detection_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame, fps, detections = process_frame_multi_model(
                    frame, 
                    models,
                    {"main": confidence_threshold}
                )
                
                if processed_frame is not None:
                    video_placeholder.image(processed_frame)
                    
                    # Mise à jour des statistiques
                    if enable_analytics and detections:
                        for det in detections:
                            st.session_state.detection_history.append({
                                'label': det.cls,
                                'confidence': float(det.conf),
                                'timestamp': datetime.now()
                            })
                
                time.sleep(0.01)  # Évite la surcharge CPU
            
            cap.release()
            
    except Exception as e:
        st.error(f"Erreur webcam: {str(e)}")
        st.session_state.detection_active = False
        st.experimental_rerun() 