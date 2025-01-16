# Real-Time Object Detection App 🎯

Une application de détection d'objets en temps réel utilisant YOLOv8 et Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-yellow.svg)

## 🌟 Fonctionnalités

- 📸 Détection en temps réel via webcam
- 📹 Support des fichiers vidéo
- 🤖 Multiples modèles YOLO (v8 et v3)
- 👥 Détection de visages
- 📊 Analyses et statistiques en temps réel
- 📷 Capture d'écran
- 🎛️ Interface utilisateur intuitive

## 🚀 Installation

1. Clonez le repository :
```bash
git clone https://github.com/votre-username/objectdetectorproject.git
cd objectdetectorproject
```

2. Créez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Téléchargez les modèles pré-entraînés :
```bash
python src/download_models.py
```

## 💻 Utilisation

1. Lancez l'application :
```bash
streamlit run src/app.py
```

2. Ouvrez votre navigateur à l'adresse : `http://localhost:8501`

3. Sélectionnez vos options :
   - Source vidéo (Webcam ou fichier)
   - Modèle de détection
   - Paramètres de détection
   - Options d'affichage

## 🛠️ Configuration

### Modèles Disponibles
- YOLOv8 (n, s, m, l, x)
- YOLOv3 (tiny, standard, SPP)
- YOLO-Face
- YOLO-Pose
- YOLO-Seg

### Configuration Système Requise
- Python 3.8+
- CPU multi-cœurs (GPU recommandé)
- Webcam pour la détection en temps réel
- 4GB RAM minimum

## 📊 Performance

- CPU : 10-15 FPS
- Mémoire : ~1-2GB
- Temps de chargement : 2-3s

## 📁 Structure du Projet

```
objectdetectorproject/
├── src/
│   ├── app.py              # Application principale
│   ├── download_models.py  # Script de téléchargement des modèles
│   ├── models/            # Modèles pré-entraînés
│   ├── output/            # Captures d'écran et vidéos
│   └── logs/             # Fichiers de log
├── requirements.txt       # Dépendances
├── README.md             # Documentation
└── PROGRESS.md          # Suivi du développement
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.

## 🙏 Remerciements

- [Ultralytics](https://github.com/ultralytics/yolov8) pour YOLOv8
- [Streamlit](https://streamlit.io/) pour le framework d'interface
- La communauté open source pour ses contributions 