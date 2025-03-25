# Real-Time Object Detection App with YOLO11 🎯

Une application moderne de détection d'objets en temps réel utilisant les modèles YOLO11 de dernière génération.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![YOLO11](https://img.shields.io/badge/YOLO11-latest-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-yellow.svg)

## 🌟 Fonctionnalités

- 📸 Détection en temps réel via webcam avec YOLO11
- 🔄 Compatibilité avec tous les modèles YOLO11 (nano, small, medium, large, xlarge)
- 👥 Détection d'objets, analyse de pose et segmentation
- 📊 Analyse détaillée des capacités de chaque modèle
- 🖥️ Interface utilisateur moderne et intuitive
- 🔧 Configuration flexible avec paramètres de détection ajustables
- 🚀 Optimisations pour PyTorch 2.6+ et contournement des limitations de sécurité

## 🚀 Installation

1. Clonez le repository :
```bash
git clone https://github.com/votre-username/yolo11-detector.git
cd yolo11-detector
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

## ⚡ Téléchargement des modèles YOLO11

**IMPORTANT**: Avant de lancer l'application, vous devez télécharger au moins un modèle YOLO11.

### Option 1: Téléchargement automatique

Utilisez le script intégré pour télécharger les modèles YOLO11 :
```bash
python src/download_models.py
```

Ce script téléchargera les modèles YOLO11 de base dans le dossier `src/models/`.

### Option 2: Téléchargement manuel

Téléchargez les modèles depuis le site officiel d'Ultralytics:

1. Visitez [https://github.com/ultralytics/assets/releases/tag/v8.1.0](https://github.com/ultralytics/assets/releases/tag/v8.1.0)
2. Téléchargez les fichiers souhaités (par exemple `yolo11n.pt`, `yolo11s.pt`, `yolo11n-pose.pt`, etc.)
3. Placez les fichiers dans le dossier `src/models/` de votre projet

### Modèles recommandés

| Modèle | Type | Taille | Usage recommandé |
|--------|------|--------|------------------|
| yolo11n.pt | Détection | 6.5 MB | Appareils à faible puissance, détection rapide |
| yolo11s.pt | Détection | 21.5 MB | Bon équilibre vitesse/précision |
| yolo11n-pose.pt | Pose | 7.6 MB | Analyse de pose humaine rapide |
| yolo11n-seg.pt | Segmentation | 10.4 MB | Segmentation d'objets légère |

## 💻 Utilisation

1. Assurez-vous d'avoir téléchargé au moins un modèle YOLO11 dans le dossier `src/models/`

2. Lancez l'application :
```bash
streamlit run src/main.py
```

3. Ouvrez votre navigateur à l'adresse indiquée (généralement `http://localhost:8501`)

4. Dans l'interface de l'application :
   - Sélectionnez le type de modèle YOLO11 (détection, pose, segmentation)
   - Choisissez la taille du modèle (nano, small, medium, etc.)
   - Ajustez le seuil de confiance selon vos besoins
   - Activez la webcam pour commencer la détection

## ⚠️ Résolution des problèmes courants

- **Erreur de classe C3k2**: L'application inclut déjà un patch pour résoudre les problèmes liés à la classe C3k2 manquante dans certaines versions d'Ultralytics.

- **Problèmes de sécurité PyTorch**: L'application contient un contournement intégré pour les restrictions de sécurité de PyTorch 2.6+.

- **Modèle YOLO11 non trouvé**: Vérifiez que vous avez placé au moins un fichier modèle valide (*.pt) dans le dossier `src/models/`.

- **Problème de compatibilité**: Si besoin utiliser `pip install ultralytics --upgrade`.

## 🛠️ Configuration système requise

- Python 3.8+
- PyTorch 2.6+ (recommandé)
- CPU multi-cœurs (GPU recommandé pour les modèles medium/large)
- Webcam fonctionnelle
- 8GB RAM minimum (16GB recommandé pour les grands modèles)

## 📁 Structure du Projet

```
yolo11-detector/
├── src/
│   ├── main.py              # Application principale
│   ├── download_models.py   # Script de téléchargement des modèles
│   ├── models/              # Dossier pour les modèles YOLO11
│   ├── config/              # Configurations
│   ├── logs/                # Fichiers de log
│   └── output/              # Sorties (captures d'écran, etc.)
├── requirements.txt         # Dépendances
├── .gitignore               # Fichiers ignorés par Git
├── LICENSE                  # Licence MIT
└── README.md                # Documentation
```

## 📝 License

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.

## 🙏 Remerciements

- [Ultralytics](https://github.com/ultralytics/ultralytics) pour YOLO11
- [Streamlit](https://streamlit.io/) pour le framework d'interface
- La communauté open source pour ses contributions 