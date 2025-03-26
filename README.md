# Real-Time Object Detection App with YOLO11 🎯

Une application moderne de détection d'objets en temps réel utilisant les modèles YOLO11 de dernière génération.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![YOLO11](https://img.shields.io/badge/YOLO11-latest-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-yellow.svg)

## 📋 Table des matières
- [Aperçu](#aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Prérequis système](#-prérequis-système)
- [Installation](#-installation)
- [Téléchargement des modèles](#-téléchargement-des-modèles)
- [Utilisation](#-utilisation)
- [Architecture technique](#-architecture-technique)
- [Modèles disponibles](#-modèles-disponibles)
- [Résolution des problèmes](#-résolution-des-problèmes)
- [Futures améliorations](#-futures-améliorations)
- [Licence](#-licence)
- [Remerciements](#-remerciements)

## Aperçu

Cette application offre une interface utilisateur intuitive pour la détection d'objets, l'analyse de pose et la segmentation en temps réel à l'aide des modèles YOLO11 (You Only Look Once) de dernière génération. YOLO11 représente une évolution majeure par rapport aux versions précédentes, avec des améliorations significatives en termes de précision, vitesse et variété des tâches.

L'application est développée en Python avec Streamlit pour l'interface utilisateur, OpenCV pour la capture et le traitement vidéo, et PyTorch comme backend pour les modèles YOLO11.

## 🌟 Fonctionnalités

- **Interface intuitive** : Interface utilisateur claire et conviviale développée avec Streamlit
- **Détection en temps réel** : Analyse du flux vidéo de la webcam avec une latence minimale
- **Compatibilité multi-modèles** : Support des différentes variantes et tailles de modèles YOLO11
- **Tâches spécialisées** :
  - 🔍 **Détection d'objets** standard (80 classes COCO)
  - 👤 **Analyse de pose** pour détecter et suivre les points clés du corps humain
  - 🧩 **Segmentation sémantique** pour une détection précise des contours d'objets
- **Configuration flexible** :
  - Ajustement du seuil de confiance pour les détections
  - Paramètres de traitement vidéo personnalisables
  - Visualisation détaillée des performances du modèle
- **Optimisations de sécurité** : Contournements intégrés pour les restrictions de PyTorch 2.6+
- **Analyse des performances** : Affichage en temps réel des FPS et des statistiques de détection
- **Adaptabilité au matériel** : Optimisé pour fonctionner sur CPU, avec de meilleures performances sur GPU

## 💻 Prérequis système

- **Système d'exploitation** : Windows 10/11, macOS, Linux
- **Python** : Version 3.8 ou supérieure
- **RAM** : 8 Go minimum (16 Go recommandé)
- **Processeur** : CPU multi-cœurs récent
- **GPU** : Fortement recommandé pour les modèles medium/large/xlarge (NVIDIA avec CUDA)
- **Webcam** : Intégrée ou externe fonctionnelle
- **Espace disque** : 1 Go minimum pour les modèles et l'application
- **Réseau** : Connexion Internet pour le téléchargement initial des modèles

## 🚀 Installation

1. **Clonez le repository** :
```bash
git clone https://github.com/Farx1/objectdetectorproject.git
cd objectdetectorproject
```

2. **Créez un environnement virtuel** :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Installez les dépendances** :
```bash
pip install -r requirements.txt
```

4. **[Optionnel] Installation de CUDA** : 
Si vous disposez d'une carte graphique NVIDIA compatible, installez CUDA et cuDNN pour accélérer l'inférence :
```bash
# Pour PyTorch avec CUDA 11.8
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
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

### Modèles disponibles et recommandés

| Modèle | Type | Taille | Performance | Usage recommandé |
|--------|------|--------|-------------|------------------|
| yolo11n.pt | Détection | 6.5 MB | mAP 39.5 | Appareils à faible puissance, détection rapide |
| yolo11s.pt | Détection | 21.5 MB | mAP 47.0 | Bon équilibre vitesse/précision |
| yolo11m.pt | Détection | 68.0 MB | mAP 51.5 | Applications professionnelles nécessitant précision |
| yolo11l.pt | Détection | 51.4 MB | mAP 53.4 | Haute précision, requiert GPU |
| yolo11x.pt | Détection | 114.6 MB | mAP 54.2 | Performance maximale, requiert GPU puissant |
| yolo11n-pose.pt | Pose | 7.6 MB | mAP 50.0 | Analyse de pose humaine rapide |
| yolo11n-seg.pt | Segmentation | 10.4 MB | mAP 38.9 | Segmentation d'objets légère |

## 💻 Utilisation

1. **Lancez l'application** :
```bash
streamlit run src/main.py
```

2. **Accédez à l'interface web** :
Ouvrez votre navigateur à l'adresse indiquée (généralement `http://localhost:8501`)

3. **Configuration de l'application** :
   - Sélectionnez la catégorie de modèle (Détection, Segmentation, Pose)
   - Choisissez le modèle spécifique en fonction de vos besoins et ressources
   - Ajustez le seuil de confiance à l'aide du curseur
   - Personnalisez la largeur du flux vidéo si nécessaire

4. **Démarrage de la détection** :
   - Cliquez sur le bouton "Activer la webcam"
   - Observez les détections en temps réel
   - Consultez les statistiques de performance

5. **Arrêt de la détection** :
   - Cliquez sur "Désactiver la webcam" pour arrêter le flux
   - Ou fermez simplement l'onglet du navigateur

## 🔧 Architecture technique

### Structure du projet
```
objectdetectorproject/
├── src/
│   ├── main.py                     # Application principale Streamlit
│   ├── download_models.py          # Script de téléchargement des modèles
│   ├── models/                     # Dossier contenant les modèles YOLO11
│   │   ├── yolo11n.pt              # Modèles téléchargés
│   │   ├── yolo11n-pose.pt
│   │   └── ...
│   └── logs/                       # Fichiers de logs d'exécution
├── requirements.txt                # Dépendances Python
├── LICENSE                         # Licence MIT
└── README.md                       # Documentation
```

### Composants techniques
- **Streamlit** : Framework pour l'interface utilisateur interactive
- **OpenCV** : Traitement d'images et capture vidéo
- **PyTorch** : Backend pour l'exécution des modèles YOLO11
- **Ultralytics** : API YOLO pour la détection d'objets
- **logging** : Journalisation des événements et des erreurs
- **Patching de sécurité** : Contournements pour les restrictions de sécurité PyTorch 2.6+

## 📊 Modèles disponibles

### Détection générale d'objets (YOLO11)

YOLO11 est la dernière génération des modèles YOLO, avec des améliorations significatives:

- **Précision améliorée** : +3-5% de mAP par rapport aux modèles YOLOv8 équivalents
- **Vitesse optimisée** : Inférence plus rapide sur matériel similaire
- **Architecture modernisée** : Utilisation de composants C3k2 et autres innovations
- **Tailles disponibles** :
  - **Nano (n)** : Idéal pour appareils mobiles et systèmes embarqués
  - **Small (s)** : Bon équilibre pour la plupart des applications
    (--> à partir il faut probablement un odinateur plus puissant que la moyenne)
  - **Medium (m)** : Pour applications professionnelles nécessitant plus de précision
  - **Large (l)** : Haute précision, requiert un GPU
  - **XLarge (x)** : Performance maximale, nécessite un GPU puissant

### Segmentation sémantique (YOLO11-Seg)

Ces modèles identifient non seulement les objets mais délimitent également précisément leurs contours:

- **Détection de contours précise** : Délimitation exacte des objets
- **Double métrique** : mAP box (boîtes) et mAP mask (masques)
- **Applications** : Analyse médicale, traitement industriel, réalité augmentée
- **Tailles disponibles** : Nano à XLarge, avec compromis taille/précision

### Analyse de pose (YOLO11-Pose)

Spécialisés dans la détection et le suivi des points clés du corps humain:

- **Détection de 17 points clés** : Articulations et repères corporels
- **Haute précision** : Suivi fluide des mouvements
- **Applications** : Analyse de mouvement, fitness, ergonomie, animation
- **Tailles disponibles** : Nano à XLarge

## ⚠️ Résolution des problèmes courants

### Erreur "Can't get attribute 'C3k2'"
Cette erreur est liée à la nouvelle architecture YOLO11 et aux restrictions de sécurité de PyTorch 2.6+.
- **Solution** : L'application implémente automatiquement un patch de sécurité pour contourner ce problème.

### Performance lente sur CPU
- **Solution 1** : Utilisez des modèles plus légers (nano ou small)
- **Solution 2** : Réduisez la résolution vidéo
- **Solution 3** : Augmentez le seuil de confiance pour réduire les détections

### CUDA out of memory (sur GPU)
- **Solution 1** : Utilisez un modèle plus petit
- **Solution 2** : Réduisez la résolution vidéo
- **Solution 3** : Vérifiez que d'autres applications n'utilisent pas votre GPU

### Modèle non trouvé
- **Solution 1** : Exécutez `python src/download_models.py` pour télécharger automatiquement les modèles
- **Solution 2** : Vérifiez que les fichiers .pt sont correctement placés dans le dossier `src/models/`

### Problèmes de webcam
- **Solution 1** : Assurez-vous qu'aucune autre application n'utilise votre webcam
- **Solution 2** : Modifiez l'index de la caméra dans le code si vous avez plusieurs webcams

## 🔮 Futures améliorations

- **Support YOLO11-OBB** : Intégration des modèles de détection d'objets orientés (actuellement désactivés)
- **Support YOLO11-CLS** : Ajout des modèles de classification d'images (actuellement désactivés)
- **Mode fichier vidéo** : Support du traitement de fichiers vidéo pré-enregistrés
- **Enregistrement vidéo** : Fonctionnalité pour sauvegarder les sessions de détection
- **Amélioration de l'interface** : Visualisations avancées et tableaux de bord
- **Analyse temporelle** : Suivi des objets sur la durée et analyses statistiques
- **Support multi-caméras** : Détection simultanée sur plusieurs flux vidéo

## 📝 Licence

Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## 🙏 Remerciements

- [Ultralytics](https://github.com/ultralytics/ultralytics) pour YOLO11 et leur travail exceptionnel
- [Streamlit](https://streamlit.io/) pour leur framework d'interface utilisateur
- [PyTorch](https://pytorch.org/) pour leur framework de deep learning
- [OpenCV](https://opencv.org/) pour les outils de vision par ordinateur
- La communauté open source pour ses contributions et son soutien 
