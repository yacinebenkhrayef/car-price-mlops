# car-price-mlops
# 🚗 Car Price Prediction - MLOps Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-data%20versioning-orange.svg)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-containerized-2496ED.svg)](https://www.docker.com/)

Projet MLOps complet pour la prédiction de prix de voitures d'occasion avec pipeline automatisé de bout en bout.

## 📋 Table des matières

- [Contexte](#contexte)
- [Architecture](#architecture)
- [Stack Technique](#stack-technique)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Pipeline MLOps](#pipeline-mlops)
- [Déploiement](#déploiement)
- [Monitoring](#monitoring)
- [Roadmap 8 Semaines](#roadmap-8-semaines)

## 🎯 Contexte

### Problème Métier
Développer une API de prédiction de prix de voitures d'occasion pour aider les vendeurs et acheteurs à estimer la valeur d'un véhicule basée sur ses caractéristiques.

### Objectifs MLOps
- ✅ Versioning complet (code, données, modèles)
- ✅ Pipeline reproductible et automatisé
- ✅ Tracking d'expériences avec MLflow
- ✅ Déploiement continu avec CI/CD
- ✅ Monitoring en production

## 🏗️ Architecture

```
┌─────────────────┐
│   GitHub Repo   │
└────────┬────────┘
         │
    ┌────▼─────┐
    │   DVC    │ ──── Versioning données
    └────┬─────┘
         │
    ┌────▼─────────┐
    │  Training    │
    │  Pipeline    │
    └────┬─────────┘
         │
    ┌────▼─────┐
    │  MLflow  │ ──── Tracking expériences
    └────┬─────┘
         │
    ┌────▼──────────┐
    │ Model Registry│
    └────┬──────────┘
         │
    ┌────▼─────┐
    │ FastAPI  │ ──── API déployée
    └────┬─────┘
         │
    ┌────▼─────────┐
    │ GitHub Actions│ ──── CI/CD
    └────┬──────────┘
         │
    ┌────▼─────┐
    │  Cloud   │ ──── Production
    └────┬─────┘
         │
    ┌────▼─────────┐
    │  Monitoring  │ ──── Evidently AI
    └──────────────┘
```

## 🛠️ Stack Technique

| Composant | Technologie | Raison |
|-----------|-------------|--------|
| **Versioning Code** | Git/GitHub | Standard industrie |
| **Versioning Data** | DVC | Gestion datasets volumineux |
| **Tracking ML** | MLflow | Suivi expériences & modèles |
| **ML Framework** | Scikit-learn, XGBoost | Performance & simplicité |
| **API** | FastAPI | Rapide, documentation auto |
| **Containerisation** | Docker | Reproductibilité |
| **CI/CD** | GitHub Actions | Intégration native GitHub |
| **Cloud** | Render/AWS | Déploiement production |
| **Monitoring** | Evidently AI | Détection drift |
| **Orchestration** | Prefect (optionnel) | Automatisation pipelines |

## 📁 Structure du Projet

```
car-price-mlops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Tests automatiques
│       ├── cd.yml                 # Déploiement automatique
│       └── model-training.yml     # Re-training automatique
│
├── data/
│   ├── raw/                       # Données brutes (DVC)
│   ├── processed/                 # Données transformées (DVC)
│   └── .gitignore
│
├── notebooks/
│   ├── 01_eda.ipynb              # Analyse exploratoire
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experimentation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Chargement données
│   │   └── preprocessing.py      # Feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py              # Pipeline training
│   │   ├── predict.py            # Inférence
│   │   └── evaluate.py           # Évaluation modèle
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app
│   │   ├── schemas.py            # Pydantic models
│   │   └── endpoints.py          # API routes
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── drift_detector.py     # Détection drift
│       └── logger.py             # Logging structuré
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── docker/
│   ├── Dockerfile.training       # Image training
│   ├── Dockerfile.api            # Image API
│   └── docker-compose.yml        # Services locaux
│
├── configs/
│   ├── config.yaml               # Configuration générale
│   ├── model_config.yaml         # Hyperparamètres
│   └── deployment_config.yaml    # Config déploiement
│
├── mlruns/                       # MLflow artifacts (gitignore)
├── models/                       # Modèles sauvegardés
│
├── .dvc/                         # DVC config
├── .dvcignore
├── data.dvc                      # DVC tracking
│
├── .gitignore
├── requirements.txt              # Dépendances Python
├── requirements-dev.txt          # Dépendances développement
├── setup.py                      # Package installation
├── README.md
├── Makefile                      # Commandes utiles
└── pyproject.toml               # Configuration outils

```

## 🚀 Installation

### Prérequis
- Python 3.9+
- Docker & Docker Compose
- Git
- DVC

### Setup Local

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/car-price-mlops.git
cd car-price-mlops

# 2. Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Initialiser DVC
dvc init
dvc remote add -d myremote gdrive://YOUR_GDRIVE_FOLDER_ID

# 5. Télécharger les données
dvc pull

# 6. Setup MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow server --host 0.0.0.0 --port 5000
```

## 💻 Utilisation

### 1. Exploration des Données
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Training du Modèle
```bash
# Avec MLflow tracking
python src/models/train.py --config configs/model_config.yaml

# Voir les résultats
mlflow ui
# Ouvrir http://localhost:5000
```

### 3. Lancer l'API en Local
```bash
# Avec uvicorn
uvicorn src.api.main:app --reload --port 8000

# Ou avec Docker
docker-compose up api

# Documentation API
# http://localhost:8000/docs
```

### 4. Tester l'API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "model": "Corolla",
    "year": 2018,
    "km_driven": 45000,
    "fuel": "Petrol",
    "transmission": "Manual",
    "owner": "First Owner"
  }'
```

## 🔄 Pipeline MLOps

### 1. Versioning Données (DVC)
```bash
# Ajouter de nouvelles données
dvc add data/raw/cars.csv

# Commit et push
git add data/raw/cars.csv.dvc
git commit -m "Update dataset"
dvc push
```

### 2. Tracking Expériences (MLflow)
```python
import mlflow

with mlflow.start_run():
    # Log paramètres
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log métriques
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log modèle
    mlflow.sklearn.log_model(model, "model")
```

### 3. Tests Automatiques
```bash
# Lancer tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=src --cov-report=html
```

### 4. CI/CD (GitHub Actions)
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/
```

### 5. Monitoring
```bash
# Générer rapport de drift
python src/monitoring/drift_detector.py

# Dashboard Streamlit
streamlit run src/monitoring/dashboard.py
```

## 🐳 Déploiement

### Local avec Docker
```bash
# Build images
docker-compose build

# Lancer tous les services
docker-compose up

# Services disponibles:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Monitoring: http://localhost:8501
```

### Production (Render.com)
```bash
# 1. Créer compte Render.com
# 2. Connecter repo GitHub
# 3. Créer Web Service
# 4. Configuration auto depuis Dockerfile.api
```

## 📊 Monitoring

### Métriques Trackées
- **Performance modèle**: RMSE, MAE, R²
- **API**: Latence, throughput, erreurs
- **Data drift**: Distribution features
- **Concept drift**: Dégradation prédictions

### Alertes Configurées
- Drift détecté > seuil
- Latence API > 500ms
- Taux erreur > 5%
- Disponibilité < 99%

## 📅 Roadmap 8 Semaines

| Semaine | Objectifs | Livrables |
|---------|-----------|-----------|
| **S1** | Setup projet | Repo structuré, DVC init |
| **S2** | EDA + Feature Eng | Notebooks, pipeline preprocessing |
| **S3** | Model Development | MLflow tracking, premiers modèles |
| **S4** | Training Pipeline | Pipeline automatisé, Docker, tests |
| **S5** | API Development | FastAPI déployable, documentation |
| **S6** | CI/CD | GitHub Actions, déploiement cloud |
| **S7** | Monitoring | Drift detection, dashboards |
| **S8** | Finalisation | Documentation, démo, présentation |

## 📈 Métriques Actuelles

- **Modèle**: XGBoost
- **RMSE**: 2,450 €
- **R²**: 0.89
- **Latence API**: ~50ms
- **Couverture tests**: 85%

## 🤝 Contribution

### Standards de Code
- Black pour formatting
- Pylint pour linting
- Type hints obligatoires
- Tests pour chaque feature

### Workflow Git
```bash
# Créer branche feature
git checkout -b feature/nom-feature

# Développer et commit
git add .
git commit -m "feat: description"

# Push et PR
git push origin feature/nom-feature
```

## 📝 License

MIT License

## 👥 Équipe

- **Data Engineer**: Pipeline données, DVC
- **ML Engineer**: Modèles, MLflow, API
- **DevOps**: Docker, CI/CD, déploiement

## 📞 Contact

Pour questions: sonia.gharsalli@university.tn

---

**Fait avec ❤️ pour le cours MLOps**