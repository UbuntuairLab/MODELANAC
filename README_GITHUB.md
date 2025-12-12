# MODELANAC - Air Traffic ML System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-yellow.svg)](https://lightgbm.readthedocs.io/)

Système de Machine Learning pour la gestion intelligente du trafic aérien utilisant **3 modèles ML intégrés** : XGBoost et LightGBM.

 **API en production** : [https://tagba-ubuntuairlab.hf.space](https://tagba-ubuntuairlab.hf.space) 
 **Documentation API** : [https://tagba-ubuntuairlab.hf.space/docs](https://tagba-ubuntuairlab.hf.space/docs)

---

## Les 3 Modèles ML

### Modèle ETA/ETD Prediction
**Algorithme** : XGBoost Regressor 
**Fonction** : Prédit le temps d'arrivée ajusté avec probabilités de retard 
**Performance** : MAE 4.56 min, R² 0.889 

**Sorties** :
- `eta_ajuste` : Temps d'arrivée ajusté (minutes)
- `proba_delay_15` : Probabilité retard > 15 min
- `proba_delay_30` : Probabilité retard > 30 min

### Modèle Occupation Duration
**Algorithme** : LightGBM Regressor 
**Fonction** : Prédit la durée d'occupation d'un emplacement parking 
**Performance** : MAE 4.19 min, R² 0.881 

**Sorties** :
- `temps_occupation_minutes` : Durée prédite
- Intervalle de confiance à 95%

### Modèle Conflict Detection
**Algorithme** : XGBoost Classifier (Multi-output) 
**Fonction** : Détecte les conflits et recommande des actions 
**Performance** : 96.75% accuracy 

**Sorties** :
- `risque_conflit` : 0/1 (conflit détecté)
- `risque_saturation` : 0/1 (saturation)
- `decision_recommandee` : Action à prendre (0-3)
- `decision_label` : Description de l'action

**Codes de décision** :
- `0` : Conserver l'emplacement actuel
- `1` : Réaffecter à un autre emplacement
- `2` : Envoyer vers zone militaire/cargo
- `3` : Mise en attente (holding pattern)

---

## Architecture

```
Vol entrant → Modèle 1 (ETA) → Modèle 2 (Occupation) → Modèle 3 (Conflits) → Décision
```

**Pipeline séquentiel intégré** où :
1. Le Modèle 1 prédit l'ETA ajusté
2. Le Modèle 2 utilise l'ETA pour prédire la durée d'occupation
3. Le Modèle 3 analyse tout pour détecter les conflits et recommander une action

---

## Installation

### Prérequis
- Python 3.10+
- pip

### Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/UbuntuairLab/MODELANAC.git
cd MODELANAC

# Installer les dépendances
pip install -r requirements.txt

# Entraîner les modèles (optionnel)
python scripts/train_models.py

# Tester le pipeline
python scripts/test_pipeline.py

# Lancer l'API
python api/fastapi_app.py
```

L'API sera accessible sur : `http://localhost:8000`

---

## Utilisation de l'API

### Exemple Python

```python
import requests

# Données de vol
flight_data = {
 "vitesse_actuelle": 250.0,
 "altitude": 3500.0,
 "distance_piste": 15.5,
 "temperature": 22.0,
 "vent_vitesse": 12.0,
 "visibilite": 10.0,
 "pluie": 0.5,
 "compagnie": "Air France",
 "retard_historique_compagnie": 8.5,
 "trafic_approche": 5,
 "occupation_tarmac": 0.65,
 "type_avion": "A320",
 "historique_occupation_avion": 45.0,
 "type_vol": 0,
 "passagers_estimes": 180,
 "disponibilite_emplacements": 12,
 "occupation_actuelle": 0.7,
 "meteo_score": 0.85,
 "trafic_entrant": 8,
 "trafic_sortant": 6,
 "priorite_vol": 0,
 "emplacements_futurs_libres": 3
}

# Appel à l'API
response = requests.post(
 "https://tagba-ubuntuairlab.hf.space/predict",
 json=flight_data
)

result = response.json()

# Résultats des 3 modèles
print(f"ETA ajusté: {result['model_1_eta']['eta_ajuste']} min")
print(f"Occupation: {result['model_2_occupation']['temps_occupation_minutes']} min")
print(f"Décision: {result['model_3_conflict']['decision_label']}")
```

### Endpoints disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Informations sur l'API |
| `/health` | GET | État de santé (vérifie les 3 modèles) |
| `/predict` | POST | Prédiction complète (exécute les 3 modèles) |
| `/models/info` | GET | Détails techniques des modèles |
| `/docs` | GET | Documentation OpenAPI interactive |

---

## Déploiement Docker

```bash
# Build l'image
docker build -t air-traffic-ml .

# Run le container
docker run -p 8000:8000 air-traffic-ml
```

---

## Déploiement sur Hugging Face Spaces

### Configuration

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env et remplir vos credentials
nano .env
```

### Déploiement

```bash
# Exporter les variables d'environnement
export HF_TOKEN=your_token
export HF_USERNAME=your_username

# Exécuter le script de déploiement
./deploy_hf.sh
```

Voir [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) pour plus de détails.

---

## Structure du projet

```
MODELANAC/
 models/ # Modèles ML
 model_1_eta_prediction.py
 model_2_occupation.py
 model_3_conflict_detection.py
 ml_pipeline.py # Pipeline intégré
 model_1_eta.pkl # Modèle 1 entraîné
 model_2_occupation.pkl # Modèle 2 entraîné
 model_3_conflict.pkl # Modèle 3 entraîné
 api/
 fastapi_app.py # API REST
 scripts/
 train_models.py # Entraînement
 test_pipeline.py # Tests
 demo.py # Démonstration
 utils/
 data_collection.py # Génération de données
 config/
 config.py # Configuration
 Dockerfile # Containerisation
 requirements.txt # Dépendances
 API_INTEGRATION_GUIDE.md # Guide d'intégration
 DEPLOY_GUIDE.md # Guide de déploiement
 README.md # Ce fichier
```

---

## Tests

```bash
# Test du pipeline complet
python scripts/test_pipeline.py

# Démonstration avec 5 vols
python scripts/demo.py
```

---

## Documentation

- **[API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md)** : Guide complet d'intégration API avec exemples Python, cURL, JavaScript
- **[DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)** : Instructions de déploiement (Docker, Hugging Face)
- **[MODELS_INTEGRATION.md](MODELS_INTEGRATION.md)** : Documentation technique de l'intégration des 3 modèles

---

## Technologies

- **Python 3.10+** : Langage principal
- **FastAPI** : Framework web moderne
- **XGBoost 3.1.2** : Modèles 1 et 3
- **LightGBM 4.6.0** : Modèle 2
- **scikit-learn 1.6.1** : Preprocessing et métriques
- **NumPy 1.26.4** : Calculs numériques
- **Pandas 2.2.3** : Manipulation de données
- **Pydantic 2.10.6** : Validation de données
- **Docker** : Containerisation
- **Uvicorn** : Serveur ASGI

---

## Contribution

Les contributions sont les bienvenues ! 

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## Licence

MIT License

---

## Auteurs

**UbuntuairLab Organisation**
- GitHub: [@UbuntuairLab](https://github.com/UbuntuairLab)
- Hugging Face: [@TAGBA](https://huggingface.co/TAGBA)

---

## Liens utiles

- **API Production** : https://tagba-ubuntuairlab.hf.space
- **Documentation API** : https://tagba-ubuntuairlab.hf.space/docs
- **Hugging Face Space** : https://huggingface.co/spaces/TAGBA/ubuntuairlab
- **Repository GitHub** : https://github.com/UbuntuairLab/MODELANAC

---

## Quick Start

```bash
# Clone et setup
git clone https://github.com/UbuntuairLab/MODELANAC.git
cd MODELANAC
pip install -r requirements.txt

# Test rapide
curl -X POST "https://tagba-ubuntuairlab.hf.space/predict" \
 -H "Content-Type: application/json" \
 -d '{"vitesse_actuelle": 250, "altitude": 3500, ...}'
```

---

*Système développé pour la gestion intelligente du trafic aérien - 3 modèles ML intégrés*
