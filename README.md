# Air Traffic ML System

Système de Machine Learning pour la gestion intelligente du trafic aérien utilisant 3 modèles ML intégrés : XGBoost et LightGBM.

## Description

Ce projet implémente 3 modèles de Machine Learning pour optimiser la gestion du trafic aérien :

1. **Modèle 1 - Prédiction ETA/ETD** : Prédit les retards d'arrivée/départ avec probabilités de retard >15min et >30min
2. **Modèle 2 - Occupation des emplacements** : Estime la durée d'occupation d'un emplacement de parking avec intervalle de confiance
3. **Modèle 3 - Détection de conflits** : Identifie les conflits potentiels, détecte la saturation et recommande des actions

## API en Production

**URL de base**: https://tagba-ubuntuairlab.hf.space  
**Documentation interactive**: https://tagba-ubuntuairlab.hf.space/docs  
**Hugging Face Space**: https://huggingface.co/spaces/TAGBA/ubuntuairlab  
**Repository GitHub**: https://github.com/UbuntuairLab/MODELANAC

### Endpoints disponibles

- `GET /` - Information de l'API
- `GET /health` - Vérification de l'état des modèles
- `POST /predict` - Prédiction complète pour un vol
- `GET /models/info` - Détails techniques des modèles
- `GET /docs` - Documentation OpenAPI interactive

## Technologies utilisées

- **Python 3.10+** : Langage principal
- **XGBoost 3.1.2** : Modèles 1 (ETA/ETD) et 3 (Détection conflits)
- **LightGBM 4.6.0** : Modèle 2 (Occupation)
- **FastAPI** : Framework API REST moderne
- **scikit-learn 1.6.1** : Preprocessing et métriques
- **Pandas 2.2.3** : Manipulation de données
- **NumPy 1.26.4** : Calculs numériques
- **Pydantic 2.10.6** : Validation de données
- **Uvicorn** : Serveur ASGI

## Performances des Modèles

### Modèle 1 - Prédiction ETA/ETD (XGBoost Regressor)
- MAE: 4.56 minutes
- R²: 0.889
- Accuracy retard >15min: 99.25%

### Modèle 2 - Durée d'occupation (LightGBM Regressor)
- MAE: 4.19 minutes
- R²: 0.881
- MAPE: 7.06%

### Modèle 3 - Détection de conflits (XGBoost Classifier)
- Accuracy conflit: 96.75%
- Accuracy saturation: 56%
- Accuracy décision: 93%

## Installation locale

### 1. Cloner le projet
```bash
git clone https://github.com/UbuntuairLab/MODELANAC.git
cd MODELANAC
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation

### Entraîner les modèles
```bash
python scripts/train_models.py
```

Cela va :
- Générer des données synthétiques d'entraînement
- Entraîner les 3 modèles
- Sauvegarder les modèles dans `models/`
- Afficher les métriques de performance

### Tester le pipeline
```bash
python scripts/test_pipeline.py
```

### Lancer l'API en local
```bash
python api/fastapi_app.py
```

L'API sera accessible sur `http://localhost:8000`
- Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Démonstration
```bash
python scripts/demo.py
```

## Exemple d'utilisation de l'API

### Python
```python
import requests

url = "https://tagba-ubuntuairlab.hf.space/predict"

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

response = requests.post(url, json=flight_data)
result = response.json()

print(f"ETA ajusté: {result['model_1_eta']['eta_ajuste']} minutes")
print(f"Durée occupation: {result['model_2_occupation']['temps_occupation_minutes']} minutes")
print(f"Décision: {result['model_3_conflict']['decision_label']}")
```

### cURL
```bash
curl -X POST "https://tagba-ubuntuairlab.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Structure du projet

```
MODELANAC/
├── models/                      # Modèles ML
│   ├── model_1_eta_prediction.py
│   ├── model_2_occupation.py
│   ├── model_3_conflict_detection.py
│   ├── ml_pipeline.py          # Pipeline intégré
│   ├── model_1_eta.pkl         # Modèle 1 entraîné
│   ├── model_2_occupation.pkl  # Modèle 2 entraîné
│   └── model_3_conflict.pkl    # Modèle 3 entraîné
├── api/
│   └── fastapi_app.py          # API REST
├── scripts/
│   ├── train_models.py         # Entraînement
│   ├── test_pipeline.py        # Tests
│   └── demo.py                 # Démonstration
├── utils/
│   └── data_collection.py      # Génération de données
├── config/
│   └── config.py               # Configuration
├── Dockerfile                   # Containerisation
├── requirements.txt            # Dépendances
├── API_INTEGRATION_GUIDE.md    # Guide d'intégration
├── DEPLOY_GUIDE.md             # Guide de déploiement
└── README.md                   # Ce fichier
```

## Déploiement Docker

```bash
# Build l'image
docker build -t air-traffic-ml .

# Run le container
docker run -p 8000:8000 air-traffic-ml
```

## Documentation

- **API_INTEGRATION_GUIDE.md** : Guide complet d'intégration API avec exemples Python, cURL, JavaScript
- **DEPLOY_GUIDE.md** : Instructions de déploiement (Docker, Hugging Face)
- **MODELS_INTEGRATION.md** : Documentation technique de l'intégration des 3 modèles

## Architecture Pipeline

Le système utilise un pipeline séquentiel intégré :

```
Vol entrant → Modèle 1 (ETA) → Modèle 2 (Occupation) → Modèle 3 (Conflits) → Décision
```

1. **Modèle 1** prédit l'ETA ajusté et les probabilités de retard
2. **Modèle 2** utilise l'ETA pour prédire la durée d'occupation
3. **Modèle 3** analyse tout pour détecter les conflits et recommander une action

### Codes de décision (Modèle 3)
- `0`: Conserver l'emplacement actuel
- `1`: Réaffecter à un autre emplacement
- `2`: Envoyer vers zone militaire/cargo
- `3`: Mise en attente (holding pattern)

## Contribution

Les contributions sont les bienvenues ! 

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## Licence

MIT License

## Auteurs

**UbuntuairLab Organisation**
- GitHub: [@UbuntuairLab](https://github.com/UbuntuairLab)
- Hugging Face: [@TAGBA](https://huggingface.co/TAGBA)

## Liens utiles

- **API Production** : https://tagba-ubuntuairlab.hf.space
- **Documentation API** : https://tagba-ubuntuairlab.hf.space/docs
- **Hugging Face Space** : https://huggingface.co/spaces/TAGBA/ubuntuairlab
- **Repository GitHub** : https://github.com/UbuntuairLab/MODELANAC

---

*Système développé pour la gestion intelligente du trafic aérien - 3 modèles ML intégrés*
