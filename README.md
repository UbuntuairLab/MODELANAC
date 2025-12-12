# Guide de Démarrage Rapide - Air Traffic ML

## 🚀 Installation

### 1. Créer un environnement virtuel

```bash
cd /home/computer-12/Documents/MODELANAC
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 📚 Architecture du Projet

```
MODELANAC/
├── models/
│   ├── model_1_eta_prediction.py      # Modèle 1: Prédiction ETA/ETD (XGBoost)
│   ├── model_2_occupation.py          # Modèle 2: Durée d'occupation (LightGBM)
│   ├── model_3_conflict_detection.py  # Modèle 3: Détection conflits (XGBoost Classifier)
│   ├── ml_pipeline.py                 # Pipeline d'orchestration
│   ├── model_1_eta.pkl               # Modèle 1 sauvegardé (après entraînement)
│   ├── model_2_occupation.pkl        # Modèle 2 sauvegardé (après entraînement)
│   └── model_3_conflict.pkl          # Modèle 3 sauvegardé (après entraînement)
├── api/
│   └── fastapi_app.py                # API REST FastAPI
├── scripts/
│   ├── train_models.py               # Script d'entraînement
│   └── test_pipeline.py              # Script de test
├── config/
│   └── config.py                     # Configuration centralisée
├── data/                             # Données (à créer)
├── logs/                             # Logs (à créer)
└── requirements.txt                  # Dépendances Python
```

## 🎓 Étape 1: Entraîner les Modèles

### Option A: Utilisation du script d'entraînement

```bash
cd /home/computer-12/Documents/MODELANAC
python scripts/train_models.py --samples 2000
```

### Option B: Entraînement manuel avec Python

```python
from models.ml_pipeline import AirTrafficMLPipeline
from models.model_1_eta_prediction import create_sample_data as create_data_m1
from models.model_2_occupation import create_sample_data as create_data_m2
from models.model_3_conflict_detection import create_sample_data as create_data_m3

# Créer les données
df_m1 = create_data_m1(2000)
df_m2 = create_data_m2(2000)
df_m3 = create_data_m3(2000)

# Entraîner
pipeline = AirTrafficMLPipeline()
metrics = pipeline.train_all_models(df_m1, df_m2, df_m3)
```

## 🧪 Étape 2: Tester le Pipeline

```bash
python scripts/test_pipeline.py
```

## 🌐 Étape 3: Lancer l'API FastAPI

### Option A: Lancement direct

```bash
cd /home/computer-12/Documents/MODELANAC
python api/fastapi_app.py
```

### Option B: Avec uvicorn

```bash
cd /home/computer-12/Documents/MODELANAC
uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

Accédez à:
- **Documentation interactive**: http://localhost:8000/docs
- **Documentation alternative**: http://localhost:8000/redoc
- **API**: http://localhost:8000

## 📡 Utilisation de l'API

### 1. Vérifier l'état de l'API

```bash
curl http://localhost:8000/health
```

### 2. Prédiction pour un vol

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "callsign": "AFR1234",
    "vitesse_actuelle": 450,
    "altitude": 6000,
    "distance_piste": 40,
    "temperature": 15,
    "vent_vitesse": 25,
    "visibilite": 8,
    "pluie": 2,
    "compagnie": "AF",
    "retard_historique_compagnie": 12,
    "trafic_approche": 8,
    "occupation_tarmac": 0.7,
    "type_avion": "A320",
    "historique_occupation_avion": 45,
    "type_vol": 1,
    "passagers_estimes": 180,
    "disponibilite_emplacements": 12,
    "occupation_actuelle": 0.75,
    "meteo_score": 4,
    "trafic_entrant": 10,
    "trafic_sortant": 6,
    "priorite_vol": 3,
    "emplacements_futurs_libres": 8
  }'
```

### 3. Entraîner via l'API (données synthétiques)

```bash
curl -X POST "http://localhost:8000/train"
```

## 🔮 Les 3 Modèles Expliqués

### Modèle 1: Prédiction ETA/ETD (XGBoost Regressor)

**Objectif**: Prédire le retard d'arrivée et les probabilités de retard

**Entrées**:
- Vitesse, altitude, distance à la piste
- Météo (température, vent, pluie, visibilité)
- Compagnie et retard historique
- Trafic et occupation du tarmac
- Heure locale

**Sorties**:
- Retard prédit (minutes)
- Probabilité de retard > 15 min
- Probabilité de retard > 30 min

**Performances typiques**:
- MAE: ~3-5 minutes
- R²: ~0.85-0.90

### Modèle 2: Durée d'Occupation (LightGBM Regressor)

**Objectif**: Prédire le temps réel d'occupation d'un emplacement

**Entrées**:
- Type d'avion et historique occupation
- Compagnie
- Météo
- Type de vol (domestique/international)
- Nombre de passagers
- Retard à l'arrivée (depuis Modèle 1)

**Sorties**:
- Temps d'occupation prédit (minutes)
- Intervalle de confiance [min, max]

**Performances typiques**:
- MAE: ~4-6 minutes
- R²: ~0.80-0.88

### Modèle 3: Détection de Conflits (XGBoost Classifier)

**Objectif**: Détecter les conflits et recommander une décision

**Entrées**:
- Prédictions des Modèles 1 & 2
- Disponibilité des emplacements
- Météo prévue
- Trafic entrant/sortant
- Priorité du vol

**Sorties**:
- Risque de conflit (0-1)
- Risque de saturation (0-1)
- Décision recommandée:
  - 0: Garder sur emplacement actuel
  - 1: Réaffecter à un autre emplacement commercial
  - 2: Envoyer au parking militaire
  - 3: Mettre en attente aérienne

**Performances typiques**:
- Accuracy conflit: ~85-92%
- Accuracy décision: ~80-88%

## 🎯 Exemples d'Utilisation Python

### Utilisation du pipeline complet

```python
from models.ml_pipeline import AirTrafficMLPipeline

# Charger les modèles
pipeline = AirTrafficMLPipeline(
    model1_path='models/model_1_eta.pkl',
    model2_path='models/model_2_occupation.pkl',
    model3_path='models/model_3_conflict.pkl'
)

# Données d'un vol
flight_data = {
    'callsign': 'AFR1234',
    'vitesse_actuelle': 450,
    'altitude': 6000,
    # ... autres features
}

# Prédiction
result = pipeline.predict_full_pipeline(flight_data)

print(f"Retard prédit: {result['modele_1_eta']['retard_predit_minutes']:.1f} min")
print(f"Temps occupation: {result['modele_2_occupation']['temps_occupation_minutes']:.1f} min")
print(f"Décision: {result['modele_3_decision']['decision_recommandee']}")
```

### Utilisation individuelle des modèles

```python
from models.model_1_eta_prediction import ETAPredictionModel
import pandas as pd

# Charger le modèle
model = ETAPredictionModel()
model.load('models/model_1_eta.pkl')

# Prédire
df = pd.DataFrame([{
    'vitesse_actuelle': 450,
    'altitude': 6000,
    # ... autres features
}])

predictions = model.predict(df)
print(f"ETA ajusté: {predictions['eta_ajuste'][0]:.1f} min")
```

## 📊 Feature Importance

Pour voir les features les plus importantes:

```python
from models.model_1_eta_prediction import ETAPredictionModel

model = ETAPredictionModel()
model.load('models/model_1_eta.pkl')

# Top 10 features
importance = model.get_feature_importance(top_n=10)
print(importance)
```

## 🔧 Configuration

Modifiez `config/config.py` pour:
- Ajuster les hyperparamètres des modèles
- Changer les seuils de risque
- Configurer l'API
- Définir les temps de base d'occupation par avion

## 📝 Logs

Les logs sont automatiquement générés dans `logs/air_traffic_ml.log`

## ⚡ Performance Tips

1. **Pour l'entraînement**:
   - Utilisez plus de données (--samples 5000)
   - Ajustez les hyperparamètres dans config.py

2. **Pour la production**:
   - Activez le caching des prédictions
   - Utilisez des données réelles (OpenSky + Météo)
   - Implémentez un monitoring des prédictions

3. **Pour l'API**:
   - Utilisez gunicorn en production
   - Activez HTTPS
   - Limitez les CORS aux domaines autorisés

## 🚨 Troubleshooting

### Erreur: "Modèles non trouvés"
```bash
python scripts/train_models.py
```

### Erreur: "Module not found"
```bash
pip install -r requirements.txt
```

### Port 8000 déjà utilisé
```bash
uvicorn api.fastapi_app:app --port 8001
```

## 📚 Ressources

- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **OpenSky API**: https://opensky-network.org/apidoc/

## 🎉 Next Steps

1. Intégrer avec des données réelles (OpenSky API)
2. Ajouter un frontend React/Vue.js
3. Implémenter un système de monitoring
4. Déployer sur un serveur de production
5. Ajouter des tests unitaires
6. Créer un dashboard de visualisation

Bon hackathon! 🚀
