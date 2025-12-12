---
title: Air Traffic ML API
emoji: 
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Air Traffic ML API 

API de prédiction IA pour la gestion du trafic aérien utilisant XGBoost et LightGBM.

**Space Hugging Face**: [TAGBA/ubuntuairlab](https://huggingface.co/spaces/TAGBA/ubuntuairlab)  
**API URL**: https://tagba-ubuntuairlab.hf.space  
**Documentation**: https://tagba-ubuntuairlab.hf.space/docs

##  Fonctionnalités

### Modèle 1: Prédiction ETA/ETD
- Algorithme: **XGBoost Regressor**
- Sorties: Temps ajusté + probabilités de retard (15min, 30min)
- Performance: MAE 4.56 min, R² 0.889

### Modèle 2: Durée d'occupation
- Algorithme: **LightGBM Regressor**
- Sorties: Durée prédite + intervalle de confiance (95%)
- Performance: MAE 4.19 min, R² 0.881

### Modèle 3: Détection de conflits
- Algorithme: **XGBoost Classifier**
- Sorties: Risques (conflit, saturation) + décision recommandée
- Performance: 96.75% accuracy

##  Utilisation rapide

### Exemple Python
```python
import requests

response = requests.post(
    "https://tagba-ubuntuairlab.hf.space/predict",
    json={
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
)

result = response.json()
print(f"ETA: {result['model_1_eta']['eta_ajuste']} minutes")
print(f"Décision: {result['model_3_conflict']['decision_label']}")
```

### Endpoints disponibles
- `GET /` - Informations API
- `GET /health` - État de santé
- `POST /predict` - Prédiction pour un vol
- `GET /models/info` - Détails des modèles
- `GET /docs` - Documentation OpenAPI interactive

##  Documentation complète

Consultez la documentation complète d'intégration: [API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md)

##  Auteur

**Username Hugging Face**: TAGBA  
**Space**: ubuntuairlab
