#  Documentation API - Air Traffic ML System

##  Vue d'ensemble

Cette API fournit des prédictions IA en temps réel pour la gestion du trafic aérien, incluant :
- **Modèle 1** : Prédiction ETA/ETD avec probabilités de retard
- **Modèle 2** : Durée d'occupation des emplacements de parking
- **Modèle 3** : Détection de conflits et recommandations de décision

---

##  Déploiement

### URL de base
```
Production: https://tagba-ubuntuairlab.hf.space
Documentation: https://tagba-ubuntuairlab.hf.space/docs
Hugging Face Space: https://huggingface.co/spaces/TAGBA/ubuntuairlab
```

### Installation locale
```bash
git clone https://huggingface.co/spaces/TAGBA/ubuntuairlab
cd ubuntuairlab
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Déploiement Docker
```bash
docker build -t air-traffic-api .
docker run -p 7860:7860 air-traffic-api
```

---

##  Endpoints disponibles

### 1. **GET /** - Information de l'API
Retourne les informations générales sur l'API.

**Réponse:**
```json
{
  "message": "Air Traffic ML API",
  "version": "1.0.0",
  "status": "online",
  "endpoints": {
    "health": "/health",
    "predict": "/predict (POST)",
    "models_info": "/models/info",
    "docs": "/docs"
  }
}
```

---

### 2. **GET /health** - Vérification de l'état
Vérifie que l'API et les modèles sont opérationnels.

**Réponse:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-11T15:30:00",
  "models_loaded": true
}
```

**Codes de statut:**
- `healthy` : Tous les modèles sont chargés et opérationnels
- `models_not_trained` : API en ligne mais modèles non disponibles

---

### 3. **POST /predict** - Prédiction pour un vol

Effectue une prédiction complète pour un vol donné.

#### Paramètres d'entrée (FlightData)

| Paramètre | Type | Description | Exemple |
|-----------|------|-------------|---------|
| `callsign` | string (opt.) | Indicatif d'appel | "AF1234" |
| `icao24` | string (opt.) | Code ICAO24 | "3944ef" |
| `vitesse_actuelle` | float | Vitesse en kt | 250.0 |
| `altitude` | float | Altitude en pieds | 3500.0 |
| `distance_piste` | float | Distance à la piste (km) | 15.5 |
| `temperature` | float | Température (°C) | 22.0 |
| `vent_vitesse` | float | Vitesse du vent (kt) | 12.0 |
| `visibilite` | float | Visibilité (km) | 10.0 |
| `pluie` | float | Intensité de pluie (mm/h) | 0.5 |
| `compagnie` | string | Code compagnie | "Air France" |
| `retard_historique_compagnie` | float | Retard moyen historique (min) | 8.5 |
| `trafic_approche` | int | Nombre d'avions en approche | 5 |
| `occupation_tarmac` | float | Taux d'occupation tarmac (0-1) | 0.65 |
| `type_avion` | string | Type d'appareil | "A320" |
| `historique_occupation_avion` | float | Durée moyenne historique (min) | 45.0 |
| `type_vol` | int | 0=arrivée, 1=départ | 0 |
| `passagers_estimes` | int | Nombre de passagers | 180 |
| `disponibilite_emplacements` | int | Places disponibles | 12 |
| `occupation_actuelle` | float | Taux occupation actuel (0-1) | 0.7 |
| `meteo_score` | float | Score météo (0-1) | 0.85 |
| `trafic_entrant` | int | Vols entrants prévus | 8 |
| `trafic_sortant` | int | Vols sortants prévus | 6 |
| `priorite_vol` | int | 0=normal, 1=prioritaire | 0 |
| `emplacements_futurs_libres` | int | Places à libérer | 3 |
| `timestamp` | string (opt.) | Horodatage ISO 8601 | "2025-12-11T15:30:00" |

#### Exemple de requête

**Python avec requests:**
```python
import requests

url = "https://tagba-ubuntuairlab.hf.space/predict"

flight_data = {
    "callsign": "AF1234",
    "icao24": "3944ef",
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
print(result)
```

**cURL:**
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

**JavaScript (Fetch):**
```javascript
const flightData = {
  vitesse_actuelle: 250.0,
  altitude: 3500.0,
  distance_piste: 15.5,
  temperature: 22.0,
  vent_vitesse: 12.0,
  visibilite: 10.0,
  pluie: 0.5,
  compagnie: "Air France",
  retard_historique_compagnie: 8.5,
  trafic_approche: 5,
  occupation_tarmac: 0.65,
  type_avion: "A320",
  historique_occupation_avion: 45.0,
  type_vol: 0,
  passagers_estimes: 180,
  disponibilite_emplacements: 12,
  occupation_actuelle: 0.7,
  meteo_score: 0.85,
  trafic_entrant: 8,
  trafic_sortant: 6,
  priorite_vol: 0,
  emplacements_futurs_libres: 3
};

fetch('https://tagba-ubuntuairlab.hf.space/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(flightData)
})
.then(response => response.json())
.then(data => console.log(data));
```

#### Réponse

```json
{
  "model_1_eta": {
    "eta_ajuste": 25.34,
    "proba_delay_15": 0.23,
    "proba_delay_30": 0.08,
    "estimation_minutes": 25.34,
    "confiance_retard_15min": "23%",
    "confiance_retard_30min": "8%"
  },
  "model_2_occupation": {
    "temps_occupation_minutes": 42.7,
    "temps_min_minutes": 38.5,
    "temps_max_minutes": 46.9,
    "intervalle_confiance": "95%"
  },
  "model_3_conflict": {
    "risque_conflit": 0,
    "proba_conflit": 0.12,
    "risque_saturation": 1,
    "proba_saturation": 0.78,
    "decision_recommandee": 0,
    "decision_label": "Conserver emplacement",
    "explication": "Faible risque de conflit (12%), saturation élevée (78%). Action recommandée: Conserver emplacement"
  },
  "metadata": {
    "timestamp": "2025-12-11T15:30:00",
    "pipeline_version": "1.0.0"
  }
}
```

#### Structure de la réponse

**model_1_eta** (Prédiction ETA/ETD)
- `eta_ajuste`: Temps estimé d'arrivée ajusté (minutes)
- `proba_delay_15`: Probabilité de retard > 15 min (0-1)
- `proba_delay_30`: Probabilité de retard > 30 min (0-1)

**model_2_occupation** (Durée d'occupation)
- `temps_occupation_minutes`: Durée prédite (minutes)
- `temps_min_minutes`: Borne inférieure (intervalle 95%)
- `temps_max_minutes`: Borne supérieure (intervalle 95%)

**model_3_conflict** (Détection de conflits)
- `risque_conflit`: 0=pas de conflit, 1=conflit détecté
- `proba_conflit`: Probabilité de conflit (0-1)
- `risque_saturation`: 0=capacité OK, 1=saturation
- `proba_saturation`: Probabilité de saturation (0-1)
- `decision_recommandee`: Code de décision (voir ci-dessous)
- `decision_label`: Libellé de la décision
- `explication`: Description détaillée de la situation

**Codes de décision:**
- `0`: Conserver l'emplacement actuel
- `1`: Réaffecter à un autre emplacement
- `2`: Envoyer vers zone militaire/cargo
- `3`: Mise en attente (holding pattern)

---

### 4. **GET /models/info** - Informations sur les modèles

Retourne les détails techniques des modèles ML.

**Réponse:**
```json
{
  "models_trained": true,
  "model_1_eta": {
    "type": "XGBoost Regressor",
    "features": 28,
    "outputs": ["eta_ajuste", "proba_delay_15", "proba_delay_30"]
  },
  "model_2_occupation": {
    "type": "LightGBM Regressor",
    "features": 26,
    "outputs": ["temps_occupation_minutes"]
  },
  "model_3_conflict": {
    "type": "XGBoost Classifier",
    "features": 29,
    "outputs": ["risque_conflit", "risque_saturation", "decision_recommandee"]
  }
}
```

---

##  Intégration dans votre système

### Option 1: Client Python

Créez un client réutilisable:

```python
# air_traffic_client.py
import requests
from typing import Dict, Optional
from datetime import datetime

class AirTrafficAPIClient:
    def __init__(self, base_url: str = "https://tagba-ubuntuairlab.hf.space"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Vérifie l'état de l'API"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_flight(self, flight_data: Dict) -> Dict:
        """Effectue une prédiction pour un vol"""
        response = self.session.post(
            f"{self.base_url}/predict",
            json=flight_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_models_info(self) -> Dict:
        """Récupère les informations sur les modèles"""
        response = self.session.get(f"{self.base_url}/models/info")
        response.raise_for_status()
        return response.json()
    
    def predict_with_retry(self, flight_data: Dict, max_retries: int = 3) -> Optional[Dict]:
        """Prédiction avec retry automatique"""
        for attempt in range(max_retries):
            try:
                return self.predict_flight(flight_data)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Tentative {attempt + 1} échouée, retry...")
        return None

# Utilisation
if __name__ == "__main__":
    client = AirTrafficAPIClient()
    
    # Vérifier l'état
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Prédiction
    flight = {
        "vitesse_actuelle": 250.0,
        "altitude": 3500.0,
        "distance_piste": 15.5,
        # ... autres paramètres
    }
    
    result = client.predict_flight(flight)
    print(f"ETA ajusté: {result['model_1_eta']['eta_ajuste']} minutes")
    print(f"Décision: {result['model_3_conflict']['decision_label']}")
```

### Option 2: Intégration FastAPI existante

Ajoutez l'API comme service externe:

```python
# services/air_traffic_service.py
from fastapi import HTTPException
import httpx
from typing import Dict

class AirTrafficService:
    def __init__(self, api_url: str = "https://tagba-ubuntuairlab.hf.space"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def predict(self, flight_data: Dict) -> Dict:
        try:
            response = await self.client.post(
                f"{self.api_url}/predict",
                json=flight_data
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Service Air Traffic indisponible: {str(e)}"
            )
    
    async def close(self):
        await self.client.aclose()

# Dans votre FastAPI app
from fastapi import FastAPI, Depends

app = FastAPI()
air_traffic = AirTrafficService()

@app.on_event("shutdown")
async def shutdown_event():
    await air_traffic.close()

@app.post("/flights/analyze")
async def analyze_flight(flight_data: Dict):
    result = await air_traffic.predict(flight_data)
    return result
```

### Option 3: Webhook/Event-driven

Pour intégration asynchrone:

```python
# webhook_handler.py
import asyncio
import aiohttp
from typing import List, Dict

async def process_flight_batch(flights: List[Dict], webhook_url: str):
    """Traite un lot de vols et envoie les résultats"""
    api_url = "https://tagba-ubuntuairlab.hf.space/predict"
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for flight in flights:
            task = asyncio.create_task(
                predict_and_notify(session, api_url, flight, webhook_url)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

async def predict_and_notify(session, api_url, flight_data, webhook_url):
    """Prédiction + notification webhook"""
    try:
        # Prédiction
        async with session.post(api_url, json=flight_data) as response:
            prediction = await response.json()
        
        # Notification
        payload = {
            "flight": flight_data.get("callsign"),
            "prediction": prediction,
            "timestamp": prediction["metadata"]["timestamp"]
        }
        
        async with session.post(webhook_url, json=payload) as response:
            return await response.json()
    
    except Exception as e:
        return {"error": str(e), "flight": flight_data.get("callsign")}

# Utilisation
flights = [
    {"callsign": "AF1234", "vitesse_actuelle": 250, ...},
    {"callsign": "LH5678", "vitesse_actuelle": 280, ...},
]

results = asyncio.run(
    process_flight_batch(flights, "https://your-webhook.com/notify")
)
```

---

##  Optimisations et bonnes pratiques

### 1. Connection Pooling
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.3)
adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
session.mount('https://', adapter)

# Utiliser session.post() au lieu de requests.post()
```

### 2. Traitement par batch
```python
async def predict_batch(flights: List[Dict]) -> List[Dict]:
    """Traite plusieurs vols en parallèle"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            session.post(
                "https://tagba-ubuntuairlab.hf.space/predict",
                json=flight
            )
            for flight in flights
        ]
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]
```

### 3. Caching des résultats
```python
from functools import lru_cache
import hashlib
import json

def flight_hash(flight_data: Dict) -> str:
    """Génère un hash unique pour un vol"""
    data_str = json.dumps(flight_data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

@lru_cache(maxsize=1000)
def predict_cached(flight_hash: str, flight_json: str) -> Dict:
    """Prédiction avec cache"""
    flight_data = json.loads(flight_json)
    response = requests.post(
        "https://tagba-ubuntuairlab.hf.space/predict",
        json=flight_data
    )
    return response.json()

# Utilisation
flight = {...}
result = predict_cached(
    flight_hash(flight),
    json.dumps(flight, sort_keys=True)
)
```

---

##  Gestion des erreurs

### Codes HTTP
- `200`: Succès
- `422`: Validation error (paramètres invalides)
- `500`: Erreur serveur
- `503`: Service indisponible (modèles non chargés)

### Exemple de gestion robuste
```python
def predict_with_error_handling(flight_data: Dict) -> Dict:
    try:
        response = requests.post(
            "https://tagba-ubuntuairlab.hf.space/predict",
            json=flight_data,
            timeout=30
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout - API ne répond pas"}
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            return {
                "success": False,
                "error": "Données invalides",
                "details": e.response.json()
            }
        elif e.response.status_code == 503:
            return {"success": False, "error": "Modèles non disponibles"}
        else:
            return {"success": False, "error": f"Erreur HTTP: {e}"}
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Erreur réseau: {e}"}
```

---

##  Monitoring et métriques

### Exemple de logging
```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_with_metrics(flight_data: Dict) -> Dict:
    start_time = time.time()
    
    try:
        response = requests.post(
            "https://tagba-ubuntuairlab.hf.space/predict",
            json=flight_data
        )
        
        latency = time.time() - start_time
        
        logger.info(
            f"Prediction successful - "
            f"Latency: {latency:.2f}s - "
            f"Flight: {flight_data.get('callsign', 'unknown')}"
        )
        
        return response.json()
    
    except Exception as e:
        latency = time.time() - start_time
        logger.error(
            f"Prediction failed - "
            f"Latency: {latency:.2f}s - "
            f"Error: {str(e)}"
        )
        raise
```

---

##  Tests

### Test unitaire
```python
import pytest
from air_traffic_client import AirTrafficAPIClient

def test_health_check():
    client = AirTrafficAPIClient()
    health = client.health_check()
    assert health["status"] in ["healthy", "models_not_trained"]

def test_prediction():
    client = AirTrafficAPIClient()
    
    flight_data = {
        "vitesse_actuelle": 250.0,
        "altitude": 3500.0,
        "distance_piste": 15.5,
        "temperature": 22.0,
        "vent_vitesse": 12.0,
        "visibilite": 10.0,
        "pluie": 0.5,
        "compagnie": "Test Airlines",
        "retard_historique_compagnie": 5.0,
        "trafic_approche": 3,
        "occupation_tarmac": 0.5,
        "type_avion": "A320",
        "historique_occupation_avion": 40.0,
        "type_vol": 0,
        "passagers_estimes": 150,
        "disponibilite_emplacements": 10,
        "occupation_actuelle": 0.6,
        "meteo_score": 0.8,
        "trafic_entrant": 5,
        "trafic_sortant": 4,
        "priorite_vol": 0,
        "emplacements_futurs_libres": 2
    }
    
    result = client.predict_flight(flight_data)
    
    assert "model_1_eta" in result
    assert "model_2_occupation" in result
    assert "model_3_conflict" in result
    assert isinstance(result["model_1_eta"]["eta_ajuste"], float)
```

---

##  Support et ressources

- **Documentation interactive**: https://tagba-ubuntuairlab.hf.space/docs
- **Hugging Face Space**: https://huggingface.co/spaces/TAGBA/ubuntuairlab
- **Username Hugging Face**: TAGBA
- **Space Name**: ubuntuairlab
- **Version API**: 1.0.0

---

##  Changelog

### Version 1.0.0 (2025-12-11)
-  Déploiement initial sur Hugging Face Spaces
-  3 modèles ML opérationnels (XGBoost, LightGBM)
-  Endpoints REST complets
-  Documentation OpenAPI
-  Support CORS pour intégration web
-  Fix libgomp1 pour LightGBM

---

##  Exemple complet d'intégration

```python
#!/usr/bin/env python3
"""
Exemple complet d'intégration de l'API Air Traffic ML
Space Hugging Face: TAGBA/ubuntuairlab
"""
import requests
import json
from datetime import datetime
from typing import Dict, List

class AirTrafficIntegration:
    def __init__(self, api_url: str = "https://tagba-ubuntuairlab.hf.space"):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
    
    def analyze_flight(self, flight_data: Dict) -> Dict:
        """Analyse complète d'un vol"""
        # 1. Vérifier l'API
        health = self.session.get(f"{self.api_url}/health").json()
        if health["status"] != "healthy":
            raise Exception("API non disponible")
        
        # 2. Prédiction
        response = self.session.post(
            f"{self.api_url}/predict",
            json=flight_data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        # 3. Analyse des résultats
        analysis = {
            "flight_id": flight_data.get("callsign", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "eta": {
                "minutes": result["model_1_eta"]["eta_ajuste"],
                "risk_delay_15": result["model_1_eta"]["proba_delay_15"],
                "risk_delay_30": result["model_1_eta"]["proba_delay_30"]
            },
            "occupation": {
                "duration_minutes": result["model_2_occupation"]["temps_occupation_minutes"],
                "range": (
                    result["model_2_occupation"]["temps_min_minutes"],
                    result["model_2_occupation"]["temps_max_minutes"]
                )
            },
            "conflict": {
                "has_conflict": bool(result["model_3_conflict"]["risque_conflit"]),
                "has_saturation": bool(result["model_3_conflict"]["risque_saturation"]),
                "recommended_action": result["model_3_conflict"]["decision_label"],
                "explanation": result["model_3_conflict"]["explication"]
            },
            "alerts": []
        }
        
        # 4. Génération d'alertes
        if result["model_1_eta"]["proba_delay_30"] > 0.3:
            analysis["alerts"].append({
                "level": "warning",
                "message": f"Risque élevé de retard >30min ({result['model_1_eta']['proba_delay_30']*100:.0f}%)"
            })
        
        if result["model_3_conflict"]["risque_conflit"]:
            analysis["alerts"].append({
                "level": "critical",
                "message": "Conflit détecté - Action requise"
            })
        
        if result["model_3_conflict"]["risque_saturation"]:
            analysis["alerts"].append({
                "level": "warning",
                "message": "Saturation de capacité détectée"
            })
        
        return analysis
    
    def process_multiple_flights(self, flights: List[Dict]) -> List[Dict]:
        """Traite plusieurs vols"""
        results = []
        for flight in flights:
            try:
                analysis = self.analyze_flight(flight)
                results.append(analysis)
            except Exception as e:
                results.append({
                    "flight_id": flight.get("callsign", "unknown"),
                    "error": str(e)
                })
        return results

# Utilisation
if __name__ == "__main__":
    integration = AirTrafficIntegration()
    
    # Données de vol exemple
    flight = {
        "callsign": "AF1234",
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
    
    # Analyse
    analysis = integration.analyze_flight(flight)
    
    # Affichage des résultats
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    # Alertes
    if analysis["alerts"]:
        print("\n  ALERTES:")
        for alert in analysis["alerts"]:
            print(f"  [{alert['level'].upper()}] {alert['message']}")
```

---

*Documentation générée le 2025-12-11 pour Air Traffic ML API v1.0.0*
*Hugging Face Space: TAGBA/ubuntuairlab*
*API URL: https://tagba-ubuntuairlab.hf.space*
