"""
API FastAPI pour le Pipeline ML de Gestion du Trafic Aérien
Expose les 3 modèles via des endpoints REST
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn
import sys
import os

# Ajouter le répertoire models au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from ml_pipeline import AirTrafficMLPipeline

# Créer l'application FastAPI
app = FastAPI(
 title="Air Traffic ML API",
 description="API de prédiction IA pour la gestion du trafic aérien",
 version="1.0.0"
)

# Configuration CORS
app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

# Initialiser le pipeline
pipeline = None

# ==================== Modèles Pydantic ====================

class FlightData(BaseModel):
 """Données d'entrée pour un vol"""
 # Identification
 callsign: Optional[str] = Field(None, description="Indicatif du vol")
 icao24: Optional[str] = Field(None, description="Code ICAO24 de l'avion")
 
 # Modèle 1 - ETA
 vitesse_actuelle: float = Field(..., description="Vitesse actuelle (km/h)")
 altitude: float = Field(..., description="Altitude (mètres)")
 distance_piste: float = Field(..., description="Distance à la piste (km)")
 temperature: float = Field(..., description="Température (°C)")
 vent_vitesse: float = Field(..., description="Vitesse du vent (km/h)")
 visibilite: float = Field(..., description="Visibilité (km)")
 pluie: float = Field(..., description="Intensité de pluie (0-10)")
 compagnie: str = Field(..., description="Code compagnie aérienne")
 retard_historique_compagnie: float = Field(..., description="Retard moyen historique (min)")
 trafic_approche: int = Field(..., description="Nombre d'avions en approche")
 occupation_tarmac: float = Field(..., description="Taux d'occupation tarmac (0-1)")
 
 # Modèle 2 - Occupation
 type_avion: str = Field(..., description="Type d'avion (A320, B737, etc.)")
 historique_occupation_avion: float = Field(..., description="Temps moyen occupation (min)")
 type_vol: int = Field(..., description="Type de vol (0=domestique, 1=international)")
 passagers_estimes: int = Field(..., description="Nombre de passagers estimés")
 
 # Modèle 3 - Conflit
 disponibilite_emplacements: int = Field(..., description="Nombre d'emplacements disponibles")
 occupation_actuelle: float = Field(..., description="Taux d'occupation actuel (0-1)")
 meteo_score: float = Field(..., description="Score météo (0-10, 10=très mauvais)")
 trafic_entrant: int = Field(..., description="Nombre de vols en approche")
 trafic_sortant: int = Field(..., description="Nombre de vols au départ")
 priorite_vol: int = Field(..., description="Priorité du vol (1-5)")
 emplacements_futurs_libres: int = Field(..., description="Emplacements libres dans 30min")
 
 timestamp: Optional[str] = Field(None, description="Timestamp ISO format")
 
 class Config:
 json_schema_extra = {
 "example": {
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
 "emplacements_futurs_libres": 8,
 }
 }


class PredictionResponse(BaseModel):
 """Réponse de prédiction complète"""
 timestamp: str
 vol_info: Dict
 modele_1_eta: Dict
 modele_2_occupation: Dict
 modele_3_decision: Dict
 actions_recommandees: List[str]


class HealthResponse(BaseModel):
 """Réponse du health check"""
 status: str
 timestamp: str
 models_loaded: bool


# ==================== Endpoints ====================

@app.on_event("startup")
async def startup_event():
 """Charge les modèles au démarrage"""
 global pipeline
 
 # Chemins vers les modèles sauvegardés
 models_dir = "/home/computer-12/Documents/MODELANAC/models"
 model1_path = f"{models_dir}/model_1_eta.pkl"
 model2_path = f"{models_dir}/model_2_occupation.pkl"
 model3_path = f"{models_dir}/model_3_conflict.pkl"
 
 # Vérifier si les modèles existent
 models_exist = all(os.path.exists(p) for p in [model1_path, model2_path, model3_path])
 
 if models_exist:
 print(" Chargement des modèles sauvegardés...")
 pipeline = AirTrafficMLPipeline(model1_path, model2_path, model3_path)
 print(" Modèles chargés avec succès !")
 else:
 print(" Modèles non trouvés, initialisation sans modèles pré-entraînés")
 pipeline = AirTrafficMLPipeline()
 print("ℹ Les modèles devront être entraînés via /train")


@app.get("/", response_model=Dict)
async def root():
 """Page d'accueil de l'API"""
 return {
 "message": "Air Traffic ML API",
 "version": "1.0.0",
 "endpoints": {
 "health": "/health",
 "predict": "/predict (POST)",
 "predict_batch": "/predict/batch (POST)",
 "docs": "/docs",
 }
 }


@app.get("/health", response_model=HealthResponse)
async def health_check():
 """Vérifie l'état de l'API et des modèles"""
 models_loaded = (
 pipeline is not None and
 pipeline.model_eta.model_eta is not None and
 pipeline.model_occupation.model is not None and
 pipeline.model_conflict.model_conflict is not None
 )
 
 return HealthResponse(
 status="healthy" if models_loaded else "models_not_trained",
 timestamp=datetime.now().isoformat(),
 models_loaded=models_loaded
 )


@app.post("/predict", response_model=PredictionResponse)
async def predict_flight(flight: FlightData):
 """
 Prédit l'ETA, le temps d'occupation et détecte les conflits pour un vol
 
 Args:
 flight: Données du vol
 
 Returns:
 Prédictions complètes avec recommandations
 """
 if pipeline is None:
 raise HTTPException(status_code=500, detail="Pipeline non initialisé")
 
 if pipeline.model_eta.model_eta is None:
 raise HTTPException(
 status_code=503,
 detail="Modèles non entraînés. Utilisez /train pour entraîner les modèles."
 )
 
 try:
 # Convertir les données Pydantic en dict
 flight_data = flight.dict()
 
 # Ajouter timestamp si absent
 if not flight_data.get('timestamp'):
 flight_data['timestamp'] = datetime.now().isoformat()
 
 # Faire la prédiction
 result = pipeline.predict_full_pipeline(flight_data)
 
 return PredictionResponse(**result)
 
 except Exception as e:
 raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(flights: List[FlightData]):
 """
 Prédit pour plusieurs vols en batch
 
 Args:
 flights: Liste de données de vols
 
 Returns:
 Liste de prédictions
 """
 if pipeline is None:
 raise HTTPException(status_code=500, detail="Pipeline non initialisé")
 
 if pipeline.model_eta.model_eta is None:
 raise HTTPException(
 status_code=503,
 detail="Modèles non entraînés"
 )
 
 try:
 # Convertir en liste de dicts
 flights_data = [f.dict() for f in flights]
 
 # Ajouter timestamps si absents
 for flight_data in flights_data:
 if not flight_data.get('timestamp'):
 flight_data['timestamp'] = datetime.now().isoformat()
 
 # Prédiction batch
 results = pipeline.predict_batch(flights_data)
 
 return {
 "total_flights": len(results),
 "predictions": results
 }
 
 except Exception as e:
 raise HTTPException(status_code=500, detail=f"Erreur de prédiction batch: {str(e)}")


@app.post("/train")
async def train_models(background_tasks: BackgroundTasks):
 """
 Entraîne les modèles sur des données synthétiques (pour la démo)
 En production, cela devrait utiliser de vraies données
 """
 if pipeline is None:
 raise HTTPException(status_code=500, detail="Pipeline non initialisé")
 
 try:
 # Créer des données synthétiques
 from model_1_eta_prediction import create_sample_data as create_data_m1
 from model_2_occupation import create_sample_data as create_data_m2
 from model_3_conflict_detection import create_sample_data as create_data_m3
 
 print(" Création de données d'entraînement...")
 df_m1 = create_data_m1(2000)
 df_m2 = create_data_m2(2000)
 df_m3 = create_data_m3(2000)
 
 # Entraîner
 print(" Début de l'entraînement...")
 metrics = pipeline.train_all_models(df_m1, df_m2, df_m3)
 
 return {
 "status": "success",
 "message": "Modèles entraînés avec succès",
 "metrics": metrics
 }
 
 except Exception as e:
 raise HTTPException(status_code=500, detail=f"Erreur d'entraînement: {str(e)}")


@app.get("/models/info")
async def models_info():
 """Retourne les informations sur les modèles"""
 if pipeline is None:
 raise HTTPException(status_code=500, detail="Pipeline non initialisé")
 
 models_trained = (
 pipeline.model_eta.model_eta is not None and
 pipeline.model_occupation.model is not None and
 pipeline.model_conflict.model_conflict is not None
 )
 
 info = {
 "models_trained": models_trained,
 "model_1_eta": {
 "type": "XGBoost Regressor",
 "features": len(pipeline.model_eta.feature_names) if models_trained else 0,
 "outputs": ["eta_ajuste", "proba_delay_15", "proba_delay_30"]
 },
 "model_2_occupation": {
 "type": "LightGBM Regressor",
 "features": len(pipeline.model_occupation.feature_names) if models_trained else 0,
 "outputs": ["temps_occupation_minutes"]
 },
 "model_3_conflict": {
 "type": "XGBoost Classifier",
 "features": len(pipeline.model_conflict.feature_names) if models_trained else 0,
 "outputs": ["risque_conflit", "risque_saturation", "decision_recommandee"]
 }
 }
 
 return info


# ==================== Lancement ====================

if __name__ == "__main__":
 print("=" * 80)
 print(" " * 25 + " AIR TRAFFIC ML API")
 print("=" * 80)
 print("\nDémarrage du serveur FastAPI...")
 print("Documentation interactive: http://localhost:8000/docs")
 print("API Alternative: http://localhost:8000/redoc")
 print("\n" + "=" * 80 + "\n")
 
 uvicorn.run(
 app,
 host="0.0.0.0",
 port=8000,
 log_level="info"
 )
