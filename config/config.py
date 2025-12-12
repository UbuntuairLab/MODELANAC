# Configuration pour les modèles ML
# Système de gestion du trafic aérien

# ==================== Chemins ====================
MODELS_DIR = "/home/computer-12/Documents/MODELANAC/models"
DATA_DIR = "/home/computer-12/Documents/MODELANAC/data"
LOGS_DIR = "/home/computer-12/Documents/MODELANAC/logs"

MODEL_1_PATH = f"{MODELS_DIR}/model_1_eta.pkl"
MODEL_2_PATH = f"{MODELS_DIR}/model_2_occupation.pkl"
MODEL_3_PATH = f"{MODELS_DIR}/model_3_conflict.pkl"

# ==================== Paramètres Modèle 1 - ETA ====================
MODEL_1_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'n_jobs': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42
}

# Seuils de retard
DELAY_THRESHOLD_15 = 15  # minutes
DELAY_THRESHOLD_30 = 30  # minutes

# ==================== Paramètres Modèle 2 - Occupation ====================
MODEL_2_PARAMS = {
    'n_estimators': 150,
    'max_depth': 10,
    'learning_rate': 0.05,
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

# Temps d'occupation de base par type d'avion (minutes)
BASE_OCCUPATION_TIMES = {
    'A320': 45,
    'A321': 50,
    'A330': 60,
    'A350': 65,
    'A380': 90,
    'B737': 45,
    'B747': 85,
    'B777': 70,
    'B787': 60,
    'E190': 40,
    'CRJ900': 35,
    'ATR72': 30,
    'DEFAULT': 50  # Pour les types inconnus
}

# Limites de temps d'occupation
MIN_OCCUPATION_TIME = 15  # minutes
MAX_OCCUPATION_TIME = 180  # minutes

# ==================== Paramètres Modèle 3 - Conflit ====================
MODEL_3_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'n_jobs': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42
}

# Seuils de risque
RISK_THRESHOLD_LOW = 0.3
RISK_THRESHOLD_MEDIUM = 0.6
RISK_THRESHOLD_HIGH = 0.8

# Types de décisions
DECISION_TYPES = {
    0: "Garder sur emplacement actuel",
    1: "Réaffecter à un autre emplacement commercial",
    2: "Envoyer au parking militaire",
    3: "Mettre en attente aérienne"
}

# ==================== Configuration API ====================
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True  # Development mode

# CORS
CORS_ORIGINS = ["*"]  # En production, limiter aux domaines autorisés

# ==================== Configuration OpenSky ====================
OPENSKY_USERNAME = None  # Optionnel, pour augmenter les limites de requêtes
OPENSKY_PASSWORD = None

OPENSKY_AREA_BOUNDS = {
    'lat_min': 48.5,  # Sud de Paris
    'lat_max': 49.5,  # Nord de Paris
    'lon_min': 1.5,   # Ouest de Paris
    'lon_max': 3.5    # Est de Paris
}

# ==================== Configuration Météo ====================
# Utiliser Open-Meteo API (gratuit, pas de clé requise)
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# Coordonnées de l'aéroport (exemple: CDG)
AIRPORT_COORDS = {
    'latitude': 49.0097,
    'longitude': 2.5479,
    'name': 'Paris Charles de Gaulle'
}

# Variables météo à récupérer
WEATHER_VARIABLES = [
    'temperature_2m',
    'windspeed_10m',
    'winddirection_10m',
    'precipitation',
    'visibility',
    'cloudcover'
]

# ==================== Configuration Base de données ====================
# Pour stocker les prédictions et l'historique
DATABASE_URL = "sqlite:///./air_traffic_ml.db"

# ==================== Logging ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = f"{LOGS_DIR}/air_traffic_ml.log"

# ==================== Features Engineering ====================
# Features temporelles
HOURS_IN_DAY = 24
DAYS_IN_WEEK = 7
MONTHS_IN_YEAR = 12

# Heures de pointe
RUSH_HOURS = [(7, 9), (17, 19)]

# ==================== Entraînement ====================
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50

# Taille des données synthétiques pour démo
SYNTHETIC_DATA_SIZE = 2000

# ==================== Alertes ====================
ALERT_LEVELS = {
    'CRITIQUE': {
        'emoji': '',
        'message': 'ALERTE CRITIQUE - Action immédiate requise',
        'priority': 10
    },
    'ATTENTION': {
        'emoji': '🟡',
        'message': 'ATTENTION - Surveiller la situation',
        'priority': 7
    },
    'VIGILANCE': {
        'emoji': '🟢',
        'message': 'VIGILANCE - Situation sous contrôle',
        'priority': 4
    },
    'NORMAL': {
        'emoji': '',
        'message': 'NORMAL - Aucune action nécessaire',
        'priority': 1
    }
}

# ==================== Capacité Aéroport ====================
# Exemple pour CDG
AIRPORT_CAPACITY = {
    'total_parking_spots': 150,
    'commercial_spots': 120,
    'military_spots': 30,
    'max_aircraft_per_hour': 60,
    'runways': 4
}

# Seuils de saturation
SATURATION_THRESHOLD_WARNING = 0.7  # 70%
SATURATION_THRESHOLD_CRITICAL = 0.9  # 90%

# ==================== Priorités des vols ====================
FLIGHT_PRIORITIES = {
    1: "Très faible (charter, cargo léger)",
    2: "Faible (vols régionaux)",
    3: "Normale (vols commerciaux standards)",
    4: "Élevée (long-courrier, correspondances)",
    5: "Très élevée (urgence médicale, VIP)"
}

# ==================== Compagnies Aériennes ====================
AIRLINE_CODES = {
    'AF': 'Air France',
    'BA': 'British Airways',
    'LH': 'Lufthansa',
    'KL': 'KLM',
    'IB': 'Iberia',
    'EZY': 'EasyJet',
    'RYR': 'Ryanair',
    'UAE': 'Emirates',
    'QTR': 'Qatar Airways',
    'DAL': 'Delta Air Lines'
}

# ==================== Types d'avions ====================
AIRCRAFT_TYPES = {
    'A320': {'category': 'narrow_body', 'capacity': 180},
    'A321': {'category': 'narrow_body', 'capacity': 220},
    'A330': {'category': 'wide_body', 'capacity': 300},
    'A350': {'category': 'wide_body', 'capacity': 350},
    'A380': {'category': 'wide_body', 'capacity': 550},
    'B737': {'category': 'narrow_body', 'capacity': 175},
    'B747': {'category': 'wide_body', 'capacity': 450},
    'B777': {'category': 'wide_body', 'capacity': 400},
    'B787': {'category': 'wide_body', 'capacity': 330},
    'E190': {'category': 'regional', 'capacity': 100},
    'CRJ900': {'category': 'regional', 'capacity': 90},
    'ATR72': {'category': 'regional', 'capacity': 70},
}
