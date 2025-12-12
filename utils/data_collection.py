"""
Utilitaires pour récupérer les données OpenSky Network
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import time


class OpenSkyDataCollector:
    """
    Collecteur de données depuis l'API OpenSky Network
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialise le collecteur
        
        Args:
            username: Nom d'utilisateur OpenSky (optionnel)
            password: Mot de passe OpenSky (optionnel)
        """
        self.base_url = "https://opensky-network.org/api"
        self.auth = (username, password) if username and password else None
        
    def get_states(
        self,
        lat_min: float = None,
        lat_max: float = None,
        lon_min: float = None,
        lon_max: float = None
    ) -> List[Dict]:
        """
        Récupère l'état actuel des avions dans une zone
        
        Args:
            lat_min: Latitude minimale
            lat_max: Latitude maximale
            lon_min: Longitude minimale
            lon_max: Longitude maximale
            
        Returns:
            Liste de dictionnaires avec les états des avions
        """
        url = f"{self.base_url}/states/all"
        
        params = {}
        if all([lat_min, lat_max, lon_min, lon_max]):
            params = {
                'lamin': lat_min,
                'lamax': lat_max,
                'lomin': lon_min,
                'lomax': lon_max
            }
        
        try:
            response = requests.get(url, params=params, auth=self.auth, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'states' not in data or not data['states']:
                return []
            
            # Convertir en liste de dictionnaires
            states = []
            for state in data['states']:
                states.append({
                    'icao24': state[0],
                    'callsign': state[1].strip() if state[1] else None,
                    'origin_country': state[2],
                    'time_position': state[3],
                    'last_contact': state[4],
                    'longitude': state[5],
                    'latitude': state[6],
                    'baro_altitude': state[7],
                    'on_ground': state[8],
                    'velocity': state[9],  # m/s
                    'true_track': state[10],
                    'vertical_rate': state[11],
                    'sensors': state[12],
                    'geo_altitude': state[13],
                    'squawk': state[14],
                    'spi': state[15],
                    'position_source': state[16]
                })
            
            return states
            
        except requests.exceptions.RequestException as e:
            print(f" Erreur lors de la récupération des données OpenSky: {e}")
            return []
    
    def get_flights_for_airport(
        self,
        icao: str,
        begin: int,
        end: int
    ) -> List[Dict]:
        """
        Récupère les vols arrivant à un aéroport
        
        Args:
            icao: Code ICAO de l'aéroport (ex: LFPG pour CDG)
            begin: Timestamp de début (Unix timestamp)
            end: Timestamp de fin (Unix timestamp)
            
        Returns:
            Liste de vols
        """
        url = f"{self.base_url}/flights/arrival"
        
        params = {
            'airport': icao,
            'begin': begin,
            'end': end
        }
        
        try:
            response = requests.get(url, params=params, auth=self.auth, timeout=30)
            response.raise_for_status()
            
            flights = response.json()
            return flights if flights else []
            
        except requests.exceptions.RequestException as e:
            print(f" Erreur lors de la récupération des vols: {e}")
            return []
    
    def convert_to_dataframe(self, states: List[Dict]) -> pd.DataFrame:
        """
        Convertit les états en DataFrame
        
        Args:
            states: Liste d'états d'avions
            
        Returns:
            DataFrame pandas
        """
        if not states:
            return pd.DataFrame()
        
        df = pd.DataFrame(states)
        
        # Convertir la vitesse de m/s en km/h
        if 'velocity' in df.columns:
            df['vitesse_actuelle'] = df['velocity'] * 3.6  # m/s -> km/h
        
        # Convertir l'altitude en mètres
        if 'baro_altitude' in df.columns:
            df['altitude'] = df['baro_altitude']
        
        # Ajouter timestamp
        df['timestamp'] = datetime.now()
        
        return df
    
    def get_aircraft_info(self, icao24: str) -> Optional[Dict]:
        """
        Récupère les informations sur un avion spécifique
        
        Args:
            icao24: Code ICAO24 de l'avion
            
        Returns:
            Dictionnaire avec les infos ou None
        """
        # Note: Cette API nécessite souvent une authentification
        # Pour la démo, on retourne un exemple
        return {
            'icao24': icao24,
            'type': 'A320',  # À récupérer depuis une base de données réelle
            'operator': 'Unknown'
        }


class WeatherDataCollector:
    """
    Collecteur de données météo depuis Open-Meteo API
    """
    
    def __init__(self):
        """Initialise le collecteur météo"""
        self.base_url = "https://api.open-meteo.com/v1/forecast"
    
    def get_weather(
        self,
        latitude: float,
        longitude: float
    ) -> Dict:
        """
        Récupère les données météo pour une position
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dictionnaire avec les données météo
        """
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'current': 'temperature_2m,windspeed_10m,precipitation,visibility,cloudcover',
            'timezone': 'Europe/Paris'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'current' in data:
                current = data['current']
                return {
                    'temperature': current.get('temperature_2m', 15),
                    'vent_vitesse': current.get('windspeed_10m', 0),
                    'pluie': min(current.get('precipitation', 0) * 2, 10),  # 0-10
                    'visibilite': max(current.get('visibility', 10000) / 1000, 0.1),  # km
                    'meteo_score': self._calculate_weather_score(current),
                    'timestamp': datetime.now()
                }
            
            return self._default_weather()
            
        except requests.exceptions.RequestException as e:
            print(f" Erreur lors de la récupération de la météo: {e}")
            return self._default_weather()
    
    def _calculate_weather_score(self, weather_data: Dict) -> float:
        """
        Calcule un score météo (0=excellent, 10=très mauvais)
        
        Args:
            weather_data: Données météo
            
        Returns:
            Score entre 0 et 10
        """
        score = 0
        
        # Vent (0-50 km/h)
        wind = weather_data.get('windspeed_10m', 0)
        score += min(wind / 10, 4)
        
        # Précipitations (0-10mm)
        precip = weather_data.get('precipitation', 0)
        score += min(precip, 3)
        
        # Visibilité (inversée)
        visibility = weather_data.get('visibility', 10000)
        if visibility < 5000:
            score += 3
        elif visibility < 8000:
            score += 1
        
        return min(score, 10)
    
    def _default_weather(self) -> Dict:
        """Retourne des valeurs météo par défaut"""
        return {
            'temperature': 15,
            'vent_vitesse': 15,
            'pluie': 0,
            'visibilite': 10,
            'meteo_score': 2,
            'timestamp': datetime.now()
        }


def calculate_distance_to_airport(
    lat: float,
    lon: float,
    airport_lat: float,
    airport_lon: float
) -> float:
    """
    Calcule la distance entre un avion et un aéroport (formule de Haversine)
    
    Args:
        lat: Latitude de l'avion
        lon: Longitude de l'avion
        airport_lat: Latitude de l'aéroport
        airport_lon: Longitude de l'aéroport
        
    Returns:
        Distance en km
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convertir en radians
    lat, lon, airport_lat, airport_lon = map(radians, [lat, lon, airport_lat, airport_lon])
    
    # Formule de Haversine
    dlat = airport_lat - lat
    dlon = airport_lon - lon
    a = sin(dlat/2)**2 + cos(lat) * cos(airport_lat) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Rayon de la Terre en km
    
    return c * r


if __name__ == "__main__":
    print("=" * 60)
    print("TEST DES COLLECTEURS DE DONNÉES")
    print("=" * 60)
    
    # Test OpenSky
    print("\n Test OpenSky Network...")
    opensky = OpenSkyDataCollector()
    
    # Zone autour de Paris CDG
    states = opensky.get_states(
        lat_min=48.5,
        lat_max=49.5,
        lon_min=1.5,
        lon_max=3.5
    )
    
    print(f" {len(states)} avions détectés dans la zone")
    
    if states:
        print("\nExemple d'avion:")
        print(f"  ICAO24: {states[0]['icao24']}")
        print(f"  Callsign: {states[0]['callsign']}")
        print(f"  Vitesse: {states[0]['velocity']:.1f} m/s")
        print(f"  Altitude: {states[0]['baro_altitude']:.0f} m" if states[0]['baro_altitude'] else "  Au sol")
    
    # Test Météo
    print("\n Test Open-Meteo...")
    weather = WeatherDataCollector()
    
    # Météo CDG
    weather_data = weather.get_weather(49.0097, 2.5479)
    
    print(f" Météo récupérée:")
    print(f"  Température: {weather_data['temperature']:.1f}°C")
    print(f"  Vent: {weather_data['vent_vitesse']:.1f} km/h")
    print(f"  Visibilité: {weather_data['visibilite']:.1f} km")
    print(f"  Score météo: {weather_data['meteo_score']:.1f}/10")
    
    print("\n Tests terminés !")
