"""
Modèle 2 : Prédiction de durée d'occupation d'un emplacement
Utilise LightGBM pour prédire le temps réel d'occupation au sol
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class OccupationPredictionModel:
    """
    Modèle de prédiction de durée d'occupation basé sur LightGBM
    
    Features:
    - type_avion: Type d'avion (A320, B737, A380, etc.)
    - historique_occupation_avion: Temps moyen d'occupation de cet avion
    - compagnie: Code compagnie aérienne
    - temperature: Température en °C
    - vent_vitesse: Vitesse du vent en km/h
    - visibilite: Visibilité en km
    - pluie: Intensité de pluie (0-10)
    - type_vol: Domestique (0) ou International (1)
    - passagers_estimes: Nombre de passagers estimés
    - retard_arrivee: Retard à l'arrivée en minutes
    - heure_arrivee: Heure d'arrivée (0-23)
    - jour_semaine: Jour de la semaine (0-6)
    - charge_bagages: Estimation de la charge de bagages
    """
    
    def __init__(self, n_estimators: int = 150, max_depth: int = 10, learning_rate: float = 0.05):
        """
        Initialise le modèle d'occupation
        
        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur maximale des arbres
            learning_rate: Taux d'apprentissage
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
        # Paramètres LightGBM optimisés
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
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
        
        # Temps de base par type d'avion (en minutes)
        self.base_occupation_times = {
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
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features supplémentaires pour améliorer la prédiction
        
        Args:
            df: DataFrame avec les features de base
            
        Returns:
            DataFrame avec features enrichies
        """
        df = df.copy()
        
        # Features temporelles
        if 'timestamp' in df.columns:
            df['heure_arrivee'] = pd.to_datetime(df['timestamp']).dt.hour
            df['jour_semaine'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['mois'] = pd.to_datetime(df['timestamp']).dt.month
            df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
            df['est_heure_pointe'] = ((df['heure_arrivee'] >= 6) & (df['heure_arrivee'] <= 10) | 
                                      (df['heure_arrivee'] >= 16) & (df['heure_arrivee'] <= 20)).astype(int)
        
        # Temps de base selon le type d'avion
        if 'type_avion' in df.columns:
            df['temps_base_avion'] = df['type_avion'].map(self.base_occupation_times).fillna(50)
        
        # Complexité opérationnelle
        if 'type_vol' in df.columns and 'passagers_estimes' in df.columns:
            df['complexite_operation'] = df['type_vol'] * (df['passagers_estimes'] / 100)
        
        # Impact météo sur le temps d'occupation
        if 'pluie' in df.columns and 'vent_vitesse' in df.columns:
            df['impact_meteo'] = (df['pluie'] / 10) * 5 + (df['vent_vitesse'] / 50) * 3
        
        # Charge de travail estimée
        if 'passagers_estimes' in df.columns:
            df['charge_bagages'] = df['passagers_estimes'] * np.random.uniform(1, 2, len(df))
        
        # Impact du retard sur l'occupation
        if 'retard_arrivee' in df.columns:
            df['retard_normalise'] = np.clip(df['retard_arrivee'] / 60, 0, 2)  # Normaliser entre 0 et 2
            df['urgence_depart'] = (df['retard_arrivee'] > 30).astype(int)  # Si retard > 30min, urgence
        
        # Features cycliques
        if 'heure_arrivee' in df.columns:
            df['heure_sin'] = np.sin(2 * np.pi * df['heure_arrivee'] / 24)
            df['heure_cos'] = np.cos(2 * np.pi * df['heure_arrivee'] / 24)
        
        if 'jour_semaine' in df.columns:
            df['jour_sin'] = np.sin(2 * np.pi * df['jour_semaine'] / 7)
            df['jour_cos'] = np.cos(2 * np.pi * df['jour_semaine'] / 7)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Prépare et encode les features
        
        Args:
            df: DataFrame avec les données
            fit: Si True, fit les encoders et scalers
            
        Returns:
            Array numpy avec les features préparées
        """
        df = self.engineer_features(df)
        
        # Colonnes catégorielles à encoder
        categorical_cols = [col for col in ['type_avion', 'compagnie'] if col in df.columns]
        
        # Encoder les variables catégorielles
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Gérer les nouvelles catégories
                df[col + '_encoded'] = df[col].apply(
                    lambda x: self.label_encoders[col].transform([str(x)])[0] 
                    if str(x) in self.label_encoders[col].classes_ 
                    else -1
                )
        
        # Sélectionner les features numériques
        exclude_cols = ['timestamp', 'type_avion', 'compagnie', 'temps_occupation_reel', 
                       'icao24', 'callsign', 'emplacement']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if fit:
            self.feature_names = feature_cols
        
        X = df[self.feature_names].fillna(0).values
        
        # Normalisation
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def train(self, df: pd.DataFrame, target_col: str = 'temps_occupation_reel') -> Dict:
        """
        Entraîne le modèle de prédiction d'occupation
        
        Args:
            df: DataFrame d'entraînement
            target_col: Nom de la colonne cible (temps d'occupation en minutes)
            
        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        print(" Début de l'entraînement du Modèle 2 - Prédiction d'occupation")
        
        # Préparer les features
        X = self.prepare_features(df, fit=True)
        y = df[target_col].values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entraîner le modèle LightGBM
        print("    Entraînement du modèle LightGBM...")
        self.model = lgb.LGBMRegressor(**self.params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Évaluation
        metrics = self._evaluate(X_test, y_test)
        
        print(" Entraînement terminé !")
        print(f"   MAE: {metrics['mae']:.2f} minutes")
        print(f"   RMSE: {metrics['rmse']:.2f} minutes")
        print(f"   R²: {metrics['r2']:.3f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def _evaluate(self, X_test, y_test) -> Dict:
        """Évalue les performances du modèle"""
        y_pred = self.model.predict(X_test)
        
        # Calculer MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': mape
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prédit le temps d'occupation pour de nouvelles données
        
        Args:
            df: DataFrame avec les features
            
        Returns:
            Array avec les temps d'occupation prédits (en minutes)
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez train() d'abord.")
        
        X = self.prepare_features(df, fit=False)
        predictions = self.model.predict(X)
        
        # Assurer que les prédictions sont positives et réalistes
        predictions = np.clip(predictions, 15, 180)  # Entre 15 min et 3h
        
        return predictions
    
    def predict_with_confidence(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prédit avec intervalle de confiance
        
        Args:
            df: DataFrame avec les features
            
        Returns:
            Dictionnaire avec prédictions et intervalles de confiance
        """
        predictions = self.predict(df)
        
        # Estimer l'incertitude (simplifiée)
        # Dans un cas réel, utiliser des méthodes comme Quantile Regression
        uncertainty = predictions * 0.15  # ±15% d'incertitude
        
        return {
            'temps_occupation': predictions,
            'temps_min': np.maximum(predictions - uncertainty, 15),
            'temps_max': np.minimum(predictions + uncertainty, 180),
            'incertitude': uncertainty
        }
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Retourne l'importance des features
        
        Args:
            top_n: Nombre de features à retourner
            
        Returns:
            DataFrame avec les features et leur importance
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def save(self, filepath: str):
        """Sauvegarde le modèle complet"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'params': self.params,
            'base_occupation_times': self.base_occupation_times
        }
        joblib.dump(model_data, filepath)
        print(f" Modèle 2 sauvegardé: {filepath}")
    
    def load(self, filepath: str):
        """Charge un modèle sauvegardé"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.params = model_data['params']
        self.base_occupation_times = model_data['base_occupation_times']
        print(f" Modèle 2 chargé: {filepath}")


# Fonction utilitaire pour créer des données d'exemple
def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Crée des données synthétiques pour tester le modèle
    
    Args:
        n_samples: Nombre d'échantillons à générer
        
    Returns:
        DataFrame avec des données synthétiques
    """
    np.random.seed(42)
    
    types_avion = ['A320', 'A321', 'A330', 'B737', 'B777', 'B787', 'E190', 'ATR72']
    compagnies = ['AF', 'BA', 'LH', 'KL', 'IB', 'EZY', 'RYR']
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='45min'),
        'type_avion': np.random.choice(types_avion, n_samples),
        'compagnie': np.random.choice(compagnies, n_samples),
        'temperature': np.random.uniform(-5, 35, n_samples),
        'vent_vitesse': np.random.uniform(0, 60, n_samples),
        'visibilite': np.random.uniform(2, 10, n_samples),
        'pluie': np.random.uniform(0, 8, n_samples),
        'type_vol': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),  # 60% domestique
        'passagers_estimes': np.random.randint(50, 350, n_samples),
        'retard_arrivee': np.random.uniform(0, 60, n_samples),
        'historique_occupation_avion': np.random.uniform(35, 80, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Simuler le temps d'occupation réel (target)
    base_times = df['type_avion'].map({
        'A320': 45, 'A321': 50, 'A330': 60, 'B737': 45,
        'B777': 70, 'B787': 60, 'E190': 40, 'ATR72': 30
    })
    
    # Facteurs influençant le temps
    meteo_impact = (df['pluie'] / 10) * 8 + (df['vent_vitesse'] / 60) * 5
    vol_impact = df['type_vol'] * 15  # International prend plus de temps
    passagers_impact = (df['passagers_estimes'] / 100) * 3
    retard_impact = (df['retard_arrivee'] / 60) * 5  # Si en retard, on accélère
    
    df['temps_occupation_reel'] = (
        base_times + 
        meteo_impact + 
        vol_impact + 
        passagers_impact - 
        retard_impact +
        np.random.normal(0, 5, n_samples)
    )
    
    df['temps_occupation_reel'] = np.clip(df['temps_occupation_reel'], 20, 120)
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("MODÈLE 2 : PRÉDICTION DURÉE D'OCCUPATION")
    print("=" * 60)
    
    # Créer des données d'exemple
    print("\n Création de données synthétiques...")
    df_train = create_sample_data(n_samples=2000)
    df_test = create_sample_data(n_samples=500)
    
    # Entraîner le modèle
    model = OccupationPredictionModel(n_estimators=100, max_depth=8, learning_rate=0.05)
    metrics = model.train(df_train)
    
    # Tester les prédictions
    print("\n Test de prédiction sur 5 vols...")
    predictions = model.predict_with_confidence(df_test.head(5))
    
    results = pd.DataFrame({
        'Type_Avion': df_test.head(5)['type_avion'].values,
        'Temps_Réel': df_test.head(5)['temps_occupation_reel'].values,
        'Temps_Prédit': predictions['temps_occupation'],
        'Temps_Min': predictions['temps_min'],
        'Temps_Max': predictions['temps_max'],
    })
    print(results)
    
    # Feature importance
    print("\n Top 10 des features les plus importantes:")
    print(model.get_feature_importance(top_n=10))
    
    # Sauvegarder le modèle
    model.save('/home/computer-12/Documents/MODELANAC/models/model_2_occupation.pkl')
    
    print("\n Modèle 2 prêt à l'emploi !")
