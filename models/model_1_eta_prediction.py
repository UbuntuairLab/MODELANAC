"""
Modèle 1 : Prédiction de retard ETA/ETD
Utilise XGBoost pour prédire l'heure d'arrivée ajustée et les probabilités de retard
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ETAPredictionModel:
    """
    Modèle de prédiction de retard ETA/ETD basé sur XGBoost
    
    Features:
    - vitesse_actuelle: vitesse de l'avion en km/h
    - altitude: altitude actuelle en mètres
    - distance_piste: distance à la piste en km
    - temperature: température en °C
    - vent_vitesse: vitesse du vent en km/h
    - visibilite: visibilité en km
    - pluie: intensité de pluie (0-10)
    - compagnie: code compagnie aérienne
    - retard_historique_compagnie: retard moyen historique (minutes)
    - trafic_approche: nombre d'avions en approche
    - occupation_tarmac: taux d'occupation du tarmac (0-1)
    - heure: heure locale (0-23)
    - jour_semaine: jour de la semaine (0-6)
    - mois: mois (1-12)
    """
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 8, learning_rate: float = 0.1):
        """
        Initialise le modèle ETA
        
        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur maximale des arbres
            learning_rate: Taux d'apprentissage
        """
        self.model_eta = None
        self.model_delay_15 = None
        self.model_delay_30 = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
        # Paramètres XGBoost optimisés pour un hackathon
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'n_jobs': -1,
            'gamma': 0,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42
        }
        
        # Paramètres pour les classifieurs de probabilité de retard
        self.params_classifier = self.params.copy()
        self.params_classifier['objective'] = 'binary:logistic'
        self.params_classifier['eval_metric'] = 'logloss'
    
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
            df['heure'] = pd.to_datetime(df['timestamp']).dt.hour
            df['jour_semaine'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['mois'] = pd.to_datetime(df['timestamp']).dt.month
            df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
            df['est_rush_hour'] = ((df['heure'] >= 7) & (df['heure'] <= 9) | 
                                   (df['heure'] >= 17) & (df['heure'] <= 19)).astype(int)
        
        # Features d'interaction
        if 'distance_piste' in df.columns and 'vitesse_actuelle' in df.columns:
            df['temps_theorique_arrivee'] = (df['distance_piste'] / (df['vitesse_actuelle'] + 1)) * 60  # en minutes
        
        if 'vent_vitesse' in df.columns and 'visibilite' in df.columns:
            df['indice_meteo_adverse'] = (df['vent_vitesse'] / 100) + (1 / (df['visibilite'] + 0.1))
        
        if 'trafic_approche' in df.columns and 'occupation_tarmac' in df.columns:
            df['congestion_totale'] = df['trafic_approche'] * df['occupation_tarmac']
        
        # Features cycliques pour l'heure
        if 'heure' in df.columns:
            df['heure_sin'] = np.sin(2 * np.pi * df['heure'] / 24)
            df['heure_cos'] = np.cos(2 * np.pi * df['heure'] / 24)
        
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
        categorical_cols = ['compagnie'] if 'compagnie' in df.columns else []
        
        # Encoder les variables catégorielles
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Gérer les nouvelles catégories non vues pendant l'entraînement
                df[col + '_encoded'] = df[col].apply(
                    lambda x: self.label_encoders[col].transform([str(x)])[0] 
                    if str(x) in self.label_encoders[col].classes_ 
                    else -1
                )
        
        # Sélectionner les features numériques
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'compagnie', 'retard_reel', 'eta_ajuste', 
                        'retard_15', 'retard_30', 'icao24', 'callsign']]
        
        if fit:
            self.feature_names = feature_cols
        
        X = df[self.feature_names].fillna(0).values
        
        # Normalisation
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def train(self, df: pd.DataFrame, target_col: str = 'retard_reel') -> Dict:
        """
        Entraîne les trois modèles (ETA, retard 15min, retard 30min)
        
        Args:
            df: DataFrame d'entraînement
            target_col: Nom de la colonne cible (retard en minutes)
            
        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        print("🚀 Début de l'entraînement du Modèle 1 - Prédiction ETA/ETD")
        
        # Préparer les features
        X = self.prepare_features(df, fit=True)
        y_eta = df[target_col].values
        
        # Créer les labels pour les probabilités de retard
        y_delay_15 = (df[target_col] > 15).astype(int).values
        y_delay_30 = (df[target_col] > 30).astype(int).values
        
        # Split train/test
        X_train, X_test, y_eta_train, y_eta_test = train_test_split(
            X, y_eta, test_size=0.2, random_state=42
        )
        _, _, y_d15_train, y_d15_test = train_test_split(
            X, y_delay_15, test_size=0.2, random_state=42
        )
        _, _, y_d30_train, y_d30_test = train_test_split(
            X, y_delay_30, test_size=0.2, random_state=42
        )
        
        # 1. Modèle de prédiction ETA (régression)
        print("   📊 Entraînement du modèle de prédiction ETA...")
        self.model_eta = xgb.XGBRegressor(**self.params)
        self.model_eta.fit(
            X_train, y_eta_train,
            eval_set=[(X_test, y_eta_test)],
            verbose=False
        )
        
        # 2. Modèle de probabilité de retard > 15 min
        print("   📊 Entraînement du modèle de probabilité retard > 15min...")
        self.model_delay_15 = xgb.XGBClassifier(**self.params_classifier)
        self.model_delay_15.fit(
            X_train, y_d15_train,
            eval_set=[(X_test, y_d15_test)],
            verbose=False
        )
        
        # 3. Modèle de probabilité de retard > 30 min
        print("   📊 Entraînement du modèle de probabilité retard > 30min...")
        self.model_delay_30 = xgb.XGBClassifier(**self.params_classifier)
        self.model_delay_30.fit(
            X_train, y_d30_train,
            eval_set=[(X_test, y_d30_test)],
            verbose=False
        )
        
        # Évaluation
        metrics = self._evaluate(X_test, y_eta_test, y_d15_test, y_d30_test)
        
        print("✅ Entraînement terminé !")
        print(f"   MAE ETA: {metrics['mae_eta']:.2f} minutes")
        print(f"   R² ETA: {metrics['r2_eta']:.3f}")
        print(f"   Accuracy Retard >15min: {metrics['acc_delay_15']:.2%}")
        print(f"   Accuracy Retard >30min: {metrics['acc_delay_30']:.2%}")
        
        return metrics
    
    def _evaluate(self, X_test, y_eta_test, y_d15_test, y_d30_test) -> Dict:
        """Évalue les performances des modèles"""
        # Prédictions
        y_eta_pred = self.model_eta.predict(X_test)
        y_d15_pred = self.model_delay_15.predict(X_test)
        y_d30_pred = self.model_delay_30.predict(X_test)
        
        # Métriques
        metrics = {
            'mae_eta': mean_absolute_error(y_eta_test, y_eta_pred),
            'rmse_eta': np.sqrt(mean_squared_error(y_eta_test, y_eta_pred)),
            'r2_eta': r2_score(y_eta_test, y_eta_pred),
            'acc_delay_15': (y_d15_pred == y_d15_test).mean(),
            'acc_delay_30': (y_d30_pred == y_d30_test).mean(),
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fait des prédictions sur de nouvelles données
        
        Args:
            df: DataFrame avec les features
            
        Returns:
            Dictionnaire avec les prédictions:
            - eta_ajuste: Retard prédit en minutes
            - proba_delay_15: Probabilité de retard > 15min
            - proba_delay_30: Probabilité de retard > 30min
        """
        if self.model_eta is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez train() d'abord.")
        
        X = self.prepare_features(df, fit=False)
        
        predictions = {
            'eta_ajuste': self.model_eta.predict(X),
            'proba_delay_15': self.model_delay_15.predict_proba(X)[:, 1],
            'proba_delay_30': self.model_delay_30.predict_proba(X)[:, 1],
        }
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Retourne l'importance des features
        
        Args:
            top_n: Nombre de features à retourner
            
        Returns:
            DataFrame avec les features et leur importance
        """
        if self.model_eta is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model_eta.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def save(self, filepath: str):
        """Sauvegarde le modèle complet"""
        model_data = {
            'model_eta': self.model_eta,
            'model_delay_15': self.model_delay_15,
            'model_delay_30': self.model_delay_30,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'params': self.params
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Modèle 1 sauvegardé: {filepath}")
    
    def load(self, filepath: str):
        """Charge un modèle sauvegardé"""
        model_data = joblib.load(filepath)
        self.model_eta = model_data['model_eta']
        self.model_delay_15 = model_data['model_delay_15']
        self.model_delay_30 = model_data['model_delay_30']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.params = model_data['params']
        print(f"📂 Modèle 1 chargé: {filepath}")


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
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='30min'),
        'vitesse_actuelle': np.random.uniform(200, 900, n_samples),
        'altitude': np.random.uniform(0, 12000, n_samples),
        'distance_piste': np.random.uniform(5, 500, n_samples),
        'temperature': np.random.uniform(-10, 35, n_samples),
        'vent_vitesse': np.random.uniform(0, 80, n_samples),
        'visibilite': np.random.uniform(1, 10, n_samples),
        'pluie': np.random.uniform(0, 10, n_samples),
        'compagnie': np.random.choice(['AF', 'BA', 'LH', 'KL', 'IB'], n_samples),
        'retard_historique_compagnie': np.random.uniform(0, 30, n_samples),
        'trafic_approche': np.random.randint(0, 15, n_samples),
        'occupation_tarmac': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Simuler le retard réel (target) basé sur les features
    base_delay = (
        (df['vent_vitesse'] / 10) +
        (df['pluie'] * 2) +
        (df['trafic_approche'] * 3) +
        (df['occupation_tarmac'] * 20) +
        (df['retard_historique_compagnie'] * 0.5) +
        np.random.normal(0, 5, n_samples)
    )
    
    df['retard_reel'] = np.maximum(0, base_delay)
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("MODÈLE 1 : PRÉDICTION ETA/ETD")
    print("=" * 60)
    
    # Créer des données d'exemple
    print("\n📊 Création de données synthétiques...")
    df_train = create_sample_data(n_samples=2000)
    df_test = create_sample_data(n_samples=500)
    
    # Entraîner le modèle
    model = ETAPredictionModel(n_estimators=100, max_depth=6, learning_rate=0.1)
    metrics = model.train(df_train)
    
    # Tester les prédictions
    print("\n🔮 Test de prédiction sur 5 vols...")
    predictions = model.predict(df_test.head(5))
    
    results = pd.DataFrame({
        'Retard_Prédit': predictions['eta_ajuste'],
        'Retard_Réel': df_test.head(5)['retard_reel'].values,
        'Proba_>15min': predictions['proba_delay_15'],
        'Proba_>30min': predictions['proba_delay_30'],
    })
    print(results)
    
    # Feature importance
    print("\n📈 Top 10 des features les plus importantes:")
    print(model.get_feature_importance(top_n=10))
    
    # Sauvegarder le modèle
    model.save('/home/computer-12/Documents/MODELANAC/models/model_1_eta.pkl')
    
    print("\n✅ Modèle 1 prêt à l'emploi !")
