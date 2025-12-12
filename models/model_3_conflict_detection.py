"""
Modèle 3 : Détection de conflits d'occupation et décision IA
Utilise XGBoost Classifier pour prédire les conflits et recommander des actions
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from typing import Dict, List, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class DecisionType(Enum):
    """Types de décisions possibles"""
    GARDER_EMPLACEMENT = 0
    REAFFECTER_COMMERCIAL = 1
    ENVOYER_MILITAIRE = 2
    ATTENTE_AERIENNE = 3


class ConflictDetectionModel:
    """
    Modèle de détection de conflits et recommandation IA basé sur XGBoost
    
    Features:
    - eta_ajuste: ETA ajusté du vol (depuis Modèle 1)
    - proba_delay_15: Probabilité de retard > 15min (depuis Modèle 1)
    - proba_delay_30: Probabilité de retard > 30min (depuis Modèle 1)
    - temps_occupation_predit: Temps d'occupation prédit (depuis Modèle 2)
    - disponibilite_emplacements: Nombre d'emplacements disponibles
    - occupation_actuelle: Taux d'occupation actuel (0-1)
    - meteo_score: Score météo (0-10, 10 = très mauvais)
    - trafic_entrant: Nombre de vols en approche
    - trafic_sortant: Nombre de vols au départ
    - type_avion: Type d'avion
    - priorite_vol: Priorité du vol (1-5, 5 = très prioritaire)
    - heure: Heure locale (0-23)
    - emplacements_futurs_libres: Emplacements qui seront libres dans 30min
    """
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 10, learning_rate: float = 0.1):
        """
        Initialise le modèle de détection de conflits
        
        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur maximale des arbres
            learning_rate: Taux d'apprentissage
        """
        self.model_conflict = None  # Prédit si conflit
        self.model_saturation = None  # Prédit si saturation
        self.model_decision = None  # Recommande une action
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
        # Paramètres XGBoost pour classification
        self.params_binary = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
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
        
        # Paramètres pour classification multi-classe (décision)
        self.params_multiclass = self.params_binary.copy()
        self.params_multiclass['objective'] = 'multi:softprob'
        self.params_multiclass['eval_metric'] = 'mlogloss'
        self.params_multiclass['num_class'] = 4  # 4 types de décisions
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features supplémentaires pour améliorer la détection
        
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
            df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
            df['est_rush_hour'] = ((df['heure'] >= 7) & (df['heure'] <= 9) | 
                                   (df['heure'] >= 17) & (df['heure'] <= 19)).astype(int)
        
        # Indicateurs de saturation
        if 'disponibilite_emplacements' in df.columns and 'trafic_entrant' in df.columns:
            df['ratio_capacite'] = df['trafic_entrant'] / (df['disponibilite_emplacements'] + 1)
            df['risque_saturation'] = (df['ratio_capacite'] > 0.8).astype(int)
        
        # Score de congestion globale
        if 'occupation_actuelle' in df.columns and 'trafic_entrant' in df.columns:
            df['congestion_score'] = (
                df['occupation_actuelle'] * 0.5 + 
                (df['trafic_entrant'] / 20) * 0.3 +
                (df.get('trafic_sortant', 0) / 20) * 0.2
            )
        
        # Risque temporel (combinaison retard + occupation)
        if 'proba_delay_30' in df.columns and 'temps_occupation_predit' in df.columns:
            df['risque_temporel'] = df['proba_delay_30'] * (df['temps_occupation_predit'] / 60)
        
        # Pression sur les emplacements
        if 'emplacements_futurs_libres' in df.columns and 'trafic_entrant' in df.columns:
            df['pression_emplacements'] = df['trafic_entrant'] - df['emplacements_futurs_libres']
            df['deficit_emplacements'] = (df['pression_emplacements'] > 0).astype(int)
        
        # Impact météo + retard
        if 'meteo_score' in df.columns and 'proba_delay_15' in df.columns:
            df['impact_meteo_retard'] = (df['meteo_score'] / 10) * df['proba_delay_15']
        
        # Urgence opérationnelle
        if 'priorite_vol' in df.columns and 'eta_ajuste' in df.columns:
            df['urgence'] = df['priorite_vol'] * (1 + df['eta_ajuste'] / 60)
        
        # Features cycliques pour l'heure
        if 'heure' in df.columns:
            df['heure_sin'] = np.sin(2 * np.pi * df['heure'] / 24)
            df['heure_cos'] = np.cos(2 * np.pi * df['heure'] / 24)
        
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
        categorical_cols = ['type_avion'] if 'type_avion' in df.columns else []
        
        # Encoder les variables catégorielles
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col + '_encoded'] = df[col].apply(
                    lambda x: self.label_encoders[col].transform([str(x)])[0] 
                    if str(x) in self.label_encoders[col].classes_ 
                    else -1
                )
        
        # Sélectionner les features numériques
        exclude_cols = ['timestamp', 'type_avion', 'conflit', 'saturation', 
                       'decision', 'icao24', 'callsign', 'emplacement']
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
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Entraîne les trois modèles de classification
        
        Args:
            df: DataFrame d'entraînement avec les colonnes:
                - conflit: 0 ou 1 (target pour détection de conflit)
                - saturation: 0 ou 1 (target pour saturation)
                - decision: 0-3 (target pour recommandation)
            
        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        print(" Début de l'entraînement du Modèle 3 - Détection de conflits")
        
        # Préparer les features
        X = self.prepare_features(df, fit=True)
        y_conflict = df['conflit'].values
        y_saturation = df['saturation'].values
        y_decision = df['decision'].values
        
        # Split train/test
        X_train, X_test, y_conf_train, y_conf_test = train_test_split(
            X, y_conflict, test_size=0.2, random_state=42, stratify=y_conflict
        )
        _, _, y_sat_train, y_sat_test = train_test_split(
            X, y_saturation, test_size=0.2, random_state=42, stratify=y_saturation
        )
        _, _, y_dec_train, y_dec_test = train_test_split(
            X, y_decision, test_size=0.2, random_state=42, stratify=y_decision
        )
        
        # 1. Modèle de détection de conflit
        print("    Entraînement du modèle de détection de conflit...")
        self.model_conflict = xgb.XGBClassifier(**self.params_binary)
        self.model_conflict.fit(
            X_train, y_conf_train,
            eval_set=[(X_test, y_conf_test)],
            verbose=False
        )
        
        # 2. Modèle de détection de saturation
        print("    Entraînement du modèle de détection de saturation...")
        self.model_saturation = xgb.XGBClassifier(**self.params_binary)
        self.model_saturation.fit(
            X_train, y_sat_train,
            eval_set=[(X_test, y_sat_test)],
            verbose=False
        )
        
        # 3. Modèle de recommandation de décision
        print("    Entraînement du modèle de recommandation...")
        self.model_decision = xgb.XGBClassifier(**self.params_multiclass)
        self.model_decision.fit(
            X_train, y_dec_train,
            eval_set=[(X_test, y_dec_test)],
            verbose=False
        )
        
        # Évaluation
        metrics = self._evaluate(X_test, y_conf_test, y_sat_test, y_dec_test)
        
        print(" Entraînement terminé !")
        print(f"   Accuracy Conflit: {metrics['acc_conflict']:.2%}")
        print(f"   Accuracy Saturation: {metrics['acc_saturation']:.2%}")
        print(f"   Accuracy Décision: {metrics['acc_decision']:.2%}")
        print(f"   F1-Score Conflit: {metrics['f1_conflict']:.3f}")
        
        return metrics
    
    def _evaluate(self, X_test, y_conf_test, y_sat_test, y_dec_test) -> Dict:
        """Évalue les performances des modèles"""
        # Prédictions
        y_conf_pred = self.model_conflict.predict(X_test)
        y_sat_pred = self.model_saturation.predict(X_test)
        y_dec_pred = self.model_decision.predict(X_test)
        
        # Métriques
        metrics = {
            'acc_conflict': accuracy_score(y_conf_test, y_conf_pred),
            'precision_conflict': precision_score(y_conf_test, y_conf_pred, zero_division=0),
            'recall_conflict': recall_score(y_conf_test, y_conf_pred, zero_division=0),
            'f1_conflict': f1_score(y_conf_test, y_conf_pred, zero_division=0),
            'acc_saturation': accuracy_score(y_sat_test, y_sat_pred),
            'f1_saturation': f1_score(y_sat_test, y_sat_pred, zero_division=0),
            'acc_decision': accuracy_score(y_dec_test, y_dec_pred),
            'f1_decision': f1_score(y_dec_test, y_dec_pred, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fait des prédictions complètes sur de nouvelles données
        
        Args:
            df: DataFrame avec les features
            
        Returns:
            Dictionnaire avec:
            - risque_conflit: Probabilité de conflit (0-1)
            - risque_saturation: Probabilité de saturation (0-1)
            - decision_recommandee: Décision recommandée (0-3)
            - decision_probas: Probabilités pour chaque décision
            - conflit_binaire: Prédiction binaire conflit (0 ou 1)
            - saturation_binaire: Prédiction binaire saturation (0 ou 1)
        """
        if self.model_conflict is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez train() d'abord.")
        
        X = self.prepare_features(df, fit=False)
        
        # Prédictions
        conf_proba = self.model_conflict.predict_proba(X)[:, 1]
        sat_proba = self.model_saturation.predict_proba(X)[:, 1]
        decision = self.model_decision.predict(X)
        decision_probas = self.model_decision.predict_proba(X)
        
        predictions = {
            'risque_conflit': conf_proba,
            'risque_saturation': sat_proba,
            'decision_recommandee': decision,
            'decision_probas': decision_probas,
            'conflit_binaire': (conf_proba > 0.5).astype(int),
            'saturation_binaire': (sat_proba > 0.5).astype(int),
        }
        
        return predictions
    
    def predict_with_explanation(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prédit avec explication de la décision
        
        Args:
            df: DataFrame avec les features
            
        Returns:
            Liste de dictionnaires avec prédictions et explications
        """
        predictions = self.predict(df)
        explanations = []
        
        decision_names = {
            0: "Garder sur emplacement actuel",
            1: "Réaffecter à un autre emplacement commercial",
            2: "Envoyer au parking militaire",
            3: "Mettre en attente aérienne"
        }
        
        for i in range(len(df)):
            explanation = {
                'risque_conflit': float(predictions['risque_conflit'][i]),
                'risque_saturation': float(predictions['risque_saturation'][i]),
                'decision': decision_names[int(predictions['decision_recommandee'][i])],
                'decision_code': int(predictions['decision_recommandee'][i]),
                'confiance_decision': float(predictions['decision_probas'][i].max()),
                'alerte': self._generate_alert(
                    predictions['risque_conflit'][i],
                    predictions['risque_saturation'][i]
                ),
                'priorite': self._calculate_priority(
                    predictions['risque_conflit'][i],
                    predictions['risque_saturation'][i],
                    df.iloc[i].get('priorite_vol', 3)
                )
            }
            explanations.append(explanation)
        
        return explanations
    
    def _generate_alert(self, risk_conflict: float, risk_saturation: float) -> str:
        """Génère un message d'alerte basé sur les risques"""
        if risk_conflict > 0.8 or risk_saturation > 0.8:
            return " ALERTE CRITIQUE - Action immédiate requise"
        elif risk_conflict > 0.6 or risk_saturation > 0.6:
            return " ATTENTION - Surveiller la situation"
        elif risk_conflict > 0.4 or risk_saturation > 0.4:
            return " VIGILANCE - Situation sous contrôle"
        else:
            return " NORMAL - Aucune action nécessaire"
    
    def _calculate_priority(self, risk_conflict: float, risk_saturation: float, 
                           flight_priority: int) -> int:
        """Calcule une priorité globale (1-10)"""
        base_priority = flight_priority * 2
        risk_boost = (risk_conflict + risk_saturation) * 5
        total_priority = min(10, base_priority + risk_boost)
        return int(total_priority)
    
    def get_feature_importance(self, model_type: str = 'conflict', top_n: int = 15) -> pd.DataFrame:
        """
        Retourne l'importance des features pour un modèle spécifique
        
        Args:
            model_type: 'conflict', 'saturation' ou 'decision'
            top_n: Nombre de features à retourner
            
        Returns:
            DataFrame avec les features et leur importance
        """
        if self.model_conflict is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        model_map = {
            'conflict': self.model_conflict,
            'saturation': self.model_saturation,
            'decision': self.model_decision
        }
        
        if model_type not in model_map:
            raise ValueError(f"model_type doit être parmi: {list(model_map.keys())}")
        
        model = model_map[model_type]
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def save(self, filepath: str):
        """Sauvegarde le modèle complet"""
        model_data = {
            'model_conflict': self.model_conflict,
            'model_saturation': self.model_saturation,
            'model_decision': self.model_decision,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'params_binary': self.params_binary,
            'params_multiclass': self.params_multiclass
        }
        joblib.dump(model_data, filepath)
        print(f" Modèle 3 sauvegardé: {filepath}")
    
    def load(self, filepath: str):
        """Charge un modèle sauvegardé"""
        model_data = joblib.load(filepath)
        self.model_conflict = model_data['model_conflict']
        self.model_saturation = model_data['model_saturation']
        self.model_decision = model_data['model_decision']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.params_binary = model_data['params_binary']
        self.params_multiclass = model_data['params_multiclass']
        print(f" Modèle 3 chargé: {filepath}")


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
    
    types_avion = ['A320', 'A321', 'A330', 'B737', 'B777', 'B787', 'E190']
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
        'eta_ajuste': np.random.uniform(0, 60, n_samples),
        'proba_delay_15': np.random.uniform(0, 1, n_samples),
        'proba_delay_30': np.random.uniform(0, 0.8, n_samples),
        'temps_occupation_predit': np.random.uniform(30, 90, n_samples),
        'disponibilite_emplacements': np.random.randint(5, 25, n_samples),
        'occupation_actuelle': np.random.uniform(0.3, 1.0, n_samples),
        'meteo_score': np.random.uniform(0, 10, n_samples),
        'trafic_entrant': np.random.randint(0, 20, n_samples),
        'trafic_sortant': np.random.randint(0, 15, n_samples),
        'type_avion': np.random.choice(types_avion, n_samples),
        'priorite_vol': np.random.randint(1, 6, n_samples),
        'emplacements_futurs_libres': np.random.randint(3, 20, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Simuler les targets
    # Conflit si: peu d'emplacements + beaucoup de trafic + forte occupation
    conflit_score = (
        (1 - df['disponibilite_emplacements'] / 25) * 0.3 +
        (df['trafic_entrant'] / 20) * 0.3 +
        df['occupation_actuelle'] * 0.3 +
        (df['meteo_score'] / 10) * 0.1
    )
    df['conflit'] = (conflit_score > 0.6).astype(int)
    
    # Saturation si: trafic > emplacements disponibles
    saturation_score = (
        (df['trafic_entrant'] / (df['disponibilite_emplacements'] + 1)) * 0.5 +
        df['occupation_actuelle'] * 0.3 +
        (df['temps_occupation_predit'] / 90) * 0.2
    )
    df['saturation'] = (saturation_score > 0.7).astype(int)
    
    # Décision basée sur les conditions
    def assign_decision(row):
        if row['occupation_actuelle'] < 0.6 and row['meteo_score'] < 5:
            return 0  # Garder
        elif row['disponibilite_emplacements'] > 10:
            return 1  # Réaffecter
        elif row['trafic_entrant'] > 15 or row['saturation'] == 1:
            return 2  # Parking militaire
        else:
            return 3  # Attente aérienne
    
    df['decision'] = df.apply(assign_decision, axis=1)
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("MODÈLE 3 : DÉTECTION DE CONFLITS ET DÉCISION IA")
    print("=" * 60)
    
    # Créer des données d'exemple
    print("\n Création de données synthétiques...")
    df_train = create_sample_data(n_samples=2000)
    df_test = create_sample_data(n_samples=500)
    
    # Entraîner le modèle
    model = ConflictDetectionModel(n_estimators=150, max_depth=8, learning_rate=0.1)
    metrics = model.train(df_train)
    
    # Tester les prédictions
    print("\n Test de prédiction avec explications sur 5 vols...")
    explanations = model.predict_with_explanation(df_test.head(5))
    
    for i, exp in enumerate(explanations):
        print(f"\n--- Vol {i+1} ---")
        print(f"Décision: {exp['decision']}")
        print(f"Risque conflit: {exp['risque_conflit']:.2%}")
        print(f"Risque saturation: {exp['risque_saturation']:.2%}")
        print(f"Confiance: {exp['confiance_decision']:.2%}")
        print(f"Priorité: {exp['priorite']}/10")
        print(f"Alerte: {exp['alerte']}")
    
    # Feature importance
    print("\n Top 10 des features pour la détection de conflit:")
    print(model.get_feature_importance(model_type='conflict', top_n=10))
    
    # Sauvegarder le modèle
    model.save('/home/computer-12/Documents/MODELANAC/models/model_3_conflict.pkl')
    
    print("\n Modèle 3 prêt à l'emploi !")
