"""
Pipeline d'Intégration - Orchestration des 3 Modèles IA
Combine les prédictions des modèles 1, 2 et 3 pour une décision globale
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports des modèles
import sys
import os
sys.path.append(os.path.dirname(__file__))

from model_1_eta_prediction import ETAPredictionModel
from model_2_occupation import OccupationPredictionModel
from model_3_conflict_detection import ConflictDetectionModel


class AirTrafficMLPipeline:
    """
    Pipeline complet qui orchestre les 3 modèles ML
    
    Architecture:
    1. Données brutes (OpenSky + Météo + BD) → Préparation Features
    2. Features → Modèle 1 (ETA/ETD) → Prédictions retards
    3. Prédictions Modèle 1 + Features → Modèle 2 (Occupation) → Temps occupation
    4. Prédictions Modèles 1 & 2 + Features → Modèle 3 (Conflit) → Décision finale
    """
    
    def __init__(
        self,
        model1_path: Optional[str] = None,
        model2_path: Optional[str] = None,
        model3_path: Optional[str] = None
    ):
        """
        Initialise le pipeline avec les modèles
        
        Args:
            model1_path: Chemin vers le modèle 1 sauvegardé (optionnel)
            model2_path: Chemin vers le modèle 2 sauvegardé (optionnel)
            model3_path: Chemin vers le modèle 3 sauvegardé (optionnel)
        """
        # Initialiser les modèles
        self.model_eta = ETAPredictionModel()
        self.model_occupation = OccupationPredictionModel()
        self.model_conflict = ConflictDetectionModel()
        
        # Charger les modèles si disponibles
        if model1_path:
            self.model_eta.load(model1_path)
        if model2_path:
            self.model_occupation.load(model2_path)
        if model3_path:
            self.model_conflict.load(model3_path)
    
    def prepare_input_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Transforme les données brutes en DataFrame prêt pour les prédictions
        
        Args:
            raw_data: Dictionnaire avec les données brutes d'un vol
            
        Returns:
            DataFrame formaté
        """
        # Créer un DataFrame à partir des données brutes
        df = pd.DataFrame([raw_data])
        
        # Ajouter timestamp si absent
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        
        return df
    
    def predict_full_pipeline(self, flight_data: Dict) -> Dict:
        """
        Exécute le pipeline complet pour un vol
        
        Args:
            flight_data: Dictionnaire avec toutes les informations du vol
                Exemple:
                {
                    # Pour Modèle 1 (ETA)
                    'vitesse_actuelle': 450,
                    'altitude': 8000,
                    'distance_piste': 50,
                    'temperature': 15,
                    'vent_vitesse': 25,
                    'visibilite': 8,
                    'pluie': 2,
                    'compagnie': 'AF',
                    'retard_historique_compagnie': 12,
                    'trafic_approche': 8,
                    'occupation_tarmac': 0.7,
                    'timestamp': '2024-12-10 14:30:00',
                    
                    # Pour Modèle 2 (Occupation)
                    'type_avion': 'A320',
                    'historique_occupation_avion': 45,
                    'type_vol': 1,  # 0=domestique, 1=international
                    'passagers_estimes': 180,
                    
                    # Pour Modèle 3 (Conflit)
                    'disponibilite_emplacements': 12,
                    'occupation_actuelle': 0.75,
                    'meteo_score': 4,
                    'trafic_entrant': 10,
                    'trafic_sortant': 6,
                    'priorite_vol': 3,
                    'emplacements_futurs_libres': 8,
                }
            
        Returns:
            Dictionnaire complet avec toutes les prédictions et recommandations
        """
        print(f"\n{'='*60}")
        print("🚀 DÉBUT DU PIPELINE IA")
        print(f"{'='*60}")
        
        # Préparer les données
        df_input = self.prepare_input_data(flight_data)
        
        # ========== ÉTAPE 1 : Prédiction ETA/ETD ==========
        print("\n[1/3] 🔮 Modèle 1 : Prédiction ETA/ETD...")
        predictions_eta = self.model_eta.predict(df_input)
        
        eta_ajuste = predictions_eta['eta_ajuste'][0]
        proba_delay_15 = predictions_eta['proba_delay_15'][0]
        proba_delay_30 = predictions_eta['proba_delay_30'][0]
        
        print(f"      ✅ ETA ajusté: +{eta_ajuste:.1f} minutes")
        print(f"      ✅ Probabilité retard >15min: {proba_delay_15:.1%}")
        print(f"      ✅ Probabilité retard >30min: {proba_delay_30:.1%}")
        
        # ========== ÉTAPE 2 : Prédiction Occupation ==========
        print("\n[2/3] 🔮 Modèle 2 : Prédiction durée d'occupation...")
        
        # Ajouter le retard prédit aux données pour le modèle 2
        df_for_occupation = df_input.copy()
        df_for_occupation['retard_arrivee'] = eta_ajuste
        
        predictions_occupation = self.model_occupation.predict_with_confidence(df_for_occupation)
        
        temps_occupation = predictions_occupation['temps_occupation'][0]
        temps_min = predictions_occupation['temps_min'][0]
        temps_max = predictions_occupation['temps_max'][0]
        
        print(f"      ✅ Temps d'occupation: {temps_occupation:.1f} minutes")
        print(f"      ✅ Intervalle: [{temps_min:.1f} - {temps_max:.1f}] minutes")
        
        # ========== ÉTAPE 3 : Détection de conflits et Décision ==========
        print("\n[3/3] 🔮 Modèle 3 : Détection de conflits et décision...")
        
        # Préparer les données pour le modèle 3
        df_for_conflict = df_input.copy()
        df_for_conflict['eta_ajuste'] = eta_ajuste
        df_for_conflict['proba_delay_15'] = proba_delay_15
        df_for_conflict['proba_delay_30'] = proba_delay_30
        df_for_conflict['temps_occupation_predit'] = temps_occupation
        
        predictions_conflict = self.model_conflict.predict_with_explanation(df_for_conflict)
        conflict_result = predictions_conflict[0]
        
        print(f"      ✅ Risque de conflit: {conflict_result['risque_conflit']:.1%}")
        print(f"      ✅ Risque de saturation: {conflict_result['risque_saturation']:.1%}")
        print(f"      ✅ Décision: {conflict_result['decision']}")
        print(f"      ✅ Confiance: {conflict_result['confiance_decision']:.1%}")
        
        # ========== RÉSULTAT FINAL ==========
        print(f"\n{'='*60}")
        print("✨ RÉSULTAT FINAL DU PIPELINE")
        print(f"{'='*60}")
        print(f"{conflict_result['alerte']}")
        print(f"Priorité d'intervention: {conflict_result['priorite']}/10")
        
        # Construire le résultat complet
        result = {
            'timestamp': str(df_input['timestamp'].iloc[0]),
            'vol_info': {
                'compagnie': flight_data.get('compagnie', 'N/A'),
                'type_avion': flight_data.get('type_avion', 'N/A'),
                'callsign': flight_data.get('callsign', 'N/A'),
            },
            'modele_1_eta': {
                'retard_predit_minutes': float(eta_ajuste),
                'probabilite_retard_15min': float(proba_delay_15),
                'probabilite_retard_30min': float(proba_delay_30),
            },
            'modele_2_occupation': {
                'temps_occupation_minutes': float(temps_occupation),
                'temps_min_minutes': float(temps_min),
                'temps_max_minutes': float(temps_max),
                'incertitude_minutes': float(predictions_occupation['incertitude'][0]),
            },
            'modele_3_decision': {
                'risque_conflit': float(conflict_result['risque_conflit']),
                'risque_saturation': float(conflict_result['risque_saturation']),
                'decision_recommandee': conflict_result['decision'],
                'decision_code': int(conflict_result['decision_code']),
                'confiance_decision': float(conflict_result['confiance_decision']),
                'alerte': conflict_result['alerte'],
                'priorite': int(conflict_result['priorite']),
            },
            'actions_recommandees': self._generate_actions(
                conflict_result,
                eta_ajuste,
                temps_occupation,
                flight_data
            )
        }
        
        return result
    
    def predict_batch(self, flights_data: List[Dict]) -> List[Dict]:
        """
        Exécute le pipeline sur plusieurs vols
        
        Args:
            flights_data: Liste de dictionnaires avec les données de vols
            
        Returns:
            Liste de résultats pour chaque vol
        """
        print(f"\n🚀 Pipeline batch : traitement de {len(flights_data)} vols...")
        
        results = []
        for i, flight in enumerate(flights_data, 1):
            print(f"\n--- Vol {i}/{len(flights_data)} ---")
            result = self.predict_full_pipeline(flight)
            results.append(result)
        
        # Analyse globale
        self._analyze_batch_results(results)
        
        return results
    
    def _generate_actions(
        self,
        conflict_result: Dict,
        eta_ajuste: float,
        temps_occupation: float,
        flight_data: Dict
    ) -> List[str]:
        """Génère des actions concrètes à entreprendre"""
        actions = []
        
        # Actions basées sur le risque de conflit
        if conflict_result['risque_conflit'] > 0.7:
            actions.append("🔴 URGENT : Coordonner avec la tour de contrôle")
            actions.append("📞 Contacter la compagnie pour ajustement horaire")
        
        # Actions basées sur le risque de saturation
        if conflict_result['risque_saturation'] > 0.7:
            actions.append("🚨 Préparer emplacements alternatifs")
            actions.append("📋 Vérifier disponibilité parking militaire")
        
        # Actions basées sur la décision
        decision_code = conflict_result['decision_code']
        if decision_code == 1:  # Réaffecter
            actions.append("🔄 Identifier emplacement commercial disponible")
            actions.append("📍 Préparer équipe de guidage au sol")
        elif decision_code == 2:  # Parking militaire
            actions.append("✈️ Coordonner transfert vers zone militaire")
            actions.append("🚌 Organiser transport passagers si nécessaire")
        elif decision_code == 3:  # Attente aérienne
            actions.append("🔄 Mettre en circuit d'attente")
            actions.append("⏱️ Estimer nouvelle heure d'atterrissage")
        
        # Actions basées sur le retard
        if eta_ajuste > 30:
            actions.append(f"⏰ Informer passagers : retard estimé {eta_ajuste:.0f} min")
            actions.append("🔗 Vérifier correspondances affectées")
        
        # Actions basées sur l'occupation
        if temps_occupation > 75:
            actions.append(f"⚠️ Prévoir occupation longue ({temps_occupation:.0f} min)")
            actions.append("👥 Mobiliser équipe de maintenance complète")
        
        if not actions:
            actions.append("✅ Aucune action spécifique requise")
        
        return actions
    
    def _analyze_batch_results(self, results: List[Dict]):
        """Analyse globale des résultats batch"""
        print(f"\n{'='*60}")
        print("📊 ANALYSE GLOBALE DES VOLS")
        print(f"{'='*60}")
        
        total = len(results)
        conflits = sum(1 for r in results if r['modele_3_decision']['risque_conflit'] > 0.5)
        saturations = sum(1 for r in results if r['modele_3_decision']['risque_saturation'] > 0.5)
        retards_15 = sum(1 for r in results if r['modele_1_eta']['probabilite_retard_15min'] > 0.5)
        
        print(f"Total vols analysés: {total}")
        print(f"Vols avec risque de conflit: {conflits} ({conflits/total:.1%})")
        print(f"Vols avec risque de saturation: {saturations} ({saturations/total:.1%})")
        print(f"Vols avec retard probable >15min: {retards_15} ({retards_15/total:.1%})")
        
        # Distribution des décisions
        decisions_count = {}
        for r in results:
            decision = r['modele_3_decision']['decision_recommandee']
            decisions_count[decision] = decisions_count.get(decision, 0) + 1
        
        print("\nDistribution des décisions:")
        for decision, count in decisions_count.items():
            print(f"  - {decision}: {count} ({count/total:.1%})")
    
    def train_all_models(
        self,
        df_model1: pd.DataFrame,
        df_model2: pd.DataFrame,
        df_model3: pd.DataFrame,
        save_dir: str = '/home/computer-12/Documents/MODELANAC/models/'
    ):
        """
        Entraîne les 3 modèles et sauvegarde
        
        Args:
            df_model1: Données pour modèle 1 (ETA)
            df_model2: Données pour modèle 2 (Occupation)
            df_model3: Données pour modèle 3 (Conflit)
            save_dir: Répertoire de sauvegarde
        """
        print(f"\n{'='*60}")
        print("🎓 ENTRAÎNEMENT COMPLET DU PIPELINE")
        print(f"{'='*60}")
        
        # Entraîner modèle 1
        print("\n[1/3] Entraînement Modèle 1...")
        metrics1 = self.model_eta.train(df_model1)
        self.model_eta.save(f"{save_dir}/model_1_eta.pkl")
        
        # Entraîner modèle 2
        print("\n[2/3] Entraînement Modèle 2...")
        metrics2 = self.model_occupation.train(df_model2)
        self.model_occupation.save(f"{save_dir}/model_2_occupation.pkl")
        
        # Entraîner modèle 3
        print("\n[3/3] Entraînement Modèle 3...")
        metrics3 = self.model_conflict.train(df_model3)
        self.model_conflict.save(f"{save_dir}/model_3_conflict.pkl")
        
        print(f"\n{'='*60}")
        print("✅ TOUS LES MODÈLES SONT ENTRAÎNÉS ET SAUVEGARDÉS")
        print(f"{'='*60}")
        
        return {
            'model1_metrics': metrics1,
            'model2_metrics': metrics2,
            'model3_metrics': metrics3
        }


if __name__ == "__main__":
    print("=" * 80)
    print(" " * 20 + "PIPELINE ML - GESTION TRAFIC AÉRIEN")
    print("=" * 80)
    
    # Exemple d'utilisation du pipeline
    
    # Créer le pipeline
    pipeline = AirTrafficMLPipeline()
    
    # Créer des données de test pour l'entraînement
    print("\n📊 Création de données synthétiques pour l'entraînement...")
    
    from model_1_eta_prediction import create_sample_data as create_data_m1
    from model_2_occupation import create_sample_data as create_data_m2
    from model_3_conflict_detection import create_sample_data as create_data_m3
    
    df_m1 = create_data_m1(2000)
    df_m2 = create_data_m2(2000)
    df_m3 = create_data_m3(2000)
    
    # Entraîner tous les modèles
    all_metrics = pipeline.train_all_models(df_m1, df_m2, df_m3)
    
    # Exemple de prédiction sur un vol
    print("\n" + "=" * 80)
    print(" " * 25 + "EXEMPLE DE PRÉDICTION")
    print("=" * 80)
    
    flight_example = {
        # Modèle 1
        'vitesse_actuelle': 420,
        'altitude': 5000,
        'distance_piste': 35,
        'temperature': 18,
        'vent_vitesse': 35,
        'visibilite': 6,
        'pluie': 4,
        'compagnie': 'AF',
        'retard_historique_compagnie': 15,
        'trafic_approche': 12,
        'occupation_tarmac': 0.85,
        'timestamp': '2024-12-10 15:45:00',
        'callsign': 'AFR1234',
        
        # Modèle 2
        'type_avion': 'A320',
        'historique_occupation_avion': 48,
        'type_vol': 1,
        'passagers_estimes': 165,
        
        # Modèle 3
        'disponibilite_emplacements': 8,
        'occupation_actuelle': 0.85,
        'meteo_score': 6,
        'trafic_entrant': 14,
        'trafic_sortant': 7,
        'priorite_vol': 4,
        'emplacements_futurs_libres': 5,
    }
    
    # Prédiction
    result = pipeline.predict_full_pipeline(flight_example)
    
    # Afficher les actions recommandées
    print("\n📋 ACTIONS RECOMMANDÉES:")
    for i, action in enumerate(result['actions_recommandees'], 1):
        print(f"   {i}. {action}")
    
    print("\n✅ Pipeline opérationnel !")
