"""
Script de test du pipeline complet
"""

import sys
import os
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.ml_pipeline import AirTrafficMLPipeline
from config.config import MODELS_DIR


def test_pipeline():
    """
    Test complet du pipeline avec des exemples
    """
    print("=" * 80)
    print(" " * 25 + " TEST DU PIPELINE ML")
    print("=" * 80)
    
    # Charger les modèles
    print("\n Chargement des modèles...")
    
    model1_path = f"{MODELS_DIR}/model_1_eta.pkl"
    model2_path = f"{MODELS_DIR}/model_2_occupation.pkl"
    model3_path = f"{MODELS_DIR}/model_3_conflict.pkl"
    
    # Vérifier que les modèles existent
    models_exist = all(os.path.exists(p) for p in [model1_path, model2_path, model3_path])
    
    if not models_exist:
        print(" Erreur: Les modèles n'ont pas été trouvés !")
        print("   Exécutez d'abord: python scripts/train_models.py")
        return
    
    # Créer le pipeline
    pipeline = AirTrafficMLPipeline(model1_path, model2_path, model3_path)
    print(" Modèles chargés avec succès !\n")
    
    # ========== Scénario 1: Vol normal ==========
    print("\n" + "=" * 80)
    print(" SCÉNARIO 1 : Vol commercial normal (Air France A320)")
    print("=" * 80)
    
    flight_normal = {
        'callsign': 'AFR1234',
        'compagnie': 'AF',
        'type_avion': 'A320',
        'vitesse_actuelle': 450,
        'altitude': 5000,
        'distance_piste': 30,
        'temperature': 18,
        'vent_vitesse': 15,
        'visibilite': 10,
        'pluie': 0,
        'retard_historique_compagnie': 8,
        'trafic_approche': 5,
        'occupation_tarmac': 0.5,
        'historique_occupation_avion': 45,
        'type_vol': 1,
        'passagers_estimes': 165,
        'disponibilite_emplacements': 18,
        'occupation_actuelle': 0.5,
        'meteo_score': 2,
        'trafic_entrant': 6,
        'trafic_sortant': 5,
        'priorite_vol': 3,
        'emplacements_futurs_libres': 15,
        'timestamp': datetime.now().isoformat()
    }
    
    result1 = pipeline.predict_full_pipeline(flight_normal)
    print_result_summary(result1)
    
    # ========== Scénario 2: Situation critique ==========
    print("\n" + "=" * 80)
    print(" SCÉNARIO 2 : Situation critique (Météo difficile + Saturation)")
    print("=" * 80)
    
    flight_critical = {
        'callsign': 'BAW456',
        'compagnie': 'BA',
        'type_avion': 'A380',
        'vitesse_actuelle': 380,
        'altitude': 3000,
        'distance_piste': 15,
        'temperature': 5,
        'vent_vitesse': 55,
        'visibilite': 3,
        'pluie': 8,
        'retard_historique_compagnie': 25,
        'trafic_approche': 18,
        'occupation_tarmac': 0.95,
        'historique_occupation_avion': 90,
        'type_vol': 1,
        'passagers_estimes': 520,
        'disponibilite_emplacements': 4,
        'occupation_actuelle': 0.95,
        'meteo_score': 9,
        'trafic_entrant': 20,
        'trafic_sortant': 3,
        'priorite_vol': 5,
        'emplacements_futurs_libres': 2,
        'timestamp': datetime.now().isoformat()
    }
    
    result2 = pipeline.predict_full_pipeline(flight_critical)
    print_result_summary(result2)
    
    # ========== Scénario 3: Vol prioritaire ==========
    print("\n" + "=" * 80)
    print(" SCÉNARIO 3 : Vol prioritaire (Urgence médicale)")
    print("=" * 80)
    
    flight_priority = {
        'callsign': 'MED999',
        'compagnie': 'AF',
        'type_avion': 'B737',
        'vitesse_actuelle': 500,
        'altitude': 8000,
        'distance_piste': 60,
        'temperature': 15,
        'vent_vitesse': 20,
        'visibilite': 8,
        'pluie': 1,
        'retard_historique_compagnie': 5,
        'trafic_approche': 10,
        'occupation_tarmac': 0.7,
        'historique_occupation_avion': 40,
        'type_vol': 0,
        'passagers_estimes': 100,
        'disponibilite_emplacements': 10,
        'occupation_actuelle': 0.7,
        'meteo_score': 3,
        'trafic_entrant': 12,
        'trafic_sortant': 8,
        'priorite_vol': 5,  # Priorité maximale
        'emplacements_futurs_libres': 8,
        'timestamp': datetime.now().isoformat()
    }
    
    result3 = pipeline.predict_full_pipeline(flight_priority)
    print_result_summary(result3)
    
    print("\n" + "=" * 80)
    print(" TESTS TERMINÉS AVEC SUCCÈS !")
    print("=" * 80)


def print_result_summary(result: dict):
    """Affiche un résumé du résultat"""
    print(f"\n Résultat pour {result['vol_info']['callsign']}:")
    print(f"   Type: {result['vol_info']['type_avion']} - {result['vol_info']['compagnie']}")
    print(f"\n    Prédictions:")
    print(f"      Retard estimé: +{result['modele_1_eta']['retard_predit_minutes']:.1f} min")
    print(f"      Temps occupation: {result['modele_2_occupation']['temps_occupation_minutes']:.1f} min")
    print(f"      Risque conflit: {result['modele_3_decision']['risque_conflit']:.1%}")
    print(f"      Risque saturation: {result['modele_3_decision']['risque_saturation']:.1%}")
    print(f"\n    Décision: {result['modele_3_decision']['decision_recommandee']}")
    print(f"    {result['modele_3_decision']['alerte']}")
    print(f"    Priorité: {result['modele_3_decision']['priorite']}/10")
    print(f"\n    Actions recommandées:")
    for i, action in enumerate(result['actions_recommandees'], 1):
        print(f"      {i}. {action}")


if __name__ == "__main__":
    test_pipeline()
