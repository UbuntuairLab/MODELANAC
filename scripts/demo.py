"""
Script de démonstration complet du système
Simule un scénario réel avec plusieurs vols
"""

import sys
import os
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.ml_pipeline import AirTrafficMLPipeline
from config.config import MODELS_DIR


def run_demo():
    """
    Exécute une démonstration complète du système
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "🛫 DÉMONSTRATION SYSTÈME IA TRAFIC AÉRIEN 🛬")
    print("=" * 80)
    
    # Charger le pipeline
    print("\n📂 Chargement des modèles ML...")
    
    model1_path = f"{MODELS_DIR}/model_1_eta.pkl"
    model2_path = f"{MODELS_DIR}/model_2_occupation.pkl"
    model3_path = f"{MODELS_DIR}/model_3_conflict.pkl"
    
    # Vérifier que les modèles existent
    models_exist = all(os.path.exists(p) for p in [model1_path, model2_path, model3_path])
    
    if not models_exist:
        print("❌ Erreur: Les modèles n'ont pas été trouvés !")
        print("   Exécutez d'abord: python scripts/train_models.py")
        return
    
    pipeline = AirTrafficMLPipeline(model1_path, model2_path, model3_path)
    print("✅ Modèles chargés avec succès !")
    
    # Scénario: Plusieurs vols approchant CDG simultanément
    print("\n" + "=" * 80)
    print("📋 SCÉNARIO: Heure de pointe à Paris CDG - 5 vols en approche")
    print("=" * 80)
    
    flights_batch = [
        {
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
            'trafic_approche': 12,
            'occupation_tarmac': 0.75,
            'historique_occupation_avion': 45,
            'type_vol': 1,
            'passagers_estimes': 165,
            'disponibilite_emplacements': 10,
            'occupation_actuelle': 0.75,
            'meteo_score': 2,
            'trafic_entrant': 14,
            'trafic_sortant': 8,
            'priorite_vol': 3,
            'emplacements_futurs_libres': 7,
        },
        {
            'callsign': 'BAW456',
            'compagnie': 'BA',
            'type_avion': 'A380',
            'vitesse_actuelle': 420,
            'altitude': 4000,
            'distance_piste': 25,
            'temperature': 18,
            'vent_vitesse': 15,
            'visibilite': 10,
            'pluie': 0,
            'retard_historique_compagnie': 15,
            'trafic_approche': 12,
            'occupation_tarmac': 0.75,
            'historique_occupation_avion': 90,
            'type_vol': 1,
            'passagers_estimes': 500,
            'disponibilite_emplacements': 10,
            'occupation_actuelle': 0.75,
            'meteo_score': 2,
            'trafic_entrant': 14,
            'trafic_sortant': 8,
            'priorite_vol': 4,
            'emplacements_futurs_libres': 7,
        },
        {
            'callsign': 'LH789',
            'compagnie': 'LH',
            'type_avion': 'B777',
            'vitesse_actuelle': 460,
            'altitude': 6000,
            'distance_piste': 40,
            'temperature': 18,
            'vent_vitesse': 15,
            'visibilite': 10,
            'pluie': 0,
            'retard_historique_compagnie': 10,
            'trafic_approche': 12,
            'occupation_tarmac': 0.75,
            'historique_occupation_avion': 70,
            'type_vol': 1,
            'passagers_estimes': 350,
            'disponibilite_emplacements': 10,
            'occupation_actuelle': 0.75,
            'meteo_score': 2,
            'trafic_entrant': 14,
            'trafic_sortant': 8,
            'priorite_vol': 3,
            'emplacements_futurs_libres': 7,
        },
        {
            'callsign': 'EZY123',
            'compagnie': 'EZY',
            'type_avion': 'A320',
            'vitesse_actuelle': 440,
            'altitude': 7000,
            'distance_piste': 50,
            'temperature': 18,
            'vent_vitesse': 15,
            'visibilite': 10,
            'pluie': 0,
            'retard_historique_compagnie': 20,
            'trafic_approche': 12,
            'occupation_tarmac': 0.75,
            'historique_occupation_avion': 40,
            'type_vol': 0,
            'passagers_estimes': 180,
            'disponibilite_emplacements': 10,
            'occupation_actuelle': 0.75,
            'meteo_score': 2,
            'trafic_entrant': 14,
            'trafic_sortant': 8,
            'priorite_vol': 2,
            'emplacements_futurs_libres': 7,
        },
        {
            'callsign': 'MED999',
            'compagnie': 'AF',
            'type_avion': 'B737',
            'vitesse_actuelle': 480,
            'altitude': 3000,
            'distance_piste': 20,
            'temperature': 18,
            'vent_vitesse': 15,
            'visibilite': 10,
            'pluie': 0,
            'retard_historique_compagnie': 5,
            'trafic_approche': 12,
            'occupation_tarmac': 0.75,
            'historique_occupation_avion': 35,
            'type_vol': 0,
            'passagers_estimes': 100,
            'disponibilite_emplacements': 10,
            'occupation_actuelle': 0.75,
            'meteo_score': 2,
            'trafic_entrant': 14,
            'trafic_sortant': 8,
            'priorite_vol': 5,  # Urgence médicale
            'emplacements_futurs_libres': 7,
        }
    ]
    
    # Prédictions batch
    results = pipeline.predict_batch(flights_batch)
    
    # Afficher un tableau récapitulatif
    print("\n" + "=" * 80)
    print("📊 TABLEAU RÉCAPITULATIF DES DÉCISIONS")
    print("=" * 80)
    
    header = f"{'Vol':<12} {'Type':<8} {'Retard':<10} {'Occup.':<10} {'Conflit':<10} {'Décision':<40}"
    print(header)
    print("-" * 80)
    
    for result in results:
        vol = result['vol_info']['callsign']
        type_avion = result['vol_info']['type_avion']
        retard = f"{result['modele_1_eta']['retard_predit_minutes']:.1f} min"
        occup = f"{result['modele_2_occupation']['temps_occupation_minutes']:.1f} min"
        conflit = f"{result['modele_3_decision']['risque_conflit']:.1%}"
        
        decision = result['modele_3_decision']['decision_recommandee']
        if len(decision) > 37:
            decision = decision[:37] + "..."
        
        print(f"{vol:<12} {type_avion:<8} {retard:<10} {occup:<10} {conflit:<10} {decision:<40}")
    
    # Détails pour le vol prioritaire
    print("\n" + "=" * 80)
    print("🚨 DÉTAILS VOL PRIORITAIRE - MED999 (Urgence médicale)")
    print("=" * 80)
    
    med_result = [r for r in results if r['vol_info']['callsign'] == 'MED999'][0]
    
    print(f"\n📊 Prédictions:")
    print(f"   Retard estimé: +{med_result['modele_1_eta']['retard_predit_minutes']:.1f} minutes")
    print(f"   Temps d'occupation: {med_result['modele_2_occupation']['temps_occupation_minutes']:.1f} minutes")
    print(f"   Risque de conflit: {med_result['modele_3_decision']['risque_conflit']:.1%}")
    print(f"   Risque de saturation: {med_result['modele_3_decision']['risque_saturation']:.1%}")
    
    print(f"\n💡 Décision: {med_result['modele_3_decision']['decision_recommandee']}")
    print(f"   Confiance: {med_result['modele_3_decision']['confiance_decision']:.1%}")
    print(f"   Priorité: {med_result['modele_3_decision']['priorite']}/10")
    print(f"\n🚨 {med_result['modele_3_decision']['alerte']}")
    
    print(f"\n📋 Actions recommandées:")
    for i, action in enumerate(med_result['actions_recommandees'], 1):
        print(f"   {i}. {action}")
    
    # Statistiques globales
    print("\n" + "=" * 80)
    print("📈 STATISTIQUES GLOBALES")
    print("=" * 80)
    
    avg_delay = sum(r['modele_1_eta']['retard_predit_minutes'] for r in results) / len(results)
    avg_occup = sum(r['modele_2_occupation']['temps_occupation_minutes'] for r in results) / len(results)
    max_conflict_risk = max(r['modele_3_decision']['risque_conflit'] for r in results)
    
    print(f"\nRetard moyen prédit: {avg_delay:.1f} minutes")
    print(f"Temps d'occupation moyen: {avg_occup:.1f} minutes")
    print(f"Risque de conflit maximal: {max_conflict_risk:.1%}")
    
    decisions_count = {}
    for r in results:
        dec = r['modele_3_decision']['decision_recommandee']
        decisions_count[dec] = decisions_count.get(dec, 0) + 1
    
    print(f"\nRépartition des décisions:")
    for decision, count in decisions_count.items():
        print(f"   - {decision}: {count}")
    
    print("\n" + "=" * 80)
    print("✅ DÉMONSTRATION TERMINÉE !")
    print("=" * 80)
    
    print("\n💡 Pour lancer l'API FastAPI:")
    print("   python api/fastapi_app.py")
    print("\n💡 Pour voir la documentation:")
    print("   http://localhost:8000/docs")


if __name__ == "__main__":
    run_demo()
