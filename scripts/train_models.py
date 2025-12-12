"""
Script d'entraînement complet pour les 3 modèles ML
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.ml_pipeline import AirTrafficMLPipeline
from models.model_1_eta_prediction import create_sample_data as create_data_m1
from models.model_2_occupation import create_sample_data as create_data_m2
from models.model_3_conflict_detection import create_sample_data as create_data_m3
from config.config import MODELS_DIR, SYNTHETIC_DATA_SIZE


def train_all_models(n_samples: int = None):
    """
    Entraîne tous les modèles et sauvegarde
    
    Args:
        n_samples: Nombre d'échantillons pour l'entraînement (None = config par défaut)
    """
    if n_samples is None:
        n_samples = SYNTHETIC_DATA_SIZE
    
    print("=" * 80)
    print(" " * 25 + "🎓 ENTRAÎNEMENT DES MODÈLES ML")
    print("=" * 80)
    print(f"\nNombre d'échantillons: {n_samples}")
    print(f"Répertoire de sauvegarde: {MODELS_DIR}\n")
    
    # Créer le répertoire models s'il n'existe pas
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Créer les données synthétiques
    print("📊 Génération des données d'entraînement...")
    print("   - Données pour Modèle 1 (ETA/ETD)...")
    df_model1 = create_data_m1(n_samples)
    
    print("   - Données pour Modèle 2 (Occupation)...")
    df_model2 = create_data_m2(n_samples)
    
    print("   - Données pour Modèle 3 (Conflit)...")
    df_model3 = create_data_m3(n_samples)
    
    print("✅ Données générées !\n")
    
    # Créer le pipeline et entraîner
    pipeline = AirTrafficMLPipeline()
    
    metrics = pipeline.train_all_models(
        df_model1,
        df_model2,
        df_model3,
        save_dir=MODELS_DIR
    )
    
    # Afficher un résumé
    print("\n" + "=" * 80)
    print(" " * 30 + "📊 RÉSUMÉ DES PERFORMANCES")
    print("=" * 80)
    
    print("\n🔮 Modèle 1 - Prédiction ETA/ETD:")
    print(f"   MAE: {metrics['model1_metrics']['mae_eta']:.2f} minutes")
    print(f"   R²: {metrics['model1_metrics']['r2_eta']:.3f}")
    print(f"   Accuracy retard >15min: {metrics['model1_metrics']['acc_delay_15']:.2%}")
    
    print("\n🔮 Modèle 2 - Durée d'occupation:")
    print(f"   MAE: {metrics['model2_metrics']['mae']:.2f} minutes")
    print(f"   RMSE: {metrics['model2_metrics']['rmse']:.2f} minutes")
    print(f"   R²: {metrics['model2_metrics']['r2']:.3f}")
    
    print("\n🔮 Modèle 3 - Détection de conflits:")
    print(f"   Accuracy conflit: {metrics['model3_metrics']['acc_conflict']:.2%}")
    print(f"   Accuracy saturation: {metrics['model3_metrics']['acc_saturation']:.2%}")
    print(f"   Accuracy décision: {metrics['model3_metrics']['acc_decision']:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("=" * 80)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraîner les modèles ML")
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Nombre d\'échantillons pour l\'entraînement (défaut: 2000)'
    )
    
    args = parser.parse_args()
    
    train_all_models(n_samples=args.samples)
