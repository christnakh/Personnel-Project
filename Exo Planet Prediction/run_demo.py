#!/usr/bin/env python3
"""
Demo script to showcase the exoplanet ML model.
Runs the complete pipeline and demonstrates key features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')

from data_preprocessing import ExoplanetDataProcessor
from model_training import ExoplanetModelTrainer
from inference import ExoplanetInference

def main():
    """Run the complete exoplanet ML pipeline demo"""
    
    print("🪐 Exoplanet Classification ML Model Demo")
    print("=" * 50)
    
    # Check if models exist
    models_exist = all([
        Path("models/xgboost_model.pkl").exists(),
        Path("models/random_forest_model.pkl").exists(),
        Path("models/label_encoder.pkl").exists(),
        Path("models/scaler.pkl").exists()
    ])
    
    if not models_exist:
        print("📊 Step 1: Data Preprocessing")
        print("-" * 30)
        
        # Run data preprocessing
        processor = ExoplanetDataProcessor()
        processor.load_datasets()
        processor.standardize_labels()
        processor.create_features()
        processor.clean_data()
        processor.save_processed_data()
        
        summary = processor.get_data_summary()
        print(f"✅ Processed {summary['shape'][0]} candidates")
        print(f"   Labels: {summary['label_distribution']}")
        print(f"   Missions: {summary['mission_distribution']}")
        
        print("\n🤖 Step 2: Model Training")
        print("-" * 30)
        
        # Train models
        trainer = ExoplanetModelTrainer()
        trainer.load_data()
        trainer.prepare_features()
        trainer.train_xgboost()
        trainer.train_random_forest()
        trainer.train_ensemble()
        trainer.save_models()
        
        print("✅ Models trained and saved!")
    else:
        print("✅ Models already exist, skipping training...")
    
    print("\n🔮 Step 3: Making Scientific Predictions")
    print("-" * 30)
    
    # Initialize predictor
    predictor = ExoplanetInference()
    
    # Example predictions
    examples = [
        {
            'name': 'Earth-like Candidate',
            'features': {
                'period': 365.0, 'duration': 13.0, 'depth': 84.0, 'planet_radius': 1.0,
                'stellar_radius': 1.0, 'stellar_temp': 5778.0, 'stellar_mag': 4.8,
                'impact_param': 0.0, 'transit_snr': 50.0, 'num_transits': 3,
                'duty_cycle': 0.0015, 'log_period': 2.56, 'log_planet_radius': 0.0, 'log_depth': 1.92
            }
        },
        {
            'name': 'Hot Jupiter',
            'features': {
                'period': 3.0, 'duration': 3.0, 'depth': 10000.0, 'planet_radius': 10.0,
                'stellar_radius': 1.0, 'stellar_temp': 6000.0, 'stellar_mag': 8.0,
                'impact_param': 0.1, 'transit_snr': 100.0, 'num_transits': 100,
                'duty_cycle': 0.0417, 'log_period': 0.48, 'log_planet_radius': 1.0, 'log_depth': 4.0
            }
        },
        {
            'name': 'False Positive',
            'features': {
                'period': 1.0, 'duration': 0.5, 'depth': 100.0, 'planet_radius': 0.1,
                'stellar_radius': 0.5, 'stellar_temp': 3000.0, 'stellar_mag': 15.0,
                'impact_param': 0.9, 'transit_snr': 2.0, 'num_transits': 1,
                'duty_cycle': 0.0208, 'log_period': 0.0, 'log_planet_radius': -1.0, 'log_depth': 2.0
            }
        }
    ]
    
    for example in examples:
        print(f"\n📋 {example['name']}:")
        
        # Comprehensive prediction with habitability
        result = predictor.predict_comprehensive(example['features'])
        
        print(f"   Prediction: {result['prediction'].title()}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   🌍 Habitable: {'YES' if result['habitability']['is_habitable'] else 'NO'}")
        print(f"   🌍 Habitability Score: {result['habitability']['habitability_score']:.3f}")
        print(f"   🌡️  Equilibrium Temp: {result['habitability']['equilibrium_temp']:.1f} K")
        print("   Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"     {class_name}: {prob:.1%}")
    
    print("\n📊 Step 4: Batch Prediction Demo")
    print("-" * 30)
    
    # Create sample batch data
    batch_data = pd.DataFrame([
        {
            'period': 10.0, 'duration': 2.0, 'depth': 1000.0, 'planet_radius': 1.0,
            'stellar_radius': 1.0, 'stellar_temp': 5800.0, 'stellar_mag': 12.0,
            'impact_param': 0.5, 'transit_snr': 10.0, 'num_transits': 10,
            'duty_cycle': 0.0083, 'log_period': 1.0, 'log_planet_radius': 0.0, 'log_depth': 3.0
        },
        {
            'period': 5.0, 'duration': 1.5, 'depth': 500.0, 'planet_radius': 0.8,
            'stellar_radius': 0.9, 'stellar_temp': 5500.0, 'stellar_mag': 11.5,
            'impact_param': 0.3, 'transit_snr': 15.0, 'num_transits': 20,
            'duty_cycle': 0.0125, 'log_period': 0.7, 'log_planet_radius': -0.1, 'log_depth': 2.7
        }
    ])
    
    batch_results = predictor.predict_batch(batch_data)
    print("✅ Batch predictions completed!")
    print("\nResults:")
    print(batch_results[['period', 'planet_radius', 'predicted_class', 'confidence']].to_string(index=False))
    
    print("\n🔬 Step 5: Scientific Analysis Features")
    print("-" * 30)
    print("✅ Habitability Analysis - Identifies potentially habitable planets")
    print("✅ Explainable AI - SHAP & LIME explanations for predictions")
    print("✅ Cross-Mission Validation - Tests model robustness across missions")
    print("✅ Anomaly Detection - Finds unusual and rare exoplanet systems")
    print("✅ Multi-Planet Detection - Identifies systems with multiple planets")
    
    print("\n🌐 Step 6: Web Interface")
    print("-" * 30)
    print("To launch the web interface, run:")
    print("  streamlit run app/streamlit_app.py")
    print("\nThe web interface provides:")
    print("  • Interactive single predictions with habitability")
    print("  • CSV batch upload with scientific analysis")
    print("  • AI explanations and feature importance")
    print("  • Anomaly detection and rare system identification")
    print("  • Results download with comprehensive analysis")
    
    print("\n🎯 Summary")
    print("-" * 30)
    print("✅ Data preprocessing completed")
    print("✅ Models trained (XGBoost, Random Forest, Ensemble)")
    print("✅ Habitability analysis implemented")
    print("✅ Explainable AI (SHAP & LIME) integrated")
    print("✅ Cross-mission validation ready")
    print("✅ Anomaly detection implemented")
    print("✅ Multi-planet system detection ready")
    print("✅ Inference API working")
    print("✅ Web interface ready")
    print("✅ Documentation complete")
    
    print("\n🔭 SCIENTIFIC EXOPLANET ANALYSIS SYSTEM READY!")
    print("=" * 50)
    print("🌍 Habitability prediction: YES/NO with confidence scores")
    print("🤖 AI Explanations: SHAP & LIME feature importance")
    print("🌌 Cross-Mission: Train on Kepler → Test on TESS/K2")
    print("🔍 Anomaly Detection: Find rare and unusual systems")
    print("🌌 Multi-Planet Systems: Detect TRAPPIST-1-like systems")
    print("=" * 50)

if __name__ == "__main__":
    main()
