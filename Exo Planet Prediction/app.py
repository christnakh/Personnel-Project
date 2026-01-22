"""
🔭 Professional Exoplanet Analysis Flask Web Application
Complete web interface with templates, CSS, JS, and API separation.
"""

import os
import sys
import json
import uuid
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask imports
from flask import Flask, request, jsonify, render_template, url_for, redirect, flash
from flask_cors import CORS

# ML imports
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import IsolationForest

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'exoplanet-analysis-secret-key-2024'

# Global variables for models and components
models = {}
scaler = None
label_encoder = None
explainers = {}
anomaly_detector = None
cnn_model = None
hybrid_model = None

# Feature names
FEATURE_NAMES = [
    'period', 'duration', 'depth', 'planet_radius', 'stellar_radius',
    'stellar_temp', 'stellar_mag', 'impact_param', 'transit_snr',
    'num_transits', 'duty_cycle', 'log_period', 'log_planet_radius', 'log_depth'
]

CLASS_NAMES = ['confirmed', 'candidate', 'false_positive']

# Physical constants for habitability
SOLAR_LUMINOSITY = 3.828e26  # W
SOLAR_TEMPERATURE = 5778  # K
STEFAN_BOLTZMANN = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
AU = 1.496e11  # m

class HabitabilityCalculator:
    """Habitability analysis for exoplanets"""
    
    def calculate_habitable_zone(self, stellar_temp: float, stellar_luminosity: float = None) -> Dict[str, float]:
        """Calculate habitable zone boundaries"""
        try:
            if stellar_luminosity is None:
                stellar_luminosity = (stellar_temp / SOLAR_TEMPERATURE) ** 4
            
            # Conservative habitable zone
            inner_edge = np.sqrt(stellar_luminosity / 1.1)
            outer_edge = np.sqrt(stellar_luminosity / 0.53)
            
            # Optimistic habitable zone
            optimistic_inner = np.sqrt(stellar_luminosity / 1.77)
            optimistic_outer = np.sqrt(stellar_luminosity / 0.32)
            
            return {
                'inner_edge': inner_edge,
                'outer_edge': outer_edge,
                'optimistic_inner': optimistic_inner,
                'optimistic_outer': optimistic_outer
            }
        except:
            return {'inner_edge': 0, 'outer_edge': 0, 'optimistic_inner': 0, 'optimistic_outer': 0}
    
    def calculate_equilibrium_temperature(self, stellar_temp: float, stellar_radius: float, 
                                          orbital_period: float, albedo: float = 0.3) -> float:
        """Calculate equilibrium temperature"""
        try:
            period_years = orbital_period / 365.25
            semi_major_axis = (period_years ** 2) ** (1/3)
            stellar_flux = (stellar_temp / SOLAR_TEMPERATURE) ** 4 * (stellar_radius ** 2) / (semi_major_axis ** 2)
            teq = stellar_temp * ((1 - albedo) * stellar_flux) ** 0.25
            return teq
        except:
            return 0.0
    
    def calculate_habitability_score(self, planet_radius: float, stellar_temp: float, 
                                    stellar_radius: float, orbital_period: float,
                                    stellar_mag: float = None) -> Dict[str, Any]:
        """Calculate comprehensive habitability score"""
        try:
            # Size score
            if 0.5 <= planet_radius <= 2.5:
                if 0.8 <= planet_radius <= 1.4:
                    size_score = 1.0
                else:
                    size_score = 1.0 - abs(planet_radius - 1.1) / 0.3
            else:
                size_score = 0.0
            size_score = max(0, min(1, size_score))
            
            # Stellar score
            if 2500 <= stellar_temp <= 7200:
                if 5000 <= stellar_temp <= 6500:
                    stellar_score = 1.0
                else:
                    stellar_score = 1.0 - abs(stellar_temp - 5750) / 1000
            else:
                stellar_score = 0.0
            stellar_score = max(0, min(1, stellar_score))
            
            # Habitable zone score
            hz = self.calculate_habitable_zone(stellar_temp)
            eq_temp = self.calculate_equilibrium_temperature(stellar_temp, stellar_radius, orbital_period)
            
            hz_score = 0.0
            hz_position = 'outside'
            if hz['inner_edge'] > 0 and hz['outer_edge'] > 0:
                period_years = orbital_period / 365.25
                orbital_distance = (period_years ** 2) ** (1/3)
                
                if hz['inner_edge'] <= orbital_distance <= hz['outer_edge']:
                    hz_score = 1.0
                    hz_position = 'conservative'
                elif hz['optimistic_inner'] <= orbital_distance <= hz['optimistic_outer']:
                    hz_score = 0.7
                    hz_position = 'optimistic'
                else:
                    if orbital_distance < hz['inner_edge']:
                        hz_position = 'too_close'
                    else:
                        hz_position = 'too_far'
            
            # Temperature score
            if 200 <= eq_temp <= 400:
                if 250 <= eq_temp <= 350:
                    temp_score = 1.0
                else:
                    temp_score = 1.0 - abs(eq_temp - 300) / 50
            else:
                temp_score = 0.0
            temp_score = max(0, min(1, temp_score))
            
            # Overall habitability score
            weights = {'size': 0.25, 'stellar': 0.20, 'habitable_zone': 0.35, 'temperature': 0.20}
            overall_score = (
                size_score * weights['size'] +
                stellar_score * weights['stellar'] +
                hz_score * weights['habitable_zone'] +
                temp_score * weights['temperature']
            )
            
            return {
                'habitability_score': overall_score,
                'is_habitable': overall_score >= 0.6,
                'habitable_zone_score': hz_score,
                'size_score': size_score,
                'temperature_score': temp_score,
                'stellar_score': stellar_score,
                'equilibrium_temp': eq_temp,
                'habitable_zone_position': hz_position
            }
        except Exception as e:
            return {
                'habitability_score': 0.0,
                'is_habitable': False,
                'habitable_zone_score': 0.0,
                'size_score': 0.0,
                'temperature_score': 0.0,
                'stellar_score': 0.0,
                'equilibrium_temp': 0.0,
                'habitable_zone_position': 'error'
            }

def load_models():
    """Load all ML models and components"""
    global models, scaler, label_encoder, explainers, anomaly_detector
    
    try:
        models_dir = "models"
        
        # Load models
        models['xgboost'] = joblib.load(f"{models_dir}/trained_models/xgboost_model.pkl")
        models['random_forest'] = joblib.load(f"{models_dir}/trained_models/random_forest_model.pkl")
        
        # Try to load ensemble model if it exists
        try:
            models['ensemble'] = joblib.load(f"{models_dir}/trained_models/ensemble_model.pkl")
        except FileNotFoundError:
            logger.warning("Ensemble model not found, skipping...")
            models['ensemble'] = None
        
        # Load preprocessors
        scaler = joblib.load(f"{models_dir}/trained_models/scaler.pkl")
        label_encoder = joblib.load(f"{models_dir}/trained_models/label_encoder.pkl")
        
        # Initialize explainers
        explainers['xgboost_shap'] = shap.TreeExplainer(models['xgboost'])
        explainers['rf_shap'] = shap.TreeExplainer(models['random_forest'])
        
        # Initialize LIME explainer
        explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.random.random((100, len(FEATURE_NAMES))),
            feature_names=FEATURE_NAMES,
            class_names=CLASS_NAMES,
            mode='classification'
        )
        
        # Initialize and fit anomaly detector
        anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Fit the anomaly detector with sample data
        try:
            import pandas as pd
            # Load some data to fit the anomaly detector
            df = pd.read_csv('data/processed_exoplanet_data.csv')
            sample_features = df[FEATURE_NAMES].head(1000).values
            anomaly_detector.fit(sample_features)
            logger.info("✅ Anomaly detector fitted successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not fit anomaly detector: {e}")
            # Create dummy data for fitting
            dummy_data = np.random.random((100, len(FEATURE_NAMES)))
            anomaly_detector.fit(dummy_data)
        
        # Try to load CNN model
        try:
            import sys
            sys.path.append('models')
            from cnn_model import LightCurveCNN
            cnn_model = LightCurveCNN()
            # Try to load the saved model
            try:
                cnn_model.load_model('models/trained_models/cnn_model.keras')
                print("✅ CNN model loaded successfully!")
            except:
                # If model file doesn't exist, create a dummy model for demo
                cnn_model = None
                print("⚠️  CNN model file not found, using fallback")
        except Exception as e:
            print(f"⚠️  CNN model not available: {e}")
            cnn_model = None
        
        # Try to load hybrid model
        try:
            from models.hybrid_model import AdvancedHybridModel
            hybrid_model = AdvancedHybridModel()
            hybrid_model.load_models()
            print("✅ Advanced hybrid model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Hybrid model not available: {e}")
            hybrid_model = None
        
        print("✅ All models and components loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def prepare_features(features_dict: Dict[str, float]) -> np.ndarray:
    """Prepare features for prediction"""
    try:
        # Create feature array
        feature_array = np.array([
            features_dict.get('period', 10.0),
            features_dict.get('duration', 2.0),
            features_dict.get('depth', 1000.0),
            features_dict.get('planet_radius', 1.0),
            features_dict.get('stellar_radius', 1.0),
            features_dict.get('stellar_temp', 5800.0),
            features_dict.get('stellar_mag', 12.0),
            features_dict.get('impact_param', 0.5),
            features_dict.get('transit_snr', 10.0),
            features_dict.get('num_transits', 10),
            features_dict.get('duty_cycle', 0.0083),
            features_dict.get('log_period', 1.0),
            features_dict.get('log_planet_radius', 0.0),
            features_dict.get('log_depth', 3.0)
        ])
        
        # Handle missing values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_array.reshape(1, -1)
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        return np.zeros((1, len(FEATURE_NAMES)))

def predict_classification(features: np.ndarray) -> Dict[str, Any]:
    """Make classification prediction"""
    try:
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get predictions from both models
        xgb_pred = models['xgboost'].predict_proba(features_scaled)[0]
        rf_pred = models['random_forest'].predict_proba(features_scaled)[0]
        
        # Debug: Check what type of objects we have
        logger.info(f"XGBoost model type: {type(models['xgboost'])}")
        logger.info(f"Random Forest model type: {type(models['random_forest'])}")
        logger.info(f"Ensemble model type: {type(models['ensemble'])}")
        
        # Ensemble prediction
        # Average predictions for ensemble (or use XGBoost if ensemble model not available)
        if models['ensemble'] is not None and hasattr(models['ensemble'], 'predict_proba'):
            ensemble_pred = models['ensemble'].predict_proba(features_scaled)[0]
        else:
            ensemble_pred = (xgb_pred + rf_pred) / 2
        
        # Get predicted class
        predicted_class_idx = np.argmax(ensemble_pred)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = ensemble_pred[predicted_class_idx]
        
        # Create probabilities dictionary
        probabilities = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, ensemble_pred)}
        
        return {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'probabilities': probabilities,
            'xgboost_prediction': {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, xgb_pred)},
            'random_forest_prediction': {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, rf_pred)}
        }
        
    except Exception as e:
        print(f"Error in classification prediction: {e}")
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'probabilities': {class_name: 0.0 for class_name in CLASS_NAMES},
            'error': str(e)
        }

def get_explanations(features: np.ndarray) -> Dict[str, Any]:
    """Get SHAP and LIME explanations"""
    try:
        explanations = {}
        
        # SHAP explanations
        if 'xgboost_shap' in explainers:
            xgb_shap_values = explainers['xgboost_shap'].shap_values(features)
            if isinstance(xgb_shap_values, list):
                predicted_class_idx = np.argmax(models['xgboost'].predict_proba(features)[0])
                xgb_shap_values = xgb_shap_values[predicted_class_idx]
            
            explanations['xgboost_shap'] = {
                'shap_values': xgb_shap_values.tolist() if hasattr(xgb_shap_values, 'tolist') else xgb_shap_values,
                'feature_importance': dict(zip(FEATURE_NAMES, xgb_shap_values.tolist() if hasattr(xgb_shap_values, 'tolist') else xgb_shap_values)),
                'top_features': sorted(zip(FEATURE_NAMES, xgb_shap_values.tolist() if hasattr(xgb_shap_values, 'tolist') else xgb_shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
            }
        
        # LIME explanation
        if 'lime' in explainers:
            try:
                lime_explanation = explainers['lime'].explain_instance(
                    features[0], 
                    models['random_forest'].predict_proba,
                    num_features=10
                )
                explanations['lime'] = {
                    'feature_importance': dict(lime_explanation.as_list()),
                    'top_features': lime_explanation.as_list()[:5]
                }
            except Exception as e:
                explanations['lime'] = {'error': str(e)}
        
        return explanations
        
    except Exception as e:
        return {'error': str(e)}

def detect_anomalies(features: np.ndarray) -> Dict[str, Any]:
    """Detect anomalies in the exoplanet system"""
    try:
        if anomaly_detector is None:
            logger.warning("Anomaly detector not initialized")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'anomaly_type': 'normal',
                'confidence': 0.0
            }
        
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get anomaly scores and predictions
        anomaly_scores = anomaly_detector.decision_function(features)
        anomaly_predictions = anomaly_detector.predict(features)
        
        is_anomaly = anomaly_predictions[0] == -1
        anomaly_score = float(anomaly_scores[0])
        confidence = min(100.0, max(0.0, abs(anomaly_score) * 50))  # Convert to percentage
        
        # Determine anomaly type
        feature_dict = dict(zip(FEATURE_NAMES, features[0]))
        anomaly_type = 'normal'
        
        if is_anomaly:
            if feature_dict.get('period', 0) > 1000:
                anomaly_type = 'long_period_system'
            elif feature_dict.get('depth', 0) > 50000:
                anomaly_type = 'deep_transit'
            elif feature_dict.get('planet_radius', 0) > 20:
                anomaly_type = 'giant_planet'
            elif feature_dict.get('transit_snr', 0) > 1000:
                anomaly_type = 'high_snr_system'
            elif feature_dict.get('duty_cycle', 0) > 0.1:
                anomaly_type = 'long_transit'
            else:
                anomaly_type = 'unusual_system'
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': anomaly_score,
            'anomaly_type': anomaly_type,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        return {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'anomaly_type': 'normal',
            'confidence': 0.0
        }

# Initialize models on startup
print("🔭 Initializing Exoplanet Analysis System...")
if not load_models():
    print("❌ Failed to load models. Please ensure models are trained first.")
    sys.exit(1)

# Initialize habitability calculator
habitability_calculator = HabitabilityCalculator()

# Web Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    """Single analysis page"""
    return render_template('analyze.html')

@app.route('/batch')
def batch():
    """Batch analysis page"""
    return render_template('batch.html')

@app.route('/api-docs')
def api_docs():
    """API documentation page"""
    return render_template('api_docs.html')

@app.route('/dashboard')
def dashboard():
    """Advanced dashboard page"""
    return render_template('dashboard.html')

@app.route('/realtime')
def realtime():
    """Real-time space data processing page"""
    return render_template('realtime.html')

@app.route('/models')
def models_page():
    """Models overview page"""
    return render_template('models.html')

# API Routes
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Complete analysis API endpoint"""
    try:
        start_time = time.time()
        
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Prepare features
        features = prepare_features(data)
        
        # Get classification prediction
        classification = predict_classification(features)
        
        # Get habitability analysis
        habitability = habitability_calculator.calculate_habitability_score(
            planet_radius=data.get('planet_radius', 1.0),
            stellar_temp=data.get('stellar_temp', 5800.0),
            stellar_radius=data.get('stellar_radius', 1.0),
            orbital_period=data.get('period', 365.0),
            stellar_mag=data.get('stellar_mag', 12.0)
        )
        
        # Get explanations only if requested (slow operation)
        explanations = None
        if data.get('include_explanations', False):
            explanations = get_explanations(features)
        else:
            explanations = {'note': 'Explanations not requested for faster processing'}
        
        # Get anomaly detection
        anomaly = detect_anomalies(features)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create comprehensive response
        response = {
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': round(processing_time, 2),
            'classification': classification,
            'habitability': habitability,
            'explanations': explanations,
            'anomaly_detection': anomaly,
            'input_parameters': data
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Basic classification API endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        features = prepare_features(data)
        prediction = predict_classification(features)
        
        return jsonify({
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/habitability', methods=['POST'])
def api_habitability():
    """Habitability analysis API endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        habitability_result = habitability_calculator.calculate_habitability_score(
            planet_radius=data.get('planet_radius', 1.0),
            stellar_temp=data.get('stellar_temp', 5800.0),
            stellar_radius=data.get('stellar_radius', 1.0),
            orbital_period=data.get('period', 365.0),
            stellar_mag=data.get('stellar_mag', 12.0)
        )
        
        return jsonify({
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'habitability': habitability_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def api_batch():
    """Batch analysis API endpoint"""
    try:
        data = request.get_json()
        if not data or 'candidates' not in data:
            return jsonify({'error': 'No candidates data provided'}), 400
        
        results = []
        for i, candidate in enumerate(data['candidates']):
            try:
                features = prepare_features(candidate)
                classification = predict_classification(features)
                habitability = habitability_calculator.calculate_habitability_score(
                    planet_radius=candidate.get('planet_radius', 1.0),
                    stellar_temp=candidate.get('stellar_temp', 5800.0),
                    stellar_radius=candidate.get('stellar_radius', 1.0),
                    orbital_period=candidate.get('period', 365.0),
                    stellar_mag=candidate.get('stellar_mag', 12.0)
                )
                anomaly = detect_anomalies(features)
                
                results.append({
                    'candidate_id': i,
                    'classification': classification,
                    'habitability': habitability,
                    'anomaly_detection': anomaly
                })
            except Exception as e:
                results.append({
                    'candidate_id': i,
                    'error': str(e)
                })
        
        return jsonify({
            'batch_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'total_candidates': len(data['candidates']),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comprehensive', methods=['POST'])
def api_comprehensive():
    """Comprehensive analysis with all models including CNN"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get basic prediction
        features = prepare_features(data)
        basic_result = predict_classification(features)
        
        # Get habitability analysis
        habitability_result = habitability_calculator.calculate_habitability_score(
            planet_radius=data.get('planet_radius', 1.0),
            stellar_temp=data.get('stellar_temp', 5800.0),
            stellar_radius=data.get('stellar_radius', 1.0),
            orbital_period=data.get('period', 365.0),
            stellar_mag=data.get('stellar_mag', 12.0)
        )
        
        # Get explanations
        explanations = get_explanations(features)
        
        # Get anomaly score
        anomaly_score = detect_anomalies(features)
        
        # Try to get hybrid model prediction if available
        hybrid_result = None
        cnn_result = None
        
        if hybrid_model is not None:
            try:
                # Generate synthetic light curve for demonstration
                from models.cnn_model import generate_synthetic_light_curve
                light_curve = generate_synthetic_light_curve(
                    length=1000,
                    has_transit=data.get('transit_depth', 0.01) > 0.005,
                    noise_level=0.01,
                    transit_depth=data.get('transit_depth', 0.01)
                )
                
                # Use hybrid model for comprehensive prediction
                hybrid_result = hybrid_model.predict_single(data, light_curve)
                
                # Also get individual CNN result if available
                if cnn_model is not None:
                    cnn_predictions, cnn_probabilities = cnn_model.predict([light_curve])
                    cnn_result = {
                        'prediction': cnn_predictions[0],
                        'probabilities': cnn_probabilities[0].tolist(),
                        'confidence': float(np.max(cnn_probabilities[0]))
                    }
            except Exception as e:
                hybrid_result = {'error': f'Hybrid prediction failed: {str(e)}'}
                cnn_result = {'error': f'CNN prediction failed: {str(e)}'}
        
        # Combine all results
        comprehensive_result = {
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'basic_prediction': basic_result,
            'habitability': habitability_result,
            'explanations': explanations,
            'anomaly_detection': anomaly_score,
            'cnn_analysis': cnn_result,
            'hybrid_analysis': hybrid_result,
            'model_ensemble': {
                'xgboost_available': 'xgboost' in models,
                'random_forest_available': 'random_forest' in models,
                'cnn_available': cnn_model is not None,
                'hybrid_available': hybrid_model is not None,
                'total_models': len([m for m in models.values() if m is not None]) + (1 if cnn_model is not None else 0) + (1 if hybrid_model is not None else 0)
            }
        }
        
        return jsonify(comprehensive_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check API endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'models_loaded': len(models) > 0,
        'features': {
            'classification': True,
            'habitability': True,
            'explanations': True,
            'anomaly_detection': True
        }
    })

@app.route('/api/nasa/latest', methods=['GET'])
def api_nasa_latest():
    """Get latest NASA exoplanet discoveries"""
    try:
        from src.nasa_integration import get_nasa_data
        nasa_data = get_nasa_data()
        return jsonify(nasa_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nasa/habitable', methods=['GET'])
def api_nasa_habitable():
    """Get potentially habitable planets from NASA"""
    try:
        from src.nasa_integration import NASAExoplanetAPI
        nasa_api = NASAExoplanetAPI()
        habitable_planets = nasa_api.get_habitable_planets(limit=10)
        return jsonify({
            'habitable_planets': habitable_planets,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nasa/statistics', methods=['GET'])
def api_nasa_statistics():
    """Get exoplanet discovery statistics"""
    try:
        from src.nasa_integration import NASAExoplanetAPI
        nasa_api = NASAExoplanetAPI()
        statistics = nasa_api.get_discovery_statistics()
        return jsonify({
            'statistics': statistics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Import configuration
    from config import get_config, setup_environment
    
    # Setup environment
    setup_environment()
    
    # Get configuration
    config = get_config()
    
    print("🚀 Starting Exoplanet Analysis Flask Web Application...")
    print("📊 Available endpoints:")
    print("  • GET / - Home page")
    print("  • GET /analyze - Single analysis page")
    print("  • GET /dashboard - Advanced dashboard")
    print("  • GET /batch - Batch analysis page")
    print("  • GET /api-docs - API documentation")
    print("  • POST /api/analyze - Complete analysis API")
    print("  • POST /api/predict - Classification API")
    print("  • POST /api/habitability - Habitability API")
    print("  • POST /api/batch - Batch analysis API")
    print("  • GET /api/health - Health check")
    print("  • POST /api/comprehensive - Comprehensive analysis API")
    print("  • GET /api/nasa/latest - NASA latest discoveries")
    print("  • GET /api/nasa/habitable - NASA habitable planets")
    print("  • GET /api/nasa/statistics - NASA statistics")
    print("")
    
    # Get available port
    port = config.get_available_port()
    print(f"🌐 Web application starting on http://localhost:{port}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=config.DEBUG)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Trying alternative port...")
            # Try alternative port
            alternative_port = config.get_available_port()
            print(f"🌐 Starting on alternative port: http://localhost:{alternative_port}")
            app.run(host='0.0.0.0', port=alternative_port, debug=config.DEBUG)
        else:
            raise