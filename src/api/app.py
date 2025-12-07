"""
Flask API for IVF Patient Response Prediction
==============================================
API REST pour la prédiction des réponses ovariennes en FIV
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from pathlib import Path
import sys
import os
import traceback

# ============================================================================
# CONFIGURATION DES CHEMINS
# ============================================================================

# Obtenir le répertoire actuel (src/api/)
current_dir = Path(__file__).parent.absolute()
# Remonter à la racine du projet (Tanit/)
project_root = current_dir.parent.parent
# Ajouter src/ au PYTHONPATH
src_dir = project_root / 'src'

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

print(f"📂 Project root: {project_root}")
print(f"📂 Source directory: {src_dir}")
print(f"📂 Current directory: {current_dir}")

# ============================================================================
# IMPORT DU PREDICTOR
# ============================================================================

try:
    # Import depuis src.model.predict
    from model.predict import PatientResponsePredictor
    print("✅ Import success: model.predict")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print(f"💡 Trying alternative import...")
    try:
        # Import direct
        sys.path.insert(0, str(project_root / 'src' / 'model'))
        from predict import PatientResponsePredictor
        print("✅ Import success: direct import")
    except ImportError as e2:
        print(f"❌ All imports failed!")
        print(f"Error 1: {e}")
        print(f"Error 2: {e2}")
        print(f"\n💡 Solution: Vérifiez que predict.py existe dans:")
        print(f"   {project_root / 'src' / 'model' / 'predict.py'}")
        sys.exit(1)

# ============================================================================
# CONFIGURATION DU LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALISATION FLASK
# ============================================================================

app = Flask(__name__)
CORS(app)  # Activer CORS pour React/Streamlit

# Variable globale pour le prédicteur
predictor = None

# ============================================================================
# FONCTION D'INITIALISATION
# ============================================================================

def initialize_predictor():
    """Initialise le prédicteur au démarrage de l'application"""
    global predictor
    try:
        # Chemin vers les modèles sauvegardés
        model_dir = project_root / 'src' / 'model' / 'saved_models'
        
        logger.info(f"📁 Loading model from: {model_dir}")
        
        # Vérifier que le dossier existe
        if not model_dir.exists():
            logger.error(f"❌ Model directory not found: {model_dir}")
            logger.error("💡 Run training first: python src/model/train.py")
            return False
        
        # Vérifier que best_model.pkl existe
        best_model_path = model_dir / 'best_model.pkl'
        if not best_model_path.exists():
            logger.error(f"❌ best_model.pkl not found in {model_dir}")
            logger.error("💡 Run evaluation first: python src/model/evaluate.py")
            return False
        
        # Initialiser le prédicteur
        predictor = PatientResponsePredictor(model_dir=str(model_dir))
        predictor.load_model()
        
        logger.info("✅ Predictor initialized successfully")
        logger.info(f"📊 Model type: {type(predictor.model).__name__}")
        logger.info(f"📊 Features: {len(predictor.feature_info.get('feature_columns', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# ============================================================================
# ROUTES API
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Endpoint de santé pour vérifier que l'API fonctionne
    
    Returns:
        JSON avec status de l'API
    """
    return jsonify({
        'status': 'healthy',
        'message': 'IVF Prediction API is running',
        'model_loaded': predictor is not None,
        'version': '1.0.0'
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict_single():
    """
    Prédiction pour un seul patient
    
    Body JSON attendu:
    {
        "Age": 32,
        "Cycle_number": 1,
        "Protocol": "flex anta",
        "AMH": 2.5,
        "N_Follicles": 15,
        "E2_day5": 300,
        "AFC": 15
    }
    
    Returns:
        JSON avec la prédiction et les probabilités
    """
    try:
        # Vérifier que le modèle est chargé
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the server.'
            }), 503
        
        # Récupérer les données du patient
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        logger.info(f"📥 Received prediction request: {patient_data}")
        
        # Valider les champs requis
        required_fields = ['Age', 'AMH', 'N_Follicles', 'E2_day5', 'AFC']
        missing_fields = [field for field in required_fields if field not in patient_data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'missing_fields': missing_fields
            }), 400
        
        # Valeurs par défaut pour les champs optionnels
        if 'Cycle_number' not in patient_data:
            patient_data['Cycle_number'] = 1
        if 'Protocol' not in patient_data:
            patient_data['Protocol'] = 'flexible antagonist'
        
        # Faire la prédiction
        result = predictor.predict_single_patient(patient_data, verbose=False)
        
        # Formater la réponse
        response = {
            'success': True,
            'predicted_class': result['predicted_response'],
            'confidence': float(result['confidence']),
            'probabilities': {k: float(v) for k, v in result['probabilities'].items()},
            'interpretation': result['clinical_interpretation'],
            'recommendations': result['recommendations']
        }
        
        logger.info(f"✅ Prediction successful: {result['predicted_response']} ({result['confidence']*100:.1f}%)")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'details': traceback.format_exc() if app.debug else None
        }), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Prédiction pour plusieurs patients
    
    Body JSON attendu:
    {
        "patients": [
            {"Age": 32, "AMH": 2.5, ...},
            {"Age": 28, "AMH": 3.5, ...}
        ]
    }
    
    Returns:
        JSON avec toutes les prédictions
    """
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the server.'
            }), 503
        
        data = request.get_json()
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({
                'success': False,
                'error': 'No patients data provided'
            }), 400
        
        logger.info(f"📥 Received batch prediction request for {len(patients)} patients")
        
        # Prédictions pour chaque patient
        predictions = []
        errors = []
        
        for idx, patient_data in enumerate(patients):
            try:
                # Valeurs par défaut
                if 'Cycle_number' not in patient_data:
                    patient_data['Cycle_number'] = 1
                if 'Protocol' not in patient_data:
                    patient_data['Protocol'] = 'flexible antagonist'
                
                result = predictor.predict_single_patient(patient_data, verbose=False)
                
                predictions.append({
                    'patient_index': idx,
                    'patient_data': patient_data,
                    'predicted_class': result['predicted_response'],
                    'confidence': float(result['confidence']),
                    'probabilities': {k: float(v) for k, v in result['probabilities'].items()},
                    'recommendations': result['recommendations']
                })
            
            except Exception as e:
                errors.append({
                    'patient_index': idx,
                    'error': str(e)
                })
                logger.error(f"❌ Error predicting patient {idx}: {str(e)}")
        
        response = {
            'success': True,
            'total_patients': len(patients),
            'successful_predictions': len(predictions),
            'failed_predictions': len(errors),
            'predictions': predictions
        }
        
        if errors:
            response['errors'] = errors
        
        logger.info(f"✅ Batch prediction completed: {len(predictions)}/{len(patients)} successful")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """
    Informations sur le modèle chargé
    
    Returns:
        JSON avec les détails du modèle
    """
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        info = {
            'success': True,
            'model_type': type(predictor.model).__name__,
            'features': predictor.feature_info.get('feature_columns', []),
            'n_features': len(predictor.feature_info.get('feature_columns', [])),
            'classes': list(predictor.label_encoder.classes_),
            'numerical_features': predictor.feature_info.get('numerical_features', []),
            'categorical_features': predictor.feature_info.get('categorical_features', [])
        }
        
        if predictor.model_metadata:
            info['metadata'] = predictor.model_metadata
        
        return jsonify(info), 200
    
    except Exception as e:
        logger.error(f"❌ Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/validate', methods=['POST'])
def validate_input():
    """
    Valide les données d'entrée sans faire de prédiction
    
    Body JSON: données patient
    
    Returns:
        JSON avec résultat de validation
    """
    try:
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        required_fields = ['Age', 'AMH', 'N_Follicles', 'E2_day5', 'AFC']
        
        # Vérifier les champs manquants
        missing = [f for f in required_fields if f not in patient_data]
        
        # Vérifier les plages de valeurs
        validation_errors = []
        
        if 'Age' in patient_data:
            age = patient_data['Age']
            if not isinstance(age, (int, float)) or not (18 <= age <= 50):
                validation_errors.append("Age must be a number between 18 and 50")
        
        if 'AMH' in patient_data:
            amh = patient_data['AMH']
            if not isinstance(amh, (int, float)) or not (0 <= amh <= 20):
                validation_errors.append("AMH must be a number between 0 and 20 ng/mL")
        
        if 'AFC' in patient_data:
            afc = patient_data['AFC']
            if not isinstance(afc, (int, float)) or not (0 <= afc <= 50):
                validation_errors.append("AFC must be a number between 0 and 50")
        
        if 'N_Follicles' in patient_data:
            n_fol = patient_data['N_Follicles']
            if not isinstance(n_fol, (int, float)) or not (0 <= n_fol <= 50):
                validation_errors.append("N_Follicles must be a number between 0 and 50")
        
        if 'E2_day5' in patient_data:
            e2 = patient_data['E2_day5']
            if not isinstance(e2, (int, float)) or not (0 <= e2 <= 5000):
                validation_errors.append("E2_day5 must be a number between 0 and 5000 pg/mL")
        
        if 'Protocol' in patient_data:
            valid_protocols = ['agonist', 'flex anta', 'fix anta', 
                             'flexible antagonist', 'fixed antagonist']
            protocol = str(patient_data['Protocol']).lower()
            if protocol not in valid_protocols:
                validation_errors.append(f"Protocol must be one of: {', '.join(valid_protocols)}")
        
        is_valid = len(missing) == 0 and len(validation_errors) == 0
        
        return jsonify({
            'success': True,
            'valid': is_valid,
            'missing_fields': missing,
            'validation_errors': validation_errors
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Statistiques de l'API (pour le dashboard)
    
    Returns:
        JSON avec statistiques
    """
    try:
        stats = {
            'success': True,
            'total_predictions': 0,  # À implémenter avec une vraie DB
            'model_accuracy': 0.942,
            'model_loaded': predictor is not None,
            'uptime': 'N/A'  # À implémenter
        }
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# GESTIONNAIRES D'ERREURS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Gestionnaire pour les routes non trouvées"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /api/health',
            'POST /api/predict',
            'POST /api/predict/batch',
            'GET /api/model/info',
            'POST /api/validate',
            'GET /api/stats'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestionnaire pour les erreurs internes"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔬 IVF PATIENT RESPONSE PREDICTION API")
    print("="*80)
    print(f"\n📂 Project root: {project_root}")
    print(f"📂 Model directory: {project_root / 'src' / 'model' / 'saved_models'}")
    print("\n🚀 Initializing predictor...")
    
    if initialize_predictor():
        print("✅ Predictor loaded successfully")
        print(f"📊 Model: {type(predictor.model).__name__}")
        print(f"📊 Features: {len(predictor.feature_info.get('feature_columns', []))}")
        print("\n📡 Available endpoints:")
        print("   - GET  /api/health          : Health check")
        print("   - POST /api/predict         : Single patient prediction")
        print("   - POST /api/predict/batch   : Batch predictions")
        print("   - GET  /api/model/info      : Model information")
        print("   - POST /api/validate        : Validate input data")
        print("   - GET  /api/stats           : API statistics")
        print("\n🌐 Starting Flask server on http://localhost:5000")
        print("="*80 + "\n")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Éviter de charger le modèle 2 fois
        )
    else:
        print("\n" + "="*80)
        print("❌ FAILED TO INITIALIZE PREDICTOR")
        print("="*80)
        print("\n💡 Troubleshooting:")
        print("   1. Vérifiez que les modèles sont entraînés:")
        print("      cd src/model && python train.py && python evaluate.py")
        print("\n   2. Vérifiez que NumPy est compatible:")
        print("      pip install numpy==1.24.3")
        print("\n   3. Vérifiez les chemins:")
        print(f"      Model dir: {project_root / 'src' / 'model' / 'saved_models'}")
        print(f"      best_model.pkl should exist")
        print("\n" + "="*80 + "\n")
        sys.exit(1)