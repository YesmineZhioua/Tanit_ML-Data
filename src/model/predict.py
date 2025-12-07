import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import joblib
import json
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatientResponsePredictor:
    """
    Predictor class for patient response stratification.
    """

    def __init__(self, model_dir: str = 'C:\\Users\\yesmine\\Desktop\\Tanit\\src\\model\\saved_models'):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_info = None
        self.model_metadata = None
        logger.info(f"Initializing predictor from {self.model_dir}")

    def load_model(self, model_name: str = None) -> None:
        """
        Load trained model and all required preprocessors.
        """
        try:
            self.scaler = joblib.load(self.model_dir / 'C:\\Users\\yesmine\\Desktop\\Tanit\\src\\model\\saved_models\\scaler.pkl')
            self.label_encoder = joblib.load(self.model_dir / 'C:\\Users\\yesmine\\Desktop\\Tanit\\src\\model\\saved_models\\label_encoder.pkl')
            self.feature_info = joblib.load(self.model_dir / 'C:\\Users\\yesmine\\Desktop\\Tanit\\src\\model\\saved_models\\feature_info.pkl')
            logger.info("Loaded preprocessors successfully")
            logger.info(f"Expected features: {self.feature_info['feature_columns']}")
            logger.info(f"Categorical features: {self.feature_info.get('categorical_features', [])}")
            logger.info(f"Numerical features: {self.feature_info.get('numerical_features', [])}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Preprocessor files not found in {self.model_dir}. "
                "Please ensure the model has been trained and saved."
            ) from e

        if model_name is None or model_name.strip() == "":
            best_model_path = self.model_dir / 'C:\\Users\\yesmine\\Desktop\\Tanit\\src\\model\\saved_models\\best_model.pkl'
            if not best_model_path.exists():
                raise FileNotFoundError(f"Best model file not found: {best_model_path}")
            self.model = joblib.load(best_model_path)
            logger.info(f"Loaded BEST model from {best_model_path}")
            model_name = 'best_model'
        else:
            model_path = self.model_dir / f'{model_name}_model.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model: {model_name}")

        # Load metadata if exists
        metadata_path = self.model_dir / f'{model_name}_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            logger.info("Loaded model metadata")

    def preprocess_input(self, input_data: Union[pd.DataFrame, Dict, np.ndarray]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        """
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, np.ndarray):
            input_df = pd.DataFrame(input_data, columns=self.feature_info['feature_columns'])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        logger.info(f"Input data shape before preprocessing: {input_df.shape}")
        logger.info(f"Input columns: {input_df.columns.tolist()}")
        logger.info(f"Input data:\n{input_df}")

        # Get expected features from the model
        expected_features = self.feature_info['feature_columns']
        logger.info(f"Expected features by model: {expected_features}")

        # Create a new dataframe with expected features
        processed_df = pd.DataFrame()

        # Handle cycle_number
        if 'cycle_number' in expected_features:
            if 'Cycle_number' in input_df.columns:
                processed_df['cycle_number'] = input_df['Cycle_number']
            elif 'cycle_number' in input_df.columns:
                processed_df['cycle_number'] = input_df['cycle_number']
            else:
                processed_df['cycle_number'] = 0
                logger.warning("cycle_number not found in input, using 0")

        # Prepare data for scaling (using original column names)
        scaling_data = pd.DataFrame()
        original_to_robust = {
            'Age': 'Age_robust',
            'AMH': 'AMH_numeric_robust',
            'N_Follicles': 'n_Follicles_robust',
            'E2_day5': 'E2_day5_numeric_robust',
            'AFC': 'AFC_robust'
        }

        for original_col, robust_col in original_to_robust.items():
            if original_col in input_df.columns:
                scaling_data[original_col] = input_df[original_col]

        logger.info(f"Data to scale:\n{scaling_data}")

        # Apply scaling
        if len(scaling_data.columns) > 0:
            scaled_values = self.scaler.transform(scaling_data)
            logger.info(f"Scaled values:\n{scaled_values}")
            
            # Map scaled values to robust feature names
            for idx, (original_col, robust_col) in enumerate(original_to_robust.items()):
                if robust_col in expected_features and original_col in scaling_data.columns:
                    col_idx = scaling_data.columns.get_loc(original_col)
                    processed_df[robust_col] = scaled_values[:, col_idx]
                    logger.info(f"Mapped {original_col} -> {robust_col}: {scaled_values[:, col_idx]}")

        # Handle Protocol encoding if needed
        if 'Protocol' in input_df.columns:
            protocol_value = input_df['Protocol'].iloc[0]
            logger.info(f"Protocol value: {protocol_value}")
            
            # Map protocol to encoded value
            protocol_mapping = {
                'agonist': 0,
                'flexible antagonist': 1,
                'flex anta': 1,
                'fixed antagonist': 2,
                'fix anta': 2
            }
            
            protocol_normalized = protocol_value.lower().strip()
            if protocol_normalized in protocol_mapping:
                if 'Protocol_encoded' in expected_features:
                    processed_df['Protocol_encoded'] = protocol_mapping[protocol_normalized]
                    logger.info(f"Protocol encoded: {protocol_mapping[protocol_normalized]}")

        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in processed_df.columns:
                logger.warning(f"Missing feature '{feature}', setting to 0")
                processed_df[feature] = 0

        # Reorder columns to match expected features
        processed_df = processed_df[expected_features]
        
        logger.info(f"Final preprocessed shape: {processed_df.shape}")
        logger.info(f"Final preprocessed data:\n{processed_df}")

        return processed_df.values

    def predict(self, input_data: Union[pd.DataFrame, Dict, np.ndarray], return_proba: bool = True) -> Dict[str, Any]:
        """
        Make predictions for input patient data.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        X = self.preprocess_input(input_data)
        predictions = self.model.predict(X)
        predicted_classes = self.label_encoder.inverse_transform(predictions)

        result = {
            'predicted_class': predicted_classes.tolist(),
            'predicted_class_index': predictions.tolist()
        }

        if return_proba:
            probabilities = self.model.predict_proba(X)
            proba_dicts = [
                {class_name: float(prob) for class_name, prob in zip(self.label_encoder.classes_, proba)}
                for proba in probabilities
            ]
            result['probabilities'] = proba_dicts
            result['confidence'] = [float(np.max(proba)) for proba in probabilities]

        return result

    def predict_single_patient(self, patient_data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Make prediction for a single patient with detailed output.
        """
        result = self.predict(patient_data, return_proba=True)
        predicted_class = result['predicted_class'][0]
        probabilities = result['probabilities'][0]
        confidence = result['confidence'][0]

        detailed_result = {
            'predicted_response': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'clinical_interpretation': self._generate_clinical_interpretation(predicted_class, probabilities, confidence),
            'recommendations': self._generate_recommendations(predicted_class, confidence)
        }

        if verbose:
            self._print_prediction_summary(detailed_result)

        return detailed_result

    def predict_batch(self, patient_data: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions for multiple patients.
        """
        result = self.predict(patient_data, return_proba=True)
        results_df = patient_data.copy()
        results_df['Predicted_Response'] = result['predicted_class']
        results_df['Confidence'] = result['confidence']

        for class_name in self.label_encoder.classes_:
            results_df[f'Prob_{class_name}'] = [proba[class_name] for proba in result['probabilities']]

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(save_path, index=False)
            logger.info(f"Saved batch predictions to {save_path}")

        return results_df

    def _generate_clinical_interpretation(self, predicted_class: str, probabilities: Dict[str, float], confidence: float) -> str:
        """
        Generate clinical interpretation of prediction.
        """
        interpretation_parts = [
            f"The patient is predicted to be a {predicted_class.upper()} responder with {confidence*100:.1f}% confidence.",
            "Probability breakdown: " + ", ".join([f"{cls}: {prob*100:.1f}%" for cls, prob in probabilities.items()])
        ]
        if confidence >= 0.8:
            interpretation_parts.append("Confidence assessment: HIGH - prediction is reliable")
        elif confidence >= 0.6:
            interpretation_parts.append("Confidence assessment: MODERATE - prediction is reasonably reliable")
        else:
            interpretation_parts.append("Confidence assessment: LOW - consider additional clinical assessment")

        return "\n".join(interpretation_parts)

    def _generate_recommendations(self, predicted_class: str, confidence: float) -> List[str]:
        """
        Generate clinical recommendations based on prediction and confidence.
        """
        recs = []

        # Recommendations by class
        if predicted_class.lower() == 'low':
            recs.extend([
                "Consider higher starting dose of gonadotropins",
                "Plan for more frequent monitoring",
                "Counsel patient about potential for cycle cancellation",
                "Evaluate for additional ovarian reserve testing",
                "Consider alternative protocols (e.g., microdose flare)"
            ])
        elif predicted_class.lower() == 'high':
            recs.extend([
                "Use lower starting dose to reduce OHSS risk",
                "Consider GnRH agonist trigger for final maturation",
                "Plan for close monitoring of estradiol levels",
                "Counsel patient about OHSS symptoms",
                "Consider 'freeze-all' strategy if OHSS risk is high"
            ])
        else:  # optimal
            recs.extend([
                "Standard protocol and dosing appropriate",
                "Regular monitoring schedule",
                "Good prognosis for treatment success",
                "Proceed with standard care pathway"
            ])

        return recs

    def _print_prediction_summary(self, result: Dict[str, Any]) -> None:
        """
        Print formatted prediction summary.
        """
        print("\n" + "="*80)
        print("PATIENT RESPONSE PREDICTION SUMMARY")
        print("="*80)
        print(f"\nPredicted Response: {result['predicted_response'].upper()}")
        print(f"Confidence: {result['confidence']*100:.2f}%\n")
        print("Class Probabilities:")
        for class_name, prob in result['probabilities'].items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {class_name:10s}: {bar} {prob*100:.2f}%")
        print("\nCLINICAL INTERPRETATION:")
        print(result['clinical_interpretation'])
        print("\nCLINICAL RECOMMENDATIONS:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")
        print("\n" + "="*80 + "\n")


def main():
    """Example usage of the predictor."""
    logger.info("Starting prediction pipeline")

    predictor = PatientResponsePredictor()
    predictor.load_model()

    # Single patient example - utilisez les noms de colonnes ORIGINAUX
    patient_data = {
        'Age': 24,
        'Cycle_number': 1,
        'Protocol': 'flex anta',
        'AMH': 1.81,
        'N_Follicles': 18,
        'E2_day5': 351.81,
        'AFC': 17
    }
    
    predictor.predict_single_patient(patient_data, verbose=True)

    logger.info("Prediction pipeline completed successfully")


if __name__ == "__main__":
    main()