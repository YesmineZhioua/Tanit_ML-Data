"""
Patient Response Stratification - Training Module
=================================================
Train multiple models with hyperparameter tuning and save them for later evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime

from dataset import IVFDataset  # ton module pour charger et préparer les données

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer class for patient response stratification."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.training_history = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize candidate models with hyperparameter grids."""
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000, multi_class='multinomial'),
                'param_grid': {'C': [0.1, 1.0, 10.0], 'penalty': ['l2'], 'solver': ['lbfgs', 'saga']}
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'param_grid': {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'param_grid': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
            },
            'adaboost': {
                'model': AdaBoostClassifier(random_state=self.random_state),
                'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
            },
            'svc': {
                'model': SVC(probability=True, random_state=self.random_state),
                'param_grid': {'C': [0.1, 1], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
            }
        }

    def train_all_models(self, X, y, cv_folds: int = 5):
        """Train all models with GridSearchCV and save them."""
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for model_name, config in self.models.items():
            logger.info(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
            try:
                grid = GridSearchCV(config['model'], config['param_grid'],
                                    cv=cv_strategy, scoring='f1_weighted', n_jobs=-1, verbose=1)
                grid.fit(X, y)
                best_model = grid.best_estimator_
                best_params = grid.best_params_

                # Save model and hyperparameters
                save_dir = Path('saved_models')
                save_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(best_model, save_dir / f'{model_name}_model.pkl')
                with open(save_dir / f'{model_name}_params.json', 'w') as f:
                    json.dump({'best_params': best_params, 'timestamp': datetime.now().isoformat()}, f, indent=4)

                logger.info(f"{model_name} trained and saved with best params: {best_params}")
                self.training_history[model_name] = {'best_params': best_params, 'timestamp': datetime.now().isoformat()}

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")


def main():
    # Charger le dataset
    csv_path = 'C:\\Users\\yesmine\\Desktop\\Tanit\\data\\processed\\patients_medical_clean.csv'
    dataset = IVFDataset(csv_path)
    data = dataset.split_data(test_size=0.2, val_size=0.1, use_normalized=True)

    # Entraîner et sauvegarder tous les modèles
    trainer = ModelTrainer(random_state=42)
    trainer.train_all_models(X=data['X_train'], y=data['y_train'])


if __name__ == "__main__":
    main()
