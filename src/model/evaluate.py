import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import IVFDataset  # ton module dataset

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Répertoire où sont sauvegardés les modèles et datasets
SAVED_MODELS_DIR = Path('saved_models')
BEST_MODEL_FILE = SAVED_MODELS_DIR / "best_model.pkl"


def load_models(model_dir: Path) -> dict:
    """Charge tous les modèles sauvegardés dans le dossier."""
    models = {}
    for model_file in model_dir.glob("*_model.pkl"):
        model_name = model_file.stem.replace("_model", "")
        models[model_name] = joblib.load(model_file)
        logger.info(f"Loaded model: {model_name}")
    return models


def evaluate_model(model, X_test, y_test, class_names):
    """Évalue un modèle et affiche les métriques."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model.__class__.__name__}")
    plt.show()

    return {'y_pred': y_pred, 'y_prob': y_prob, 'accuracy': acc, 'precision': precision,
            'recall': recall, 'f1': f1}


def display_sample_probabilities(model, X_test, class_names, n_samples: int = 5):
    """Affiche quelques prédictions avec probabilités."""
    if not hasattr(model, "predict_proba"):
        logger.warning("Model does not support probability predictions.")
        return

    y_prob = model.predict_proba(X_test)
    for i in range(min(n_samples, X_test.shape[0])):
        probs = {class_names[j]: f"{y_prob[i][j]*100:.1f}%" for j in range(len(class_names))}
        logger.info(f"Sample {i}: {probs}")


def plot_metrics_comparison(results: dict):
    """Compare les métriques globales entre tous les modèles."""
    metrics_df = pd.DataFrame(results).T  # Transpose pour avoir modèles en index
    metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1']]

    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Comparaison des modèles - Metrics globales")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.show()


def plot_probability_boxplots(results: dict, class_names: list):
    """Visualise les distributions des probabilités prédictives pour chaque classe et modèle."""
    prob_data = []

    for model_name, res in results.items():
        if res['y_prob'] is not None:
            for i, probs in enumerate(res['y_prob']):
                for j, cls in enumerate(class_names):
                    prob_data.append({'Model': model_name, 'Sample': i, 'Class': cls, 'Probability': probs[j]})

    if prob_data:
        df_probs = pd.DataFrame(prob_data)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Class', y='Probability', hue='Model', data=df_probs)
        plt.title("Distribution des probabilités prédictives par modèle")
        plt.ylabel("Probabilité")
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


def select_best_model(results: dict) -> str:
    """Sélectionne le meilleur modèle basé sur le F1-score pondéré."""
    best_model_name = None
    best_f1 = -1
    for name, res in results.items():
        if res['f1'] > best_f1:
            best_f1 = res['f1']
            best_model_name = name
    logger.info(f"\n✅ Meilleur modèle sélectionné : {best_model_name} avec F1-score pondéré = {best_f1:.4f}")
    return best_model_name


def save_best_model(model, file_path: Path):
    """Sauvegarde le meilleur modèle pour usage futur."""
    joblib.dump(model, file_path)
    logger.info(f"Le meilleur modèle a été sauvegardé dans : {file_path}")


def main():
    # Charger les datasets sauvegardés
    dataset = IVFDataset("dummy_path.csv")
    data = dataset.load_datasets(SAVED_MODELS_DIR)
    X_test = data['X_test']
    y_test = data['y_test']
    class_names = data['class_names']

    logger.info(f"Loaded test set: X_test={X_test.shape}, y_test={y_test.shape}")

    # Charger les modèles
    models = load_models(SAVED_MODELS_DIR)

    # Évaluer chaque modèle
    results = {}
    for name, model in models.items():
        logger.info(f"\n{'='*50}\nEvaluating {name}\n{'='*50}")
        res = evaluate_model(model, X_test, y_test, class_names)
        display_sample_probabilities(model, X_test, class_names, n_samples=5)
        results[name] = res

    # Visualisations comparatives
    plot_metrics_comparison(results)
    plot_probability_boxplots(results, class_names)

    # Sélection et sauvegarde du meilleur modèle
    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    logger.info(f"Description du meilleur modèle ({best_model_name}): {best_model}")

    save_best_model(best_model, BEST_MODEL_FILE)


if __name__ == "__main__":
    main()
