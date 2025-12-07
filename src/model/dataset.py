import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IVFDataset:
    """
    Dataset handler pour les données IVF nettoyées et normalisées.
    
    Ce gestionnaire utilise les colonnes déjà normalisées (_normalized)
    et encodées (_encoded) créées lors du nettoyage des données.
    
    Attributes:
        data_path (str): Chemin vers le CSV nettoyé
        target_column (str): Nom de la colonne cible
        feature_columns (list): Liste des colonnes de features à utiliser
    """
    
    def __init__(self, data_path: str, target_column: str = 'Patient_Response_encoded'):
        """
        Initialise le gestionnaire de dataset.
        
        Args:
            data_path: Chemin vers le CSV nettoyé (patients_ivf_cleaned.csv)
            target_column: Colonne cible (par défaut 'Patient_Response_encoded')
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.feature_columns = []
        self.original_features = []
        self.class_names = ['low', 'optimal', 'high']  # Mapping pour les classes
        
        logger.info(f"Initialized IVF dataset handler for: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Charge le dataset nettoyé depuis le CSV.
        
        Returns:
            DataFrame contenant les données chargées
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Essayer de trouver la colonne cible avec différentes variations
        possible_target_names = [
            'Patient_Response_encoded',
            'Patient_Response',
            'Patient Response',
            'patient_response',
            'response'
        ]
        
        target_found = False
        for col_name in possible_target_names:
            if col_name in df.columns:
                if col_name != self.target_column:
                    logger.warning(f"Target column '{self.target_column}' not found, using '{col_name}'")
                    self.target_column = col_name
                target_found = True
                break
        
        if not target_found:
            raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")
        
        return df
    
    def identify_features(self, df: pd.DataFrame, use_normalized: bool = True) -> None:
        """
        Identifie les colonnes de features à utiliser.
        
        Args:
            df: DataFrame d'entrée
            use_normalized: Si True, utilise les colonnes _normalized, sinon les originales
        """
        # Colonnes à exclure absolument (insensible à la casse et aux underscores)
        exclude_patterns = [
            'patient_id',
            'patient id',
            'patient_response',
            'patient response',
            'protocol',
            'amh',
            'e2_day5',
        ]
        
        if use_normalized:
            # Utiliser les colonnes normalisées et encodées
            candidates = []
            for col in df.columns:
                col_lower = col.lower().replace('_', ' ')
                
                # Vérifier si la colonne doit être exclue
                should_exclude = any(pattern in col_lower for pattern in exclude_patterns)
                
                if not should_exclude:
                    # Inclure les colonnes normalisées, encodées et robust
                    if '_normalized' in col or '_encoded' in col or '_robust' in col:
                        candidates.append(col)
                    # Inclure aussi cycle_number qui est numérique
                    elif 'cycle' in col.lower() and df[col].dtype in ['int64', 'float64']:
                        candidates.append(col)
            
            self.feature_columns = candidates
            
            # Si pas de colonnes normalisées, utiliser les numériques
            if len(self.feature_columns) == 0:
                logger.warning("No normalized columns found, using numeric columns")
                self.feature_columns = [
                    col for col in df.columns 
                    if not any(pattern in col.lower().replace('_', ' ') for pattern in exclude_patterns)
                    and df[col].dtype in ['int64', 'float64']
                ]
        else:
            # Utiliser les colonnes originales (numériques uniquement)
            self.feature_columns = [
                col for col in df.columns 
                if not any(pattern in col.lower().replace('_', ' ') for pattern in exclude_patterns)
                and df[col].dtype in ['int64', 'float64']
                and '_normalized' not in col
                and '_encoded' not in col
                and '_robust' not in col
            ]
        
        logger.info(f"Selected {len(self.feature_columns)} features:")
        for col in self.feature_columns:
            logger.info(f"  - {col}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        use_normalized: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les données X et y pour l'entraînement.
        
        Args:
            df: DataFrame d'entrée
            use_normalized: Utiliser les colonnes normalisées
            
        Returns:
            Tuple (X, y) avec features et target
        """
        # Identifier les features si pas encore fait
        if len(self.feature_columns) == 0:
            self.identify_features(df, use_normalized=use_normalized)
        
        # Extraire X (features)
        X = df[self.feature_columns].values
        
        # Extraire et encoder y (target)
        if '_encoded' in self.target_column or df[self.target_column].dtype in ['int64', 'float64']:
            # Déjà encodé numériquement (0, 1, 2)
            y = df[self.target_column].values.astype(int)
            logger.info(f"Target already encoded: {np.unique(y)}")
        else:
            # Encoder la variable texte
            label_map = {'low': 0, 'optimal': 1, 'high': 2}
            
            # Vérifier les valeurs uniques avant encodage
            unique_values = df[self.target_column].unique()
            logger.info(f"Target unique values before encoding: {unique_values}")
            
            # Encoder
            y = df[self.target_column].map(label_map).values
            
            # Vérifier s'il y a des valeurs non mappées (NaN)
            if np.isnan(y).any():
                n_nan = np.isnan(y).sum()
                logger.error(f"Found {n_nan} unmapped values in target variable")
                logger.error(f"Valid values are: {list(label_map.keys())}")
                
                # Afficher les valeurs problématiques
                unmapped = df[self.target_column][pd.isna(y)].unique()
                logger.error(f"Unmapped values: {unmapped}")
                raise ValueError(f"Target variable contains unmapped values: {unmapped}")
            
            y = y.astype(int)
            logger.info(f"Target encoded successfully: {label_map}")
            logger.info(f"Encoded distribution: {np.bincount(y)}")
        
        # Vérifier les valeurs manquantes dans X
        if np.isnan(X).any():
            logger.warning(f"Found {np.isnan(X).sum()} NaN values in features")
            # Remplacer par 0 (les colonnes normalisées ne devraient pas avoir de NaN)
            X = np.nan_to_num(X, nan=0.0)
        
        logger.info(f"Prepared data: X shape = {X.shape}, y shape = {y.shape}")
        logger.info(f"Target classes: {np.unique(y)}")
        
        return X, y
    
    def split_data(
        self, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
        use_normalized: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Sépare les données en ensembles train/val/test.
        
        Args:
            test_size: Proportion pour le test set
            val_size: Proportion pour le validation set
            random_state: Graine aléatoire
            stratify: Stratifier selon la variable cible
            use_normalized: Utiliser les colonnes normalisées
            
        Returns:
            Dictionnaire avec X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Charger les données
        df = self.load_data()
        
        # Préparer X et y
        X, y = self.prepare_data(df, use_normalized=use_normalized)
        
        # Stratification
        stratify_array = y if stratify else None
        
        # Premier split : séparer test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_array
        )
        
        # Deuxième split : séparer validation set
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            stratify_train = y_train_val if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_train
            )
        else:
            X_train, y_train = X_train_val, y_train_val
            X_val, y_val = np.array([]), np.array([])
        
        logger.info(f"Data split:")
        logger.info(f"  - Train: {X_train.shape} samples")
        logger.info(f"  - Val:   {X_val.shape if len(X_val) > 0 else (0,)} samples")
        logger.info(f"  - Test:  {X_test.shape} samples")
        
        # Distribution des classes
        logger.info(f"\nClass distribution:")
        logger.info(f"  - Train: {np.bincount(y_train.astype(int))}")
        if len(y_val) > 0:
            logger.info(f"  - Val:   {np.bincount(y_val.astype(int))}")
        logger.info(f"  - Test:  {np.bincount(y_test.astype(int))}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_columns,
            'class_names': self.class_names
        }
    
    def get_class_distribution(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Obtient la distribution des classes dans la variable cible.
        
        Args:
            df: DataFrame à analyser (si None, charge depuis data_path)
            
        Returns:
            Series avec le compte des classes
        """
        if df is None:
            df = self.load_data()
        
        # Vérifier si la colonne cible existe
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found in DataFrame")
            logger.info(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        if '_encoded' in self.target_column or df[self.target_column].dtype in ['int64', 'float64']:
            # Si encodé numériquement, mapper les valeurs aux noms
            value_map = {0: 'low', 1: 'optimal', 2: 'high'}
            distribution = df[self.target_column].map(value_map).value_counts()
        else:
            # Sinon utiliser directement
            distribution = df[self.target_column].value_counts()
        
        logger.info(f"Class distribution:\n{distribution}")
        
        return distribution
    
    def save_feature_info(self, save_dir: str = 'saved_models') -> None:
        """
        Sauvegarde les informations sur les features.
        
        Args:
            save_dir: Répertoire de sauvegarde
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        feature_info = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'class_names': self.class_names,
            'n_features': len(self.feature_columns)
        }
        
        joblib.dump(feature_info, save_path / 'feature_info.pkl')
        logger.info(f"Saved feature info to {save_path / 'feature_info.pkl'}")
    
    def save_datasets(self, data_dict: Dict[str, np.ndarray], save_dir: str = 'saved_models') -> None:
        """
        Sauvegarde les datasets train/val/test.
        
        Args:
            data_dict: Dictionnaire contenant X_train, X_val, X_test, y_train, y_val, y_test
            save_dir: Répertoire de sauvegarde
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder chaque dataset
        np.save(save_path / 'X_train.npy', data_dict['X_train'])
        np.save(save_path / 'y_train.npy', data_dict['y_train'])
        
        if len(data_dict['X_val']) > 0:
            np.save(save_path / 'X_val.npy', data_dict['X_val'])
            np.save(save_path / 'y_val.npy', data_dict['y_val'])
        
        np.save(save_path / 'X_test.npy', data_dict['X_test'])
        np.save(save_path / 'y_test.npy', data_dict['y_test'])
        
        # Sauvegarder les métadonnées
        metadata = {
            'X_train_shape': data_dict['X_train'].shape,
            'X_val_shape': data_dict['X_val'].shape if len(data_dict['X_val']) > 0 else (0,),
            'X_test_shape': data_dict['X_test'].shape,
            'feature_names': data_dict['feature_names'],
            'class_names': data_dict['class_names'],
            'train_distribution': np.bincount(data_dict['y_train'].astype(int)).tolist(),
            'test_distribution': np.bincount(data_dict['y_test'].astype(int)).tolist()
        }
        
        if len(data_dict['X_val']) > 0:
            metadata['val_distribution'] = np.bincount(data_dict['y_val'].astype(int)).tolist()
        
        joblib.dump(metadata, save_path / 'dataset_metadata.pkl')
        
        logger.info(f"\n💾 Datasets sauvegardés dans {save_path}:")
        logger.info(f"  • X_train.npy : {data_dict['X_train'].shape}")
        logger.info(f"  • y_train.npy : {data_dict['y_train'].shape}")
        if len(data_dict['X_val']) > 0:
            logger.info(f"  • X_val.npy   : {data_dict['X_val'].shape}")
            logger.info(f"  • y_val.npy   : {data_dict['y_val'].shape}")
        logger.info(f"  • X_test.npy  : {data_dict['X_test'].shape}")
        logger.info(f"  • y_test.npy  : {data_dict['y_test'].shape}")
        logger.info(f"  • dataset_metadata.pkl")
    
    def load_feature_info(self, load_dir: str = 'saved_models') -> None:
        """
        Charge les informations sur les features.
        
        Args:
            load_dir: Répertoire de chargement
        """
        load_path = Path(load_dir)
        
        if not (load_path / 'feature_info.pkl').exists():
            raise FileNotFoundError(f"Feature info not found in {load_path}")
        
        feature_info = joblib.load(load_path / 'feature_info.pkl')
        
        self.feature_columns = feature_info['feature_columns']
        self.target_column = feature_info['target_column']
        self.class_names = feature_info['class_names']
        
        logger.info(f"Loaded feature info from {load_path / 'feature_info.pkl'}")
        logger.info(f"  - {len(self.feature_columns)} features")
    
    def load_datasets(self, load_dir: str = 'saved_models') -> Dict[str, np.ndarray]:
        """
        Charge les datasets train/val/test sauvegardés.
        
        Args:
            load_dir: Répertoire de chargement
            
        Returns:
            Dictionnaire contenant les datasets
        """
        load_path = Path(load_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Directory not found: {load_path}")
        
        # Charger les datasets
        X_train = np.load(load_path / 'X_train.npy')
        y_train = np.load(load_path / 'y_train.npy')
        X_test = np.load(load_path / 'X_test.npy')
        y_test = np.load(load_path / 'y_test.npy')
        
        # Charger validation si existe
        if (load_path / 'X_val.npy').exists():
            X_val = np.load(load_path / 'X_val.npy')
            y_val = np.load(load_path / 'y_val.npy')
        else:
            X_val = np.array([])
            y_val = np.array([])
        
        # Charger métadonnées
        metadata = joblib.load(load_path / 'dataset_metadata.pkl')
        
        logger.info(f"\n📂 Datasets chargés depuis {load_path}:")
        logger.info(f"  • X_train : {X_train.shape}")
        logger.info(f"  • y_train : {y_train.shape}")
        if len(X_val) > 0:
            logger.info(f"  • X_val   : {X_val.shape}")
            logger.info(f"  • y_val   : {y_val.shape}")
        logger.info(f"  • X_test  : {X_test.shape}")
        logger.info(f"  • y_test  : {y_test.shape}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': metadata['feature_names'],
            'class_names': metadata['class_names']
        }
    
    def get_feature_statistics(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Obtient des statistiques sur les features.
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            DataFrame avec statistiques
        """
        if df is None:
            df = self.load_data()
        
        if len(self.feature_columns) == 0:
            self.identify_features(df)
        
        stats = df[self.feature_columns].describe().T
        stats['missing'] = df[self.feature_columns].isnull().sum()
        stats['missing_pct'] = (stats['missing'] / len(df)) * 100
        
        logger.info(f"\nFeature statistics:")
        print(stats)
        
        return stats


# =============================================================================
# FONCTION UTILITAIRE
# =============================================================================

def load_ivf_dataset(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    use_normalized: bool = True
) -> Dict[str, np.ndarray]:
    """
    Fonction utilitaire pour charger rapidement le dataset IVF.
    
    Args:
        csv_path: Chemin vers le CSV nettoyé
        test_size: Proportion du test set
        val_size: Proportion du validation set
        random_state: Graine aléatoire
        use_normalized: Utiliser les colonnes normalisées
        
    Returns:
        Dictionnaire avec les données préparées
    """
    dataset = IVFDataset(csv_path)
    return dataset.split_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        use_normalized=use_normalized
    )


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("🔬 IVF DATASET LOADER - VERSION NORMALISÉE")
    print("="*80 + "\n")
    
    # Chemin vers le CSV nettoyé
    csv_path = "C:\\Users\\yesmine\\Desktop\\Tanit\\data\\processed\\patients_medical_clean.csv" 
    
    # Initialiser le dataset
    dataset = IVFDataset(csv_path, target_column='Patient_Response_encoded')
    
    # Option 1 : Charger et explorer
    print("📊 Exploration du dataset :")
    print("-"*80)
    df = dataset.load_data()
    print(f"  • Shape : {df.shape}")
    print(f"  • Colonnes : {list(df.columns)}")
    print()
    
    # Distribution des classes
    print("📈 Distribution des classes :")
    print("-"*80)
    dataset.get_class_distribution(df)
    print()
    
    # Identifier les features
    print("🔍 Identification des features :")
    print("-"*80)
    dataset.identify_features(df, use_normalized=True)
    print()
    
    # Option 2 : Préparer les données pour ML
    print("🚀 Préparation des données pour Machine Learning :")
    print("-"*80)
    data_dict = dataset.split_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        use_normalized=True
    )
    
    print(f"\n✅ Données prêtes !")
    print(f"  • X_train shape : {data_dict['X_train'].shape}")
    print(f"  • X_val shape   : {data_dict['X_val'].shape}")
    print(f"  • X_test shape  : {data_dict['X_test'].shape}")
    print(f"  • Features      : {len(data_dict['feature_names'])}")
    print(f"  • Classes       : {data_dict['class_names']}")
    
    print("\n" + "="*80)
    print("✅ DATASET PRÊT POUR L'ENTRAÎNEMENT !")
    print("="*80)
    
    # Sauvegarder les infos
    dataset.save_feature_info('saved_models')
    
    # Sauvegarder les datasets
    print("\n💾 Sauvegarde des datasets...")
    print("-"*80)
    dataset.save_datasets(data_dict, 'saved_models')
    
    # Test de chargement
    print("\n📂 Test de chargement des datasets...")
    print("-"*80)
    loaded_data = dataset.load_datasets('saved_models')
    print(f"✅ Chargement réussi !")
    print(f"  • X_train : {loaded_data['X_train'].shape}")
    print(f"  • X_test  : {loaded_data['X_test'].shape}")
   
    # Exemple d'utilisation rapide
    print("\n📝 Exemple d'utilisation rapide :")
    print("-"*80)
    print("""
# Méthode 1 : Créer et sauvegarder les datasets
from ivf_dataset import IVFDataset

dataset = IVFDataset('patients_ivf_cleaned.csv')
data = dataset.split_data(test_size=0.2, val_size=0.1)
dataset.save_datasets(data, 'saved_models')

# Méthode 2 : Charger les datasets sauvegardés
dataset = IVFDataset('patients_ivf_cleaned.csv')
data = dataset.load_datasets('saved_models')

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Méthode 3 : Utilisation rapide (une ligne)
from ivf_dataset import load_ivf_dataset

data = load_ivf_dataset(
    csv_path='patients_ivf_cleaned.csv',
    test_size=0.2,
    val_size=0.1,
    random_state=42
)
    """)
    print("-"*80)