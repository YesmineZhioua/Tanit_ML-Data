import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("📊 PIPELINE COMPLET : EDA + NETTOYAGE + VISUALISATION")
print("="*80 + "\n")


# =============================================================================
# PARTIE 1 : ANALYSE EXPLORATOIRE AVANT NETTOYAGE
# =============================================================================

def load_and_explore_data(csv_path):
    """Charge et explore les données initiales"""
    print("="*80)
    print("1️⃣ CHARGEMENT ET EXPLORATION INITIALE")
    print("="*80 + "\n")
    
    df = pd.read_csv(csv_path)
    
    print(f"✓ Données chargées : {csv_path}")
    print(f"  • Nombre de patients : {len(df)}")
    print(f"  • Nombre de colonnes : {len(df.columns)}")
    print(f"  • Taille mémoire : {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n")
    
    print("📋 APERÇU DES DONNÉES (5 premières lignes) :")
    print("-"*80)
    print(df.head())
    print("-"*80 + "\n")
    
    print("🔍 TYPES DE DONNÉES :")
    print("-"*80)
    print(df.dtypes)
    print("-"*80 + "\n")
    
    print("ℹ️ INFORMATIONS GÉNÉRALES :")
    print("-"*80)
    df.info()
    print("-"*80 + "\n")
    
    print("📈 STATISTIQUES DESCRIPTIVES :")
    print("-"*80)
    print(df.describe())
    print("-"*80 + "\n")
    
    return df


def analyze_missing_values(df):
    """Analyse détaillée des valeurs manquantes"""
    print("="*80)
    print("2️⃣ ANALYSE DES VALEURS MANQUANTES")
    print("="*80 + "\n")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Colonne': missing.index,
        'Valeurs manquantes': missing.values,
        'Pourcentage (%)': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Valeurs manquantes'] > 0].sort_values(
        'Valeurs manquantes', ascending=False
    )
    
    if len(missing_df) > 0:
        print("⚠️ VALEURS MANQUANTES DÉTECTÉES :")
        print("-"*80)
        print(missing_df.to_string(index=False))
        print("-"*80 + "\n")
        
        # Visualisation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        missing_df.plot(x='Colonne', y='Valeurs manquantes', kind='bar', ax=ax1, color='coral')
        ax1.set_title('Nombre de valeurs manquantes par colonne', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Nombre')
        ax1.set_xlabel('Colonnes')
        ax1.tick_params(axis='x', rotation=45)
        
        missing_df.plot(x='Colonne', y='Pourcentage (%)', kind='bar', ax=ax2, color='salmon')
        ax2.set_title('Pourcentage de valeurs manquantes par colonne', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Pourcentage (%)')
        ax2.set_xlabel('Colonnes')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=40, color='red', linestyle='--', label='Seuil critique 40%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('eda_01_missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Graphique sauvegardé : eda_01_missing_values_analysis.png\n")
        
        # Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='RdYlGn_r')
        plt.title('Heatmap des valeurs manquantes (rouge = manquant)', fontsize=14, fontweight='bold')
        plt.xlabel('Colonnes')
        plt.ylabel('Patients')
        plt.tight_layout()
        plt.savefig('eda_02_missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Graphique sauvegardé : eda_02_missing_values_heatmap.png\n")
    else:
        print("✅ Aucune valeur manquante détectée !\n")
    
    return missing_df


def analyze_distributions(df):
    """Analyse les distributions des variables numériques"""
    print("="*80)
    print("3️⃣ ANALYSE DES DISTRIBUTIONS")
    print("="*80 + "\n")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Patient_id' in numeric_cols:
        numeric_cols.remove('Patient_id')
    
    print(f"📊 Variables numériques : {', '.join(numeric_cols)}\n")
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution de {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Fréquence')
            
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Moyenne: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                      label=f'Médiane: {median_val:.2f}')
            ax.legend()
    
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('eda_03_distributions_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Graphique sauvegardé : eda_03_distributions_analysis.png\n")


def analyze_categorical(df):
    """Analyse les variables catégorielles"""
    print("="*80)
    print("4️⃣ ANALYSE DES VARIABLES CATÉGORIELLES")
    print("="*80 + "\n")
    
    categorical_cols = ['Protocol', 'Patient_Response']
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"📊 Distribution de {col} :")
            print("-"*80)
            value_counts = df[col].value_counts(dropna=False)
            value_pct = df[col].value_counts(dropna=False, normalize=True) * 100
            
            result = pd.DataFrame({
                'Valeur': value_counts.index,
                'Nombre': value_counts.values,
                'Pourcentage (%)': value_pct.values
            })
            print(result.to_string(index=False))
            print("-"*80 + "\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'Protocol' in df.columns:
        df['Protocol'].value_counts().plot(kind='bar', ax=axes[0], color='lightcoral')
        axes[0].set_title('Distribution des Protocoles', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Protocol')
        axes[0].set_ylabel('Nombre de patients')
        axes[0].tick_params(axis='x', rotation=45)
    
    if 'Patient_Response' in df.columns:
        df['Patient_Response'].value_counts().plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Distribution des Réponses Patients', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Patient Response')
        axes[1].set_ylabel('Nombre de patients')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda_04_categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Graphique sauvegardé : eda_04_categorical_analysis.png\n")


def detect_outliers(df):
    """Détecte les valeurs aberrantes"""
    print("="*80)
    print("5️⃣ DÉTECTION DES OUTLIERS")
    print("="*80 + "\n")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Patient_id' in numeric_cols:
        numeric_cols.remove('Patient_id')
    if 'Cycle_number' in numeric_cols:
        numeric_cols.remove('Cycle_number')
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    print("📊 Outliers détectés par variable :")
    print("-"*80)
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        if len(data) > 0:
            ax.boxplot(data, vert=True)
            ax.set_title(f'Boxplot de {col}', fontweight='bold')
            ax.set_ylabel(col)
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            print(f"  • {col:20s} : {len(outliers)} outliers")
            
            if len(outliers) > 0:
                ax.text(0.5, 0.95, f'{len(outliers)} outliers', 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    print("-"*80 + "\n")
    
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('eda_05_outliers_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Graphique sauvegardé : eda_05_outliers_detection.png\n")


def analyze_correlations(df):
    """Analyse les corrélations entre variables"""
    print("="*80)
    print("6️⃣ ANALYSE DES CORRÉLATIONS")
    print("="*80 + "\n")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Patient_id' in numeric_cols:
        numeric_cols.remove('Patient_id')
    
    corr_matrix = df[numeric_cols].corr()
    
    print("📊 MATRICE DE CORRÉLATION :")
    print("-"*80)
    print(corr_matrix.round(3))
    print("-"*80 + "\n")
    
    print("🔍 CORRÉLATIONS FORTES (|r| > 0.5) :")
    print("-"*80)
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                direction = "positive" if corr_val > 0 else "négative"
                emoji = "📈" if corr_val > 0 else "📉"
                print(f"  {emoji} {var1} ↔ {var2} : r = {corr_val:.3f} ({direction})")
    print("-"*80 + "\n")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f', vmin=-1, vmax=1)
    plt.title('Matrice de Corrélation', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('eda_06_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Graphique sauvegardé : eda_06_correlation_matrix.png\n")


def run_complete_eda(csv_path):
    """Exécute l'analyse exploratoire complète"""
    print("\n" + "🔍 ANALYSE EXPLORATOIRE COMPLÈTE (EDA)")
    print("="*80 + "\n")
    
    # 1. Charger et explorer
    df = load_and_explore_data(csv_path)
    
    # 2. Valeurs manquantes
    missing_df = analyze_missing_values(df)
    
    # 3. Distributions
    analyze_distributions(df)
    
    # 4. Variables catégorielles
    analyze_categorical(df)
    
    # 5. Outliers
    detect_outliers(df)
    
    # 6. Corrélations
    analyze_correlations(df)
    
    print("="*80)
    print("✅ ANALYSE EXPLORATOIRE TERMINÉE")
    print("="*80)
    print("\n📊 Graphiques générés :")
    print("  • eda_01_missing_values_analysis.png")
    print("  • eda_02_missing_values_heatmap.png")
    print("  • eda_03_distributions_analysis.png")
    print("  • eda_04_categorical_analysis.png")
    print("  • eda_05_outliers_detection.png")
    print("  • eda_06_correlation_matrix.png\n")
    
    return df


# =============================================================================
# PARTIE 2 : NETTOYAGE DES DONNÉES (à importer de votre module)
# =============================================================================

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class IVFMedicalDataCleaner:
    """Nettoyage avec AFC préservée"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df_original = None
        self.df_cleaned = None
        self.cleaning_report = {}
        self.iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    
    def load_data(self):
        self.df_original = pd.read_csv(self.csv_path)
        self.df_cleaned = self.df_original.copy()
        return self.df_original
    
    def clean_pipeline(self):
        """Pipeline de nettoyage simplifié pour démonstration"""
        print("\n" + "🧹 NETTOYAGE DES DONNÉES")
        print("="*80 + "\n")
        
        df = self.df_cleaned
        
        # 1. Imputation Age
        if 'Age' in df.columns and df['Age'].isnull().sum() > 0:
            median_age = df['Age'].median()
            df['Age'].fillna(median_age, inplace=True)
            print(f"✓ Age imputé : médiane = {median_age:.1f}")
        
        # 2. Extraction numériques
        if 'AMH' in df.columns and df['AMH'].dtype == 'object':
            df['AMH_numeric'] = df['AMH'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
            print(f"✓ AMH_numeric extrait")
        
        if 'E2_day5' in df.columns and df['E2_day5'].dtype == 'object':
            df['E2_day5_numeric'] = df['E2_day5'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
            print(f"✓ E2_day5_numeric extrait")
        
        # 3. Imputation simples (médiane)
        for var in ['AMH_numeric', 'n_Follicles', 'E2_day5_numeric']:
            if var in df.columns and df[var].isnull().sum() > 0:
                median_val = df[var].median()
                df[var].fillna(median_val, inplace=True)
                print(f"✓ {var} imputé : médiane = {median_val:.2f}")
        
        # 4. Imputation AFC intelligente
        if 'AFC' in df.columns and df['AFC'].isnull().sum() > 0:
            n_missing = df['AFC'].isnull().sum()
            
            imputation_features = []
            if 'AMH_numeric' in df.columns:
                imputation_features.append('AMH_numeric')
            if 'Age' in df.columns:
                imputation_features.append('Age')
            if 'n_Follicles' in df.columns:
                imputation_features.append('n_Follicles')
            imputation_features.append('AFC')
            
            imputed_values = self.iterative_imputer.fit_transform(df[imputation_features])
            afc_idx = imputation_features.index('AFC')
            df['AFC'] = imputed_values[:, afc_idx]
            
            print(f"✓ AFC imputé intelligemment : {n_missing} valeurs (r=0.77 avec AMH)")
        
        # 5. Normalisation RobustScaler
        for var in ['Age', 'AMH_numeric', 'n_Follicles', 'E2_day5_numeric', 'AFC']:
            if var in df.columns:
                median_val = df[var].median()
                Q1 = df[var].quantile(0.25)
                Q3 = df[var].quantile(0.75)
                IQR = Q3 - Q1
                if IQR != 0:
                    df[f'{var}_robust'] = (df[var] - median_val) / IQR
        
        print(f"✓ Normalisation RobustScaler appliquée")
        
        if 'Protocol' in df.columns:
    # Nettoyage : minuscules + suppression espaces superflus
                df['Protocol_clean'] = df['Protocol'].astype(str).str.strip().str.lower()

                # Extraire tous les types uniques
                unique_protocols = df['Protocol_clean'].unique()
                print("🔹 Protocoles uniques trouvés :", unique_protocols)

                # Créer des listes pour les 3 catégories
                agonist_list = [p for p in unique_protocols if 'agonist' in p and 'flex' not in p and 'fix' not in p]
                flexible_antagonist_list = [p for p in unique_protocols if 'flex' in p]
                fixed_antagonist_list = [p for p in unique_protocols if 'fix' in p]

                print("Agonist :", agonist_list)
                print("Flexible Antagonist :", flexible_antagonist_list)
                print("Fixed Antagonist :", fixed_antagonist_list)

                # Création d'un dictionnaire de mapping automatique
                protocol_mapping = {}
                for p in agonist_list:
                    protocol_mapping[p] = 0
                for p in flexible_antagonist_list:
                    protocol_mapping[p] = 1
                for p in fixed_antagonist_list:
                    protocol_mapping[p] = 2

                # Encodage
                df['Protocol_encoded'] = df['Protocol_clean'].map(protocol_mapping)

                # Pour les valeurs inconnues, assigner -1
                df['Protocol_encoded'].fillna(-1, inplace=True)

                print(f"✓ Protocol encodé : valeurs uniques après encodage -> {df['Protocol_encoded'].unique()}")

            
        # 7. Doublons
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates(keep='first')
            print(f"✓ {duplicates} doublons supprimés")
        
        self.df_cleaned = df
        self.cleaning_report['rows_after'] = len(df)
        
        print("\n✅ Nettoyage terminé")
        print(f"  • Patients : {len(df)}")
        print(f"  • Colonnes : {len(df.columns)}\n")
        
        return df
    
    def save_cleaned_data(self, output_path):
        self.df_cleaned.to_csv(output_path, index=False)
        print(f"💾 Données nettoyées sauvegardées : {output_path}\n")


# =============================================================================
# PARTIE 3 : VISUALISATION AVANT/APRÈS
# =============================================================================

class CleaningVisualizer:
    """Visualisation avant/après nettoyage"""
    
    def __init__(self, df_before, df_after, output_dir='cleaning_viz'):
        self.df_before = df_before
        self.df_after = df_after
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def viz_missing_comparison(self):
        """Comparaison valeurs manquantes"""
        print("📊 Génération visualisation : Valeurs manquantes avant/après")
        
        missing_before = self.df_before.isnull().sum()
        missing_before = missing_before[missing_before > 0]
        
        common_cols = [col for col in missing_before.index if col in self.df_after.columns]
        missing_after = self.df_after[common_cols].isnull().sum()
        
        comparison = pd.DataFrame({
            'Avant': missing_before[common_cols],
            'Après': missing_after
        })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(common_cols))
        width = 0.35
        
        ax1.bar(x - width/2, comparison['Avant'], width, label='Avant', color='coral', alpha=0.8)
        ax1.bar(x + width/2, comparison['Après'], width, label='Après', color='skyblue', alpha=0.8)
        ax1.set_xlabel('Variables')
        ax1.set_ylabel('Valeurs manquantes')
        ax1.set_title('Comparaison valeurs manquantes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(common_cols, rotation=45, ha='right')
        ax1.legend()
        
        reduction = ((comparison['Avant'] - comparison['Après']) / comparison['Avant'] * 100)
        colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in reduction]
        
        ax2.barh(common_cols, reduction, color=colors, alpha=0.8)
        ax2.set_xlabel('Réduction (%)')
        ax2.set_title('Amélioration par variable')
        ax2.axvline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_01_missing_comparison.png', dpi=300)
        plt.show()
        print(f"  ✓ viz_01_missing_comparison.png\n")
    
    def viz_afc_quality(self):
        """Qualité imputation AFC"""
        if 'AFC' not in self.df_before.columns or 'AFC' not in self.df_after.columns:
            return
        
        print("📊 Génération visualisation : Qualité imputation AFC")
        
        afc_before = self.df_before['AFC'].copy()
        afc_after = self.df_after['AFC'].copy()
        was_missing = afc_before.isnull()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Distribution
        afc_original = afc_before.dropna()
        afc_complete = afc_after.dropna()
        
        axes[0].hist(afc_original, bins=20, alpha=0.6, label='Original', 
                    color='coral', edgecolor='black')
        axes[0].hist(afc_complete, bins=20, alpha=0.6, label='Après imputation', 
                    color='skyblue', edgecolor='black')
        axes[0].set_xlabel('AFC')
        axes[0].set_ylabel('Fréquence')
        axes[0].set_title('Distribution AFC : Avant vs Après')
        axes[0].legend()
        
        # AFC vs AMH
        if 'AMH_numeric' in self.df_after.columns:
            mask_original = ~was_missing
            axes[1].scatter(self.df_after.loc[mask_original, 'AMH_numeric'], 
                          self.df_after.loc[mask_original, 'AFC'],
                          alpha=0.5, s=50, label='Valeurs originales', color='coral')
            
            mask_imputed = was_missing & afc_after.notna()
            axes[1].scatter(self.df_after.loc[mask_imputed, 'AMH_numeric'], 
                          self.df_after.loc[mask_imputed, 'AFC'],
                          alpha=0.7, s=80, marker='s', label='Valeurs imputées', 
                          color='skyblue', edgecolor='black')
            
            axes[1].set_xlabel('AMH')
            axes[1].set_ylabel('AFC')
            axes[1].set_title('AFC imputé vs AMH (r=0.77)')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_02_afc_quality.png', dpi=300)
        plt.show()
        print(f"  ✓ viz_02_afc_quality.png\n")
    
    def generate_all(self):
        print("\n🎨 Génération des visualisations avant/après\n")
        self.viz_missing_comparison()
        self.viz_afc_quality()
        print("✅ Visualisations terminées\n")


# =============================================================================
# PIPELINE COMPLET
# =============================================================================

def run_complete_pipeline(input_csv, output_csv):
    """
    Pipeline complet : EDA → Nettoyage → Visualisation
    """
    
    print("\n" + "="*80)
    print("🚀 PIPELINE COMPLET : EDA + NETTOYAGE + VISUALISATION")
    print("="*80 + "\n")
    
    # ÉTAPE 1 : EDA avant nettoyage
    print("ÉTAPE 1 : ANALYSE EXPLORATOIRE (AVANT NETTOYAGE)")
    print("="*80)
    df_original = run_complete_eda(input_csv)
    
    # ÉTAPE 2 : Nettoyage
    print("\n" + "="*80)
    print("ÉTAPE 2 : NETTOYAGE DES DONNÉES")
    print("="*80)
    cleaner = IVFMedicalDataCleaner(input_csv)
    cleaner.load_data()
    df_cleaned = cleaner.clean_pipeline()
    cleaner.save_cleaned_data(output_csv)
    
    # ÉTAPE 3 : Visualisation avant/après
    print("="*80)
    print("ÉTAPE 3 : VISUALISATION AVANT/APRÈS")
    print("="*80)
    visualizer = CleaningVisualizer(df_original, df_cleaned)
    visualizer.generate_all()
    
    print("="*80)
    print("✅ PIPELINE COMPLET TERMINÉ")
    print("="*80)
    print("\n📊 Fichiers générés :")
    print("  EDA (6 graphiques) :")
    print("    • eda_01_missing_values_analysis.png")
    print("    • eda_02_missing_values_heatmap.png")
    print("    • eda_03_distributions_analysis.png")
    print("    • eda_04_categorical_analysis.png")
    print("    • eda_05_outliers_detection.png")
    print("    • eda_06_correlation_matrix.png")
    print("\n  Nettoyage :")
    print(f"    • {output_csv}")
    print("\n  Visualisation avant/après (2 graphiques) :")
    print("    • viz_01_missing_comparison.png")
    print("    • viz_02_afc_quality.png")
    print("\n" + "="*80 + "\n")


# =============================================================================
# UTILISATION
# =============================================================================

if __name__ == "__main__":
    
    # Chemins des fichiers
    input_csv = "C:\\Users\\yesmine\\Desktop\\Tanit\\data\\raw\\patients.csv"
    output_csv = "C:\\Users\\yesmine\\Desktop\\Tanit\\data\\processed\\patients_medical_clean.csv"
    
    # Exécuter le pipeline complet
    run_complete_pipeline(input_csv, output_csv)