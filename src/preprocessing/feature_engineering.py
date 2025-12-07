import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("🔍 ANALYSE EXPLORATOIRE DES DONNÉES IVF (EDA)")
print("="*80 + "\n")

# =============================================================================
# CLASSE D'ANALYSE EXPLORATOIRE
# =============================================================================

class IVFExploratoryAnalysis:
    """Classe pour l'analyse exploratoire des données IVF"""
    
    def __init__(self, csv_path):
        """Initialise l'analyseur"""
        self.csv_path = csv_path
        self.df = None
        self.insights = []
        
    def load_data(self):
        """Charge les données"""
        print("="*80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("="*80 + "\n")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Extraire les valeurs numériques si nécessaire
        if 'AMH' in self.df.columns and self.df['AMH'].dtype == 'object':
            self.df['AMH_numeric'] = self.df['AMH'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
        
        if 'E2_day5' in self.df.columns and self.df['E2_day5'].dtype == 'object':
            self.df['E2_day5_numeric'] = self.df['E2_day5'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
        
        print(f"✓ Données chargées : {self.csv_path}")
        print(f"  • Nombre de patients : {len(self.df)}")
        print(f"  • Nombre de variables : {len(self.df.columns)}")
        print()
        
        return self.df
    
    def pattern_discovery(self):
        """1. DÉCOUVERTE DE PATTERNS"""
        print("="*80)
        print("1️⃣ DÉCOUVERTE DE PATTERNS")
        print("="*80 + "\n")
        
        df = self.df
        
        # PATTERN 1 : Distribution des réponses patients
        print("📊 PATTERN 1 : Distribution des réponses patients")
        print("-"*80)
        
        if 'Patient_Response' in df.columns:
            response_dist = df['Patient_Response'].value_counts()
            response_pct = df['Patient_Response'].value_counts(normalize=True) * 100
            
            print("Distribution :")
            for resp in ['low', 'optimal', 'high']:
                if resp in response_dist.index:
                    count = response_dist[resp]
                    pct = response_pct[resp]
                    print(f"  • {resp:10s} : {count:3d} patients ({pct:5.1f}%)")
            
            most_common = response_dist.idxmax()
            most_common_pct = response_pct[most_common]
            self.insights.append(f"La réponse la plus fréquente est '{most_common}' ({most_common_pct:.1f}%)")
            
            # Visualisation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            response_dist.plot(kind='bar', ax=ax1, color=['#ff7675', '#74b9ff', '#55efc4'])
            ax1.set_title('Distribution des réponses patients', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Type de réponse')
            ax1.set_ylabel('Nombre de patients')
            ax1.tick_params(axis='x', rotation=0)
            
            colors = ['#ff7675', '#74b9ff', '#55efc4']
            response_dist.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('Proportion des réponses patients', fontsize=14, fontweight='bold')
            ax2.set_ylabel('')
            
            plt.tight_layout()
            plt.savefig('eda_01_patient_response_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("\n💾 Graphique sauvegardé : eda_01_patient_response_distribution.png")
        
        print("-"*80 + "\n")
        
        # PATTERN 2 : Distribution des protocoles
        print("📊 PATTERN 2 : Distribution des protocoles de stimulation")
        print("-"*80)
        
        if 'Protocol' in df.columns:
            protocol_dist = df['Protocol'].value_counts()
            protocol_pct = df['Protocol'].value_counts(normalize=True) * 100
            
            print("Distribution :")
            for protocol, count in protocol_dist.items():
                pct = protocol_pct[protocol]
                print(f"  • {protocol:25s} : {count:3d} patients ({pct:5.1f}%)")
            
            most_used = protocol_dist.idxmax()
            self.insights.append(f"Le protocole le plus utilisé est '{most_used}' ({protocol_pct[most_used]:.1f}%)")
            
            plt.figure(figsize=(12, 6))
            protocol_dist.plot(kind='barh', color='#6c5ce7')
            plt.title('Distribution des protocoles de stimulation', fontsize=14, fontweight='bold')
            plt.xlabel('Nombre de patients')
            plt.ylabel('Protocole')
            plt.tight_layout()
            plt.savefig('eda_02_protocol_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("\n💾 Graphique sauvegardé : eda_02_protocol_distribution.png")
        
        print("-"*80 + "\n")
        
        # PATTERN 3 : Distribution de l'âge
        print("📊 PATTERN 3 : Distribution de l'âge des patientes")
        print("-"*80)
        
        if 'Age' in df.columns:
            age_stats = df['Age'].describe()
            
            print("Statistiques :")
            print(f"  • Moyenne : {age_stats['mean']:.1f} ans")
            print(f"  • Médiane : {age_stats['50%']:.1f} ans")
            print(f"  • Écart-type : {age_stats['std']:.1f} ans")
            print(f"  • Min - Max : {age_stats['min']:.0f} - {age_stats['max']:.0f} ans")
            
            df['Age_Category'] = pd.cut(df['Age'], 
                                        bins=[0, 30, 35, 40, 100], 
                                        labels=['<30', '30-35', '35-40', '>40'])
            age_cat_dist = df['Age_Category'].value_counts().sort_index()
            
            print("\nCatégories d'âge :")
            for cat, count in age_cat_dist.items():
                pct = (count / len(df)) * 100
                print(f"  • {cat} ans : {count:3d} patients ({pct:5.1f}%)")
            
            if age_stats['mean'] < 35:
                self.insights.append(f"Population relativement jeune (âge moyen : {age_stats['mean']:.1f} ans)")
            
            # Visualisation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.hist(df['Age'].dropna(), bins=15, color='#fd79a8', edgecolor='black', alpha=0.7)
            ax1.axvline(age_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Moyenne: {age_stats['mean']:.1f}")
            ax1.set_title('Distribution de l\'âge', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Âge (ans)')
            ax1.set_ylabel('Fréquence')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            age_cat_dist.plot(kind='bar', ax=ax2, color='#e17055')
            ax2.set_title('Distribution par catégories d\'âge', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Catégorie d\'âge')
            ax2.set_ylabel('Nombre de patients')
            ax2.tick_params(axis='x', rotation=0)
            
            plt.tight_layout()
            plt.savefig('eda_03_age_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("\n💾 Graphique sauvegardé : eda_03_age_distribution.png")
        
        print("-"*80 + "\n")
        
        # PATTERN 4 : Biomarqueurs (AMH, AFC)
        print("📊 PATTERN 4 : Distribution des biomarqueurs")
        print("-"*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if 'AMH_numeric' in df.columns:
            amh_stats = df['AMH_numeric'].describe()
            print("AMH (Anti-Müllerian Hormone) :")
            print(f"  • Moyenne : {amh_stats['mean']:.2f} ng/mL")
            print(f"  • Médiane : {amh_stats['50%']:.2f} ng/mL")
            print(f"  • Min - Max : {amh_stats['min']:.2f} - {amh_stats['max']:.2f} ng/mL")
            
            axes[0, 0].hist(df['AMH_numeric'].dropna(), bins=15, color='#00b894', edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(amh_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Moyenne: {amh_stats['mean']:.2f}")
            axes[0, 0].set_title('Distribution AMH', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('AMH (ng/mL)')
            axes[0, 0].set_ylabel('Fréquence')
            axes[0, 0].legend()
            
            axes[0, 1].boxplot(df['AMH_numeric'].dropna(), vert=True)
            axes[0, 1].set_title('Boxplot AMH', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('AMH (ng/mL)')
            
            if amh_stats['mean'] < 1.0:
                self.insights.append("AMH moyen faible (<1 ng/mL) - Réserve ovarienne réduite")
            elif amh_stats['mean'] > 3.5:
                self.insights.append("AMH moyen élevé (>3.5 ng/mL) - Bonne réserve ovarienne")
        
        if 'AFC' in df.columns:
            afc_stats = df['AFC'].describe()
            print("\nAFC (Antral Follicle Count) :")
            print(f"  • Moyenne : {afc_stats['mean']:.1f} follicules")
            print(f"  • Médiane : {afc_stats['50%']:.1f} follicules")
            print(f"  • Min - Max : {afc_stats['min']:.0f} - {afc_stats['max']:.0f} follicules")
            
            axes[1, 0].hist(df['AFC'].dropna(), bins=15, color='#fdcb6e', edgecolor='black', alpha=0.7)
            axes[1, 0].axvline(afc_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Moyenne: {afc_stats['mean']:.1f}")
            axes[1, 0].set_title('Distribution AFC', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('AFC (nombre de follicules)')
            axes[1, 0].set_ylabel('Fréquence')
            axes[1, 0].legend()
            
            axes[1, 1].boxplot(df['AFC'].dropna(), vert=True)
            axes[1, 1].set_title('Boxplot AFC', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('AFC (nombre de follicules)')
        
        plt.tight_layout()
        plt.savefig('eda_04_biomarkers_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n💾 Graphique sauvegardé : eda_04_biomarkers_distribution.png")
        print("-"*80 + "\n")
    
    def feature_correlations(self):
        """2. CORRÉLATIONS ENTRE VARIABLES"""
        print("="*80)
        print("2️⃣ CORRÉLATIONS ENTRE VARIABLES")
        print("="*80 + "\n")
        
        df = self.df
        
        numeric_cols = ['Age', 'Cycle_number', 'AMH_numeric', 'N_Follicles', 'E2_day5_numeric', 'AFC']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        if len(numeric_cols) < 2:
            print("⚠️ Pas assez de variables numériques\n")
            return
        
        corr_matrix = df[numeric_cols].corr()
        
        print("📊 MATRICE DE CORRÉLATION")
        print("-"*80)
        print(corr_matrix.round(3))
        print("-"*80 + "\n")
        
        print("🔍 CORRÉLATIONS SIGNIFICATIVES (|r| > 0.5)")
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
                    
                    if abs(corr_val) > 0.7:
                        self.insights.append(f"Corrélation forte entre {var1} et {var2} (r={corr_val:.2f})")
        
        print("-"*80 + "\n")
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, fmt='.2f', vmin=-1, vmax=1)
        plt.title('Matrice de Corrélation', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('eda_05_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Graphique sauvegardé : eda_05_correlation_matrix.png\n")
    
    def medical_insights(self):
        """3. INSIGHTS MÉDICAUX"""
        print("="*80)
        print("3️⃣ INSIGHTS MÉDICAUX")
        print("="*80 + "\n")
        
        df = self.df
        
        # Impact de l'âge sur la réponse
        print("🏥 INSIGHT 1 : Impact de l'âge sur la réponse ovarienne")
        print("-"*80)
        
        if 'Age' in df.columns and 'Patient_Response' in df.columns:
            age_by_response = df.groupby('Patient_Response')['Age'].agg(['mean', 'std', 'count'])
            
            print("Âge moyen par type de réponse :")
            for response in ['low', 'optimal', 'high']:
                if response in age_by_response.index:
                    mean_age = age_by_response.loc[response, 'mean']
                    std_age = age_by_response.loc[response, 'std']
                    count = age_by_response.loc[response, 'count']
                    print(f"  • {response:10s} : {mean_age:.1f} ± {std_age:.1f} ans (n={count:.0f})")
            
            # Test ANOVA
            groups = [df[df['Patient_Response'] == resp]['Age'].dropna() 
                     for resp in ['low', 'optimal', 'high'] 
                     if resp in df['Patient_Response'].unique()]
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"\nTest ANOVA : F={f_stat:.3f}, p={p_value:.4f}")
                if p_value < 0.05:
                    print("  ⚠️ Différence significative (p < 0.05)")
                    self.insights.append(f"L'âge influence la réponse ovarienne (p={p_value:.4f})")
            
            # Visualisation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            df.boxplot(column='Age', by='Patient_Response', ax=ax1)
            ax1.set_title('Âge selon la réponse', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Type de réponse')
            ax1.set_ylabel('Âge (ans)')
            plt.suptitle('')
            
            sns.violinplot(data=df, x='Patient_Response', y='Age', ax=ax2,
                          palette={'low': '#ff7675', 'optimal': '#74b9ff', 'high': '#55efc4'})
            ax2.set_title('Distribution de l\'âge par réponse', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('eda_07_age_vs_response.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("\n💾 Graphique sauvegardé : eda_07_age_vs_response.png")
        
        print("-"*80 + "\n")
        
        # AMH vs Follicules
        print("🏥 INSIGHT 2 : Relation AMH et nombre de follicules")
        print("-"*80)
        
        if 'AMH_numeric' in df.columns and 'N_Follicles' in df.columns:
            valid_data = df[['AMH_numeric', 'N_Follicles']].dropna()
            if len(valid_data) > 2:
                corr, p_value = stats.pearsonr(valid_data['AMH_numeric'], valid_data['N_Follicles'])
                
                print(f"Corrélation AMH ↔ Nombre de follicules :")
                print(f"  • Coefficient : r = {corr:.3f}")
                print(f"  • P-value : p = {p_value:.4f}")
                
                if p_value < 0.05 and corr > 0.5:
                    print("  ✅ Corrélation positive significative")
                    self.insights.append(f"AMH et follicules fortement corrélés (r={corr:.2f})")
                
                # Visualisation
                plt.figure(figsize=(10, 6))
                plt.scatter(df['AMH_numeric'], df['N_Follicles'], alpha=0.6, s=100, c='#0984e3')
                
                z = np.polyfit(valid_data['AMH_numeric'], valid_data['N_Follicles'], 1)
                p = np.poly1d(z)
                plt.plot(valid_data['AMH_numeric'].sort_values(), 
                        p(valid_data['AMH_numeric'].sort_values()), 
                        "r--", linewidth=2, label=f'Régression (r={corr:.2f})')
                
                plt.xlabel('AMH (ng/mL)', fontsize=12)
                plt.ylabel('Nombre de follicules', fontsize=12)
                plt.title('Relation AMH vs Follicules', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig('eda_08_amh_vs_follicles.png', dpi=300, bbox_inches='tight')
                plt.show()
                print("\n💾 Graphique sauvegardé : eda_08_amh_vs_follicles.png")
        
        print("-"*80 + "\n")
        
        # Efficacité des protocoles
        print("🏥 INSIGHT 3 : Efficacité des protocoles")
        print("-"*80)
        
        if 'Protocol' in df.columns and 'Patient_Response' in df.columns:
            protocol_response = pd.crosstab(df['Protocol'], df['Patient_Response'], normalize='index') * 100
            
            print("Répartition des réponses par protocole (%) :")
            print(protocol_response.round(1))
            
            if 'optimal' in protocol_response.columns:
                best_protocol = protocol_response['optimal'].idxmax()
                best_pct = protocol_response.loc[best_protocol, 'optimal']
                print(f"\n  ⭐ Meilleur protocole : {best_protocol} ({best_pct:.1f}% optimal)")
                self.insights.append(f"'{best_protocol}' donne le plus de réponses optimales ({best_pct:.1f}%)")
            
            protocol_response.plot(kind='bar', figsize=(12, 6),
                                  color=['#ff7675', '#74b9ff', '#55efc4'])
            plt.title('Distribution par protocole', fontsize=14, fontweight='bold')
            plt.xlabel('Protocole')
            plt.ylabel('Pourcentage (%)')
            plt.legend(title='Réponse', bbox_to_anchor=(1.05, 1))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('eda_09_protocol_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("\n💾 Graphique sauvegardé : eda_09_protocol_effectiveness.png")
        
        print("-"*80 + "\n")
    
    def generate_summary(self):
        """Génère un résumé des insights"""
        print("="*80)
        print("📋 RÉSUMÉ DES INSIGHTS")
        print("="*80 + "\n")
        
        if self.insights:
            print("🔍 Principaux insights :\n")
            for i, insight in enumerate(self.insights, 1):
                print(f"{i}. {insight}")
            print()
        else:
            print("⚠️ Aucun insight significatif identifié.\n")
        
        print("="*80)
        print("✅ ANALYSE TERMINÉE")
        print("="*80)

""" # =============================================================================
# UTILISATION
# =============================================================================

if __name__ == "__main__":
    # Remplacer par le chemin de votre fichier CSV
    analyzer = IVFExploratoryAnalysis('your_data.csv')
    
    # Exécuter l'analyse
    analyzer.load_data()
    analyzer.pattern_discovery()
    analyzer.feature_correlations()
    analyzer.medical_insights()
    analyzer.generate_summary() """