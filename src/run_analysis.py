from preprocessing.feature_engineering import IVFExploratoryAnalysis


if __name__ == "__main__":
    # Remplacer par le chemin de votre fichier CSV
    analyzer = IVFExploratoryAnalysis('C:\\Users\\yesmine\\Desktop\\Tanit\\data\\processed\\patients.csv')
    
    # Exécuter l'analyse
    analyzer.load_data()
    analyzer.pattern_discovery()
    analyzer.feature_correlations()
    analyzer.medical_insights()
    analyzer.generate_summary()
