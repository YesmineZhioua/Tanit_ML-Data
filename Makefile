# Makefile pour le projet IVF Patient Response Prediction
# Placez ce fichier à la racine du projet : C:\Users\yesmine\Desktop\Tanit\Makefile

.PHONY: help install clean dataset train evaluate predict api streamlit all pipeline fix-and-restart stop-api stop-streamlit stop-all

# Variables
PYTHON := python
PIP := pip
DATA_PATH := C:\\Users\\yesmine\\Desktop\\Tanit\\data\\processed\\patients_medical_clean.csv
MODEL_DIR := saved_models
SRC_MODEL := src/model
SRC_API := src/api

# Couleurs pour l'affichage (Windows compatible)
ECHO := @echo

help:
	$(ECHO) "================================================================================"
	$(ECHO) "               IVF PATIENT RESPONSE PREDICTION - MAKEFILE"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	$(ECHO) "Commandes disponibles:"
	$(ECHO) ""
	$(ECHO) "  SETUP & INSTALLATION:"
	$(ECHO) "    make install          - Installe toutes les dependances"
	$(ECHO) "    make clean            - Nettoie les fichiers temporaires"
	$(ECHO) ""
	$(ECHO) "  PIPELINE COMPLET:"
	$(ECHO) "    make all              - Execute tout le pipeline (dataset + train + evaluate)"
	$(ECHO) "    make pipeline         - Alias pour 'make all'"
	$(ECHO) "    make fix-and-restart  - Repare et redemarre tout (clean + all + api + streamlit)"
	$(ECHO) ""
	$(ECHO) "  ETAPES INDIVIDUELLES:"
	$(ECHO) "    make dataset          - Prepare et sauvegarde les datasets"
	$(ECHO) "    make preprocessors    - Cree scaler et label encoder"
	$(ECHO) "    make train            - Entraine tous les modeles"
	$(ECHO) "    make evaluate         - Evalue les modeles et selectionne le meilleur"
	$(ECHO) "    make predict          - Test de prediction sur un patient exemple"
	$(ECHO) ""
	$(ECHO) "  SERVICES:"
	$(ECHO) "    make api              - Demarre l'API Flask (port 5000)"
	$(ECHO) "    make streamlit        - Demarre l'interface Streamlit (port 8501)"
	$(ECHO) "    make services         - Demarre API + Streamlit ensemble"
	$(ECHO) ""
	$(ECHO) "  ARRET:"
	$(ECHO) "    make stop-api         - Arrete l'API Flask"
	$(ECHO) "    make stop-streamlit   - Arrete Streamlit"
	$(ECHO) "    make stop-all         - Arrete tous les services"
	$(ECHO) ""
	$(ECHO) "  UTILITAIRES:"
	$(ECHO) "    make test-api         - Test la connexion API"
	$(ECHO) "    make logs             - Affiche les logs"
	$(ECHO) ""
	$(ECHO) "================================================================================"

# ============================================================================
# INSTALLATION
# ============================================================================

install:
	$(ECHO) "Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install pandas numpy scikit-learn joblib flask flask-cors streamlit plotly requests matplotlib seaborn
	$(ECHO) "Installation complete!"

# ============================================================================
# NETTOYAGE
# ============================================================================

clean:
	$(ECHO) "Cleaning temporary files..."
	-@if exist __pycache__ rmdir /s /q __pycache__
	-@if exist src\__pycache__ rmdir /s /q src\__pycache__
	-@if exist src\model\__pycache__ rmdir /s /q src\model\__pycache__
	-@if exist src\api\__pycache__ rmdir /s /q src\api\__pycache__
	-@del /q *.pyc 2>nul
	-@del /q src\*.pyc 2>nul
	-@del /q src\model\*.pyc 2>nul
	-@del /q src\api\*.pyc 2>nul
	$(ECHO) "Cleaning complete!"

# ============================================================================
# PREPARATION DES DONNEES
# ============================================================================

dataset:
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  STEP 1: DATASET PREPARATION"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	cd $(SRC_MODEL) && $(PYTHON) dataset.py
	$(ECHO) ""
	$(ECHO) "Dataset preparation complete!"
	$(ECHO) ""

# ============================================================================
# CREATION DES PREPROCESSEURS
# ============================================================================

preprocessors:
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  STEP 1.5: CREATING PREPROCESSORS (SCALER & LABEL ENCODER)"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	cd $(SRC_MODEL) && $(PYTHON) create_scaler.py
	$(ECHO) "Scaler created successfully!"
	cd $(SRC_MODEL) && $(PYTHON) create_label_encoder.py
	$(ECHO) "Label encoder created successfully!"
	$(ECHO) ""
	$(ECHO) "Preprocessors creation complete!"
	$(ECHO) ""

# ============================================================================
# ENTRAINEMENT
# ============================================================================

train: dataset preprocessors
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  STEP 2: MODEL TRAINING"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	cd $(SRC_MODEL) && $(PYTHON) train.py
	$(ECHO) ""
	$(ECHO) "Training complete!"
	$(ECHO) ""

# ============================================================================
# EVALUATION
# ============================================================================

evaluate: train
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  STEP 3: MODEL EVALUATION"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	cd $(SRC_MODEL) && $(PYTHON) evaluate.py
	$(ECHO) ""
	$(ECHO) "Evaluation complete!"
	$(ECHO) ""

# ============================================================================
# PREDICTION TEST
# ============================================================================

predict:
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  TESTING PREDICTION"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	cd $(SRC_MODEL) && $(PYTHON) predict.py
	$(ECHO) ""

# ============================================================================
# SERVICES
# ============================================================================

api:
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  STARTING FLASK API"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	$(ECHO) "API will be available at: http://localhost:5000"
	$(ECHO) "Press Ctrl+C to stop"
	$(ECHO) ""
	cd $(SRC_API) && $(PYTHON) app.py

streamlit:
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  STARTING STREAMLIT INTERFACE"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	$(ECHO) "Streamlit will be available at: http://localhost:8501"
	$(ECHO) "Press Ctrl+C to stop"
	$(ECHO) ""
	cd $(SRC_API) && streamlit run streamlit_app.py

services:
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  STARTING ALL SERVICES"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	$(ECHO) "Starting API in background..."
	start /B cmd /c "cd $(SRC_API) && $(PYTHON) app.py"
	timeout /t 5 /nobreak > nul
	$(ECHO) "Starting Streamlit..."
	cd $(SRC_API) && streamlit run streamlit_app.py

# ============================================================================
# ARRET DES SERVICES
# ============================================================================

stop-api:
	$(ECHO) "Stopping Flask API..."
	-@taskkill /F /IM python.exe /FI "WINDOWTITLE eq *app.py*" 2>nul || echo No API process found
	$(ECHO) "API stopped!"

stop-streamlit:
	$(ECHO) "Stopping Streamlit..."
	-@taskkill /F /IM streamlit.exe 2>nul || echo No Streamlit process found
	-@taskkill /F /IM python.exe /FI "WINDOWTITLE eq *streamlit*" 2>nul || echo No Streamlit process found
	$(ECHO) "Streamlit stopped!"

stop-all: stop-api stop-streamlit
	$(ECHO) "All services stopped!"

# ============================================================================
# PIPELINE COMPLET
# ============================================================================

all: dataset preprocessors train evaluate
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  PIPELINE COMPLETE!"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	$(ECHO) "All steps executed successfully:"
	$(ECHO) "  1. Dataset prepared and saved"
	$(ECHO) "  2. Preprocessors (scaler & label encoder) created"
	$(ECHO) "  3. Models trained with hyperparameter tuning"
	$(ECHO) "  4. Models evaluated and best model selected"
	$(ECHO) ""
	$(ECHO) "Next steps:"
	$(ECHO) "  - Run 'make predict' to test predictions"
	$(ECHO) "  - Run 'make api' to start the API server"
	$(ECHO) "  - Run 'make streamlit' to start the web interface"
	$(ECHO) "  - Run 'make services' to start both together"
	$(ECHO) ""
	$(ECHO) "================================================================================"

pipeline: all

# ============================================================================
# FIX AND RESTART (Remplace fix_and_restart.bat)
# ============================================================================

fix-and-restart: clean all
	$(ECHO) ""
	$(ECHO) "================================================================================"
	$(ECHO) "  FIX AND RESTART COMPLETE!"
	$(ECHO) "================================================================================"
	$(ECHO) ""
	$(ECHO) "System cleaned and rebuilt successfully!"
	$(ECHO) ""
	$(ECHO) "Starting services..."
	$(ECHO) ""
	$(MAKE) services

# ============================================================================
# TESTS
# ============================================================================

test-api:
	$(ECHO) "Testing API connection..."
	curl http://localhost:5000/api/health || echo API not responding

logs:
	$(ECHO) "Displaying recent logs..."
	@if exist logs (type logs\*.log) else echo No logs found

# ============================================================================
# WORKFLOWS SPECIFIQUES
# ============================================================================

quick-start: install all services
	$(ECHO) "Quick start complete!"

dev: clean dataset train predict
	$(ECHO) "Development environment ready!"

prod: clean all api
	$(ECHO) "Production environment ready!"