"""
IVF Patient Response Prediction - Streamlit Interface
Interface professionnelle et épurée avec palette personnalisée
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


# -------------------------------------------------------------------------
# 🔧 PAGE CONFIG
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="TanitAI HealthCare - IVF Predictor",
    page_icon="💙",
    layout="wide",
)


# -------------------------------------------------------------------------
# 🎨 GLOBAL CSS - PALETTE: #d094ea, #bccefb, noir
# -------------------------------------------------------------------------
st.markdown("""
<style>
    /* Reset & Base */
    .main {
        background-color: #ffffff;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, #bccefb 0%, #d094ea 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        color: #000000;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #333333;
        font-weight: 400;
    }

    /* Navigation */
    .stButton > button {
        background-color: #bccefb;
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(188,206,251,0.3);
    }
    
    .stButton > button:hover {
        background-color: #d094ea;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(208,148,234,0.4);
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #fdf8ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #d094ea;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .info-card h4 {
        color: #000000;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .info-card p, .info-card li {
        color: #333333;
        line-height: 1.6;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #bccefb 0%, #d094ea 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    div[data-testid="metric-container"] label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #000000;
        font-weight: 700;
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 2px solid #bccefb;
        border-radius: 8px;
        color: #000000;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #d094ea;
        box-shadow: 0 0 0 2px rgba(208,148,234,0.2);
    }
    
    /* Status Messages */
    .stSuccess {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        color: #000000;
    }
    
    .stError {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #000000;
    }
    
    .stWarning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        color: #000000;
    }
    
    .stInfo {
        background-color: #e3f2fd;
        border-left: 4px solid #bccefb;
        color: #000000;
    }
    
    /* Subheader */
    .stSubheader {
        color: #000000;
        font-weight: 600;
        padding-bottom: 1rem;
        border-bottom: 2px solid #d094ea;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# 🔐 LOGIN / LOGOUT SYSTEM
# -------------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.session_state.logged_in = True

def logout():
    st.session_state.logged_in = False


# -------------------------------------------------------------------------
# 🔷 HEADER SECTION
# -------------------------------------------------------------------------
col_header, col_login = st.columns([5, 1])

with col_header:
    st.markdown("""
    <div class="app-header">
        <div class="header-content">
            <div class="header-title">TanitAI HealthCare</div>
            <div class="header-subtitle">Prédiction Intelligente de Réponse au Traitement FIV</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_login:
    st.write("")
    st.write("")
    if not st.session_state.logged_in:
        if st.button("🔐 Connexion", key="login", type="primary"):
            login()
    else:
        if st.button("🔓 Déconnexion", key="logout"):
            logout()


# If logged out
if not st.session_state.logged_in:
    st.warning("🔒 Veuillez vous connecter pour accéder au système de prédiction.")
    st.info("Cliquez sur le bouton **Connexion** ci-dessus.")
    st.stop()


# -------------------------------------------------------------------------
# 🔵 NAVIGATION
# -------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    btn_single = st.button("👤 Prédiction Individuelle", use_container_width=True)
with col2:
    btn_batch = st.button("📊 Prédiction par Lot", use_container_width=True)
with col3:
    btn_info = st.button("ℹ️ Informations Modèle", use_container_width=True)

if "page" not in st.session_state:
    st.session_state.page = "Single"

if btn_single:
    st.session_state.page = "Single"
elif btn_batch:
    st.session_state.page = "Batch"
elif btn_info:
    st.session_state.page = "Info"

page = st.session_state.page


# -------------------------------------------------------------------------
# ℹ️ INFO CARD
# -------------------------------------------------------------------------
st.markdown("""
<div class="info-card">
    <h4>📊 À Propos du Système</h4>
    <p>Ce système utilise l'intelligence artificielle pour prédire la réponse des patientes au traitement de FIV.</p>
    <p><strong>Classes de Réponse :</strong></p>
    <ul>
        <li><b>Low</b> – Sous-réponse au traitement</li>
        <li><b>Optimal</b> – Réponse normale</li>
        <li><b>High</b> – Risque de syndrome d'hyperstimulation (OHSS)</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# API CONFIG
# -------------------------------------------------------------------------
API_URL = "http://localhost:5000"

def check_api_health():
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def predict_response(data):
    try:
        r = requests.post(f"{API_URL}/api/predict", json=data, timeout=30)
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


# -------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------------------------------------------------
def create_probability_chart(proba):
    fig = go.Figure(go.Bar(
        x=list(proba.keys()),
        y=list(proba.values()),
        text=[f"{v*100:.1f}%" for v in proba.values()],
        textposition="outside",
        marker_color=["#bccefb", "#d094ea", "#333333"]
    ))
    fig.update_layout(
        height=350,
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000", size=12)
    )
    return fig

def create_gauge_chart(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#d094ea"},
            "steps": [
                {"range": [0, 60], "color": "#ffcccc"},
                {"range": [60, 80], "color": "#fff2cc"},
                {"range": [80, 100], "color": "#ccffcc"},
            ],
        },
    ))
    fig.update_layout(
        height=300,
        font=dict(color="#000000")
    )
    return fig


# -------------------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------------------
def main():
    api_ok = check_api_health()
    if not api_ok:
        st.error("❌ API hors ligne - Veuillez démarrer le backend Flask.")
        st.stop()

    st.success("✅ API Connectée")

    # ---------------------------------------------------------
    # PAGE 1 — PRÉDICTION INDIVIDUELLE
    # ---------------------------------------------------------
    if page == "Single":
        st.subheader("🎯 Analyse Patiente Individuelle")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", 18, 50, 32)
            cycle = st.number_input("Numéro de Cycle", 1, 10, 1)
            protocol = st.selectbox("Protocole de Stimulation", ["agonist", "flex anta", "fix anta"])

        with col2:
            amh = st.number_input("AMH (ng/mL)", 0.0, 20.0, 2.5)
            afc = st.number_input("AFC", 0, 50, 15)
            follicles = st.number_input("Nombre de Follicules", 0, 50, 15)
            e2 = st.number_input("E2 Jour 5 (pg/mL)", 0.0, 5000.0, 300.0)

        center = st.columns([1, 2, 1])[1]
        with center:
            run = st.button("🚀 Lancer la Prédiction", use_container_width=True)

        if run:
            payload = {
                "Age": age,
                "Cycle_number": cycle,
                "Protocol": protocol,
                "AMH": amh,
                "N_Follicles": follicles,
                "E2_day5": e2,
                "AFC": afc,
            }

            st.info("🔄 Analyse en cours…")
            res = predict_response(payload)

            if not res.get("success"):
                st.error("❌ Échec de la prédiction")
                return

            st.subheader("📊 Résultats de la Prédiction")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Classe Prédite", res["predicted_class"])
            with c2:
                st.metric("Niveau de Confiance", f"{res['confidence']*100:.1f}%")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(create_probability_chart(res["probabilities"]), use_container_width=True)
            with c2:
                st.plotly_chart(create_gauge_chart(res["confidence"]), use_container_width=True)

            st.info(res.get("interpretation", ""))

    # ---------------------------------------------------------
    # PAGE 2 — PRÉDICTION PAR LOT
    # ---------------------------------------------------------
    elif page == "Batch":
        st.subheader("📊 Prédiction par Lot")

        file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)
            st.dataframe(df, use_container_width=True)

            required = ["Age", "AMH", "N_Follicles", "E2_day5", "AFC"]
            missing = [c for c in required if c not in df.columns]

            if missing:
                st.error(f"❌ Colonnes manquantes : {missing}")
                return

            if st.button("🚀 Lancer la Prédiction par Lot"):
                clean = df.dropna(subset=required)

                r = requests.post(
                    f"{API_URL}/api/predict/batch",
                    json={"patients": clean.to_dict("records")},
                    timeout=60
                ).json()

                if not r.get("success"):
                    st.error("Échec de la prédiction par lot")
                    return

                preds = r["predictions"]
                clean["Réponse_Prédite"] = [p["predicted_class"] for p in preds]
                clean["Confiance"] = [p["confidence"] for p in preds]

                st.success("✅ Prédiction par lot terminée")
                st.dataframe(clean, use_container_width=True)

                st.download_button("📥 Télécharger les Résultats", clean.to_csv(index=False), "predictions.csv")

    # ---------------------------------------------------------
    # PAGE 3 — INFORMATIONS MODÈLE
    # ---------------------------------------------------------
    else:
        st.subheader("ℹ️ Informations sur le Modèle")

        try:
            info = requests.get(f"{API_URL}/api/model/info").json()
            if info.get("success"):
                st.json(info)
            else:
                st.error("Informations du modèle non disponibles.")
        except Exception as e:
            st.error(str(e))


# -------------------------------------------------------------------------
# RUN APP
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()