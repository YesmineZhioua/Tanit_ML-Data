"""
IVF Patient Response Prediction - Streamlit Interface
====================================================
Modern, attractive interface for patient response stratification
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
import json

# Configuration de la page
st.set_page_config(
    page_title="IVF Response Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    /* Couleurs principales */
    :root {
        --primary: #2E86AB;
        --secondary: #A23B72;
        --success: #06D6A0;
        --warning: #F77F00;
        --danger: #EF476F;
    }
    
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-align: center;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Prédiction result */
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .low-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .optimal-response {
        background: linear-gradient(135deg, #06D6A0 0%, #1B998B 100%);
        color: white;
    }
    
    .high-response {
        background: linear-gradient(135deg, #F77F00 0%, #D62828 100%);
        color: white;
    }
    
    /* Confidence meter */
    .confidence-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Progress bar custom */
    .custom-progress {
        height: 25px;
        border-radius: 12px;
        background: #e0e0e0;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .custom-progress-bar {
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        transition: width 0.6s ease;
    }
</style>
""", unsafe_allow_html=True)

# Configuration API
API_URL = "http://localhost:5000"


def check_api_health() -> bool:
    """Vérifie si l'API est accessible"""
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_response(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Appelle l'API pour obtenir une prédiction"""
    try:
        response = requests.post(
            f"{API_URL}/api/predict",
            json=patient_data,
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {'success': False, 'error': str(e)}


def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Crée un graphique des probabilités"""
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    
    colors = {
        'low': '#667eea',
        'optimal': '#06D6A0',
        'high': '#F77F00'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=values,
            marker=dict(
                color=[colors.get(c, '#667eea') for c in classes],
                line=dict(color='white', width=2)
            ),
            text=[f'{v*100:.1f}%' for v in values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Response Probability Distribution',
            font=dict(size=20, color='#333')
        ),
        xaxis_title='Response Class',
        yaxis_title='Probability',
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400,
        template='plotly_white',
        showlegend=False,
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    return fig


def create_gauge_chart(confidence: float) -> go.Figure:
    """Crée un graphique gauge pour la confiance"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 24, 'color': '#333'}},
        delta={'reference': 80, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffcccb'},
                {'range': [60, 80], 'color': '#ffe6cc'},
                {'range': [80, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(t=50, b=0, l=50, r=50),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"}
    )
    
    return fig


def main():
    # En-tête principal
    st.markdown("""
        <div class="main-header">
            <h1>🔬 IVF Patient Response Predictor</h1>
            <p>AI-Powered Clinical Decision Support System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Vérifier la connexion API
    api_status = check_api_health()
    
    if not api_status:
        st.error("⚠️ **API Server Offline** - Please ensure the Flask backend is running on port 5000")
        st.code("python app.py", language="bash")
        st.stop()
    
    # Sidebar - Informations et navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/medical-doctor.png", width=80)
        st.title("📋 Navigation")
        
        page = st.radio(
            "Select Mode",
            ["Single Patient Prediction", "Batch Prediction", "Model Information"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### 📊 About
        This system uses machine learning to predict patient response 
        to IVF treatment based on clinical parameters.
        
        **Response Classes:**
        - 🔵 **Low**: Under-response
        - 🟢 **Optimal**: Normal response  
        - 🟠 **High**: Over-response (OHSS risk)
        """)
        
        st.markdown("---")
        st.success("✅ API Connected")
    
    # ========================================================================
    # PAGE 1: Single Patient Prediction
    # ========================================================================
    if page == "Single Patient Prediction":
        st.markdown("## 🎯 Single Patient Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Patient Demographics")
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=50,
                value=32,
                help="Patient's age at treatment"
            )
            
            cycle_number = st.number_input(
                "Cycle Number",
                min_value=1,
                max_value=10,
                value=1,
                help="IVF cycle attempt number"
            )
            
            protocol = st.selectbox(
                "Stimulation Protocol",
                ["agonist", "flex anta", "fix anta"],
                help="Type of ovarian stimulation protocol"
            )
        
        with col2:
            st.markdown("### 🧬 Clinical Parameters")
            amh = st.number_input(
                "AMH (ng/mL)",
                min_value=0.0,
                max_value=20.0,
                value=2.5,
                step=0.1,
                help="Anti-Müllerian Hormone level"
            )
            
            afc = st.number_input(
                "AFC (count)",
                min_value=0,
                max_value=50,
                value=15,
                help="Antral Follicle Count"
            )
            
            n_follicles = st.number_input(
                "Number of Follicles",
                min_value=0,
                max_value=50,
                value=15,
                help="Total follicle count"
            )
            
            e2_day5 = st.number_input(
                "E2 Day 5 (pg/mL)",
                min_value=0.0,
                max_value=5000.0,
                value=300.0,
                step=10.0,
                help="Estradiol level on day 5 of stimulation"
            )
        
        st.markdown("---")
        
        # Bouton de prédiction centré
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🚀 Predict Response", use_container_width=True)
        
        if predict_button:
            # Préparer les données
            patient_data = {
                'Age': age,
                'Cycle_number': cycle_number,
                'Protocol': protocol,
                'AMH': amh,
                'N_Follicles': n_follicles,
                'E2_day5': e2_day5,
                'AFC': afc
            }
            
            with st.spinner('🔄 Analyzing patient data...'):
                result = predict_response(patient_data)
            
            if result.get('success'):
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                probabilities = result['probabilities']
                
                # Afficher le résultat principal
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                # Classe de style selon la prédiction
                class_style = f"{predicted_class.lower()}-response"
                
                st.markdown(f"""
                    <div class="prediction-result {class_style}">
                        🎯 Predicted Response: <strong>{predicted_class.upper()}</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics en colonnes
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Confidence",
                        f"{confidence*100:.1f}%",
                        delta="High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
                    )
                
                with col2:
                    st.metric(
                        "Low Risk",
                        f"{probabilities.get('low', 0)*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "OHSS Risk",
                        f"{probabilities.get('high', 0)*100:.1f}%"
                    )
                
                # Graphiques
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        create_probability_chart(probabilities),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        create_gauge_chart(confidence),
                        use_container_width=True
                    )
                
                # Interprétation et recommandations
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 💡 Clinical Interpretation")
                    st.info(result['interpretation'])
                
                with col2:
                    st.markdown("### 📋 Recommendations")
                    for i, rec in enumerate(result['recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
            
            else:
                st.error(f"❌ Prediction Error: {result.get('error', 'Unknown error')}")
    
    # ========================================================================
    # PAGE 2: Batch Prediction
    # ========================================================================
    elif page == "Batch Prediction":
        st.markdown("## 📊 Batch Patient Analysis")
        
        st.info("📁 Upload a CSV file with patient data for batch predictions")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV must contain: Age, AMH, N_Follicles, E2_day5, AFC"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📋 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown(f"**Total patients:** {len(df)}")
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner('🔄 Processing batch predictions...'):
                    patients = df.to_dict('records')
                    
                    try:
                        response = requests.post(
                            f"{API_URL}/api/predict/batch",
                            json={'patients': patients},
                            timeout=60
                        )
                        result = response.json()
                        
                        if result.get('success'):
                            predictions = result['predictions']
                            
                            # Créer DataFrame des résultats
                            results_df = df.copy()
                            results_df['Predicted_Response'] = [p['predicted_class'] for p in predictions]
                            results_df['Confidence'] = [p['confidence'] for p in predictions]
                            
                            st.success(f"✅ Processed {len(predictions)} patients successfully!")
                            
                            # Afficher les résultats
                            st.markdown("### 📊 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Statistiques
                            col1, col2, col3 = st.columns(3)
                            
                            response_counts = results_df['Predicted_Response'].value_counts()
                            
                            with col1:
                                st.metric("Low Responders", response_counts.get('low', 0))
                            with col2:
                                st.metric("Optimal Responders", response_counts.get('optimal', 0))
                            with col3:
                                st.metric("High Responders", response_counts.get('high', 0))
                            
                            # Télécharger les résultats
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        else:
                            st.error(f"❌ Batch prediction failed: {result.get('error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # ========================================================================
    # PAGE 3: Model Information
    # ========================================================================
    else:
        st.markdown("## ℹ️ Model Information")
        
        try:
            response = requests.get(f"{API_URL}/api/model/info")
            info = response.json()
            
            if info.get('success'):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Model Features")
                    st.json(info['features'])
                
                with col2:
                    st.markdown("### 📊 Response Classes")
                    for cls in info['classes']:
                        st.markdown(f"- **{cls.upper()}**")
                
                st.metric("Total Features", info['n_features'])
            else:
                st.error("Failed to load model information")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()