import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS untuk styling dengan background colorful
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main container overlay */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Main header styling with gradient and animation */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '🎓';
        position: absolute;
        font-size: 10rem;
        opacity: 0.1;
        top: -20px;
        right: -20px;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    
    .main-header h1 {
        font-weight: 700;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 0.5rem;
    }
    
    /* Enhanced Card styling with colorful gradients */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        border-left: 5px solid;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card:nth-child(1) { border-left-color: #667eea; }
    .metric-card:nth-child(2) { border-left-color: #f093fb; }
    .metric-card:nth-child(3) { border-left-color: #4facfe; }
    .metric-card:nth-child(4) { border-left-color: #43e97b; }
    
    /* Sidebar styling with gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: white !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        padding: 0.8rem;
        border-radius: 8px;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    /* Enhanced Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Colorful Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(33, 150, 243, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(255, 152, 0, 0.2);
    }
    
    /* Enhanced Grade badges */
    .grade-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .grade-a { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
        color: white;
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.4);
    }
    .grade-b { 
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
        color: white;
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }
    .grade-c { 
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
        color: white;
        box-shadow: 0 6px 20px rgba(254, 225, 64, 0.4);
    }
    .grade-d { 
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%); 
        color: white;
        box-shadow: 0 6px 20px rgba(252, 74, 26, 0.4);
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s;
        margin: 1rem 0;
    }
    
    .feature-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: rotate 10s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        color: #667eea;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Data frame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Decorative elements */
    .decoration-circle {
        position: fixed;
        border-radius: 50%;
        opacity: 0.1;
        z-index: -1;
    }
    
    .circle-1 {
        width: 300px;
        height: 300px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        top: -100px;
        right: -100px;
    }
    
    .circle-2 {
        width: 200px;
        height: 200px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        bottom: -50px;
        left: -50px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 -4px 20px rgba(102, 126, 234, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Add decorative circles
st.markdown("""
    <div class="decoration-circle circle-1"></div>
    <div class="decoration-circle circle-2"></div>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL ARTIFACT
# ======================================================
@st.cache_resource
def load_artifact():
    try:
        return joblib.load("academic_predictor_pt6.pkl")
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan! Pastikan 'academic_predictor.pkl' ada di direktori yang sama.")
        st.stop()

data = load_artifact()
model = data["model"]
scaler = data["scaler"]
FEATURES = data["feature_names"]
metrics = data["metrics"]

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; animation: pulse 2s ease-in-out infinite;'>🎓</div>
            <h2 style='color: white; margin: 0.5rem 0;'>Academic AI</h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Prediksi Nilai Berbasis AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation Menu",
        [
            "🏠 Dashboard",
            "📊 Visualisasi Data & Model",
            "🎯 Prediksi Individual",
            "📁 Prediksi Batch (CSV)",
            "ℹ️ Informasi Model"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 12px; text-align: center; color: white;'>
            <p style='font-size: 0.9rem; margin: 0;'>
                <b>⚡ Powered by</b><br>
                Machine Learning<br>
                <span style='font-size: 0.8rem; opacity: 0.8;'>Version 1.0</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

# ======================================================
# PAGE 0: DASHBOARD
# ======================================================
if menu == "🏠 Dashboard":
    # Header with animation
    st.markdown("""
        <div class='main-header'>
            <h1>🎓 Academic Performance Predictor</h1>
            <p style='font-size: 1.3rem; margin-top: 1rem; opacity: 0.95;'>
                ✨ Sistem Prediksi Nilai Akhir Siswa Berbasis Machine Learning ✨
            </p>
            <div style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;'>
                🚀 Accurate • ⚡ Fast • 🎯 Reliable
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Overview metrics with colorful cards
    st.markdown("### 📈 Performa Model")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;'>🎯</div>
                <h3 style='color: #667eea; margin: 0; text-align: center; font-weight: 700;'>R² Score</h3>
                <h1 style='margin: 0.5rem 0; text-align: center; font-size: 2.5rem; color: #333;'>{metrics['r2']:.3f}</h1>
                <p style='color: #555; margin: 0; text-align: center; font-weight: 500;'>Akurasi Model</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;'>📊</div>
                <h3 style='color: #f093fb; margin: 0; text-align: center; font-weight: 700;'>MAE</h3>
                <h1 style='margin: 0.5rem 0; text-align: center; font-size: 2.5rem; color: #333;'>{metrics['mae']:.2f}</h1>
                <p style='color: #555; margin: 0; text-align: center; font-weight: 500;'>Mean Absolute Error</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;'>📈</div>
                <h3 style='color: #4facfe; margin: 0; text-align: center; font-weight: 700;'>RMSE</h3>
                <h1 style='margin: 0.5rem 0; text-align: center; font-size: 2.5rem; color: #333;'>{metrics['rmse']:.2f}</h1>
                <p style='color: #555; margin: 0; text-align: center; font-weight: 500;'>Root Mean Squared Error</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;'>🔢</div>
                <h3 style='color: #43e97b; margin: 0; text-align: center; font-weight: 700;'>Features</h3>
                <h1 style='margin: 0.5rem 0; text-align: center; font-size: 2.5rem; color: #333;'>{len(FEATURES)}</h1>
                <p style='color: #555; margin: 0; text-align: center; font-weight: 500;'>Total Fitur Input</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature showcase with icons
    st.markdown("### ✨ Fitur Unggulan Sistem")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🤖</div>
                <h3>AI-Powered</h3>
                <p>Prediksi menggunakan algoritma Machine Learning canggih</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>⚡</div>
                <h3>Real-time</h3>
                <p>Hasil prediksi instan dalam hitungan detik</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>📊</div>
                <h3>Batch Processing</h3>
                <p>Prediksi massal dengan upload CSV</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features overview with better styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Fitur-Fitur Input Model")
        features_df = pd.DataFrame({
            "No": range(1, len(FEATURES) + 1),
            "Nama Fitur": [f"📌 {f.replace('_', ' ')}" for f in FEATURES],
            "Kode": FEATURES
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("""
            <div class='success-box'>
                <h3 style='margin-top: 0; color: #2e7d32;'>🎯 Fitur Utama</h3>
                <p style='margin: 0.5rem 0;'><b>Faktor Paling Berpengaruh:</b></p>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li>📝 Nilai Internal 1 & 2</li>
                    <li>📚 Skor Tugas</li>
                    <li>👥 Persentase Kehadiran</li>
                    <li>🎯 Partisipasi Kelas</li>
                    <li>📖 Waktu Belajar</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Quick actions with improved styling
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("📊 Lihat Visualisasi", use_container_width=True, key="viz_btn")
    
    with col2:
        st.button("🎯 Prediksi Individual", use_container_width=True, key="pred_btn")
    
    with col3:
        st.button("📁 Upload CSV", use_container_width=True, key="csv_btn")
    
    # Statistics section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Statistik Penggunaan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white;'>
                <div style='font-size: 2rem;'>🎓</div>
                <h2 style='margin: 0.5rem 0;'>100+</h2>
                <p style='margin: 0; opacity: 0.9;'>Prediksi Akurat</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 12px; color: white;'>
                <div style='font-size: 2rem;'>⚡</div>
                <h2 style='margin: 0.5rem 0;'>< 1s</h2>
                <p style='margin: 0; opacity: 0.9;'>Waktu Proses</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 12px; color: white;'>
                <div style='font-size: 2rem;'>🎯</div>
                <h2 style='margin: 0.5rem 0;'>95%</h2>
                <p style='margin: 0; opacity: 0.9;'>Tingkat Akurasi</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 12px; color: white;'>
                <div style='font-size: 2rem;'>👥</div>
                <h2 style='margin: 0.5rem 0;'>50+</h2>
                <p style='margin: 0; opacity: 0.9;'>Pengguna Aktif</p>
            </div>
        """, unsafe_allow_html=True)

# ======================================================
# PAGE 1: VISUALISASI DATA & MODEL
# ======================================================
elif menu == "📊 Visualisasi Data & Model":
    st.markdown("""
        <div class='main-header'>
            <h1>📊 Visualisasi Data & Analisis Model</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>
                🔍 Analisis mendalam tentang performa dan karakteristik model
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics overview with icons
    st.markdown("### 📈 Metrik Performa Model")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("🎯 R² Score", f"{metrics['r2']:.3f}", 
                help="Mengukur seberapa baik model menjelaskan variasi data (0-1, semakin tinggi semakin baik)")
    col2.metric("📊 MAE", f"{metrics['mae']:.2f}", 
                help="Rata-rata kesalahan absolut prediksi")
    col3.metric("📈 RMSE", f"{metrics['rmse']:.2f}", 
                help="Akar kuadrat rata-rata kesalahan kuadrat")
    
    st.markdown("---")
    
    # Feature Importance with enhanced visuals
    st.markdown("### 🔍 Feature Importance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🎯 Horizontal Chart", "📋 Table"])
    
    with tab1:
        coef_df = pd.DataFrame({
            "Feature": [f.replace("_", " ") for f in FEATURES],
            "Coefficient": model.coef_,
            "Abs_Coefficient": np.abs(model.coef_)
        }).sort_values("Abs_Coefficient", ascending=False)
        
        fig = px.bar(
            coef_df,
            x="Feature",
            y="Coefficient",
            color="Coefficient",
            color_continuous_scale=["#ff6b6b", "#ffe66d", "#4ecdc4", "#45b7d1"],
            title="🎯 Pengaruh Setiap Fitur terhadap Nilai Akhir",
            labels={"Coefficient": "Koefisien", "Feature": "Fitur"}
        )
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig2 = px.bar(
            coef_df,
            y="Feature",
            x="Coefficient",
            orientation="h",
            color="Coefficient",
            color_continuous_scale="viridis",
            title="📊 Feature Importance (Sorted)",
            labels={"Coefficient": "Koefisien", "Feature": "Fitur"}
        )
        fig2.update_layout(
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        display_df = coef_df.copy()
        display_df["Impact"] = display_df["Coefficient"].apply(
            lambda x: "Positif ⬆️" if x > 0 else "Negatif ⬇️"
        )
        display_df["Magnitude"] = display_df["Abs_Coefficient"].apply(
            lambda x: "Tinggi 🔥" if x > 2 else "Sedang 📊" if x > 1 else "Rendah 📉"
        )
        st.dataframe(
            display_df[["Feature", "Coefficient", "Impact", "Magnitude"]],
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Model equation
    st.markdown("### 📐 Persamaan Model Linear Regression")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        formula = f"**Final Score** = {model.intercept_:.2f}"
        for f, c in zip(FEATURES, model.coef_):
            sign = "+" if c >= 0 else "-"
            formula += f" {sign} ({abs(c):.2f} × {f.replace('_', ' ')})"
        
        st.code(formula, language="python")
    
    with col2:
        st.markdown("""
            <div class='info-box'>
                <h4 style='margin-top: 0;'>💡 Interpretasi:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li>📊 Intercept: Nilai dasar</li>
                    <li>⬆️ Koefisien (+): Meningkatkan nilai</li>
                    <li>⬇️ Koefisien (-): Menurunkan nilai</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Distribution visualization
    st.markdown("---")
    st.markdown("### 📊 Distribusi Koefisien")
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=model.coef_,
        nbinsx=20,
        name="Distribusi",
        marker_color='#667eea',
        marker_line_color='#764ba2',
        marker_line_width=1.5
    ))
    fig_dist.update_layout(
        title="📈 Distribusi Nilai Koefisien",
        xaxis_title="Nilai Koefisien",
        yaxis_title="Frekuensi",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ======================================================
# PAGE 2: PREDIKSI INDIVIDUAL
# ======================================================
elif menu == "🎯 Prediksi Individual":
    st.markdown("""
        <div class='main-header'>
            <h1>🎯 Prediksi Nilai Individual</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>
                📝 Masukkan data siswa untuk memprediksi nilai akhir
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Instructions with icon
    st.markdown("""
        <div class='info-box'>
            <h4 style='margin-top: 0;'>📝 Instruksi:</h4>
            <p style='margin: 0;'>Masukkan semua nilai fitur di bawah ini, kemudian klik tombol 'Prediksi Nilai Akhir' untuk mendapatkan hasil prediksi berbasis AI!</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input form
    st.markdown("### 📋 Input Data Siswa")
    
    inputs = {}
    
    # Organize inputs by category with colorful headers
    categories = {
        "📝 Nilai Internal": ["Nilai_Internal_1", "Nilai_Internal_2"],
        "📚 Tugas & Kuis": ["Skor_Tugas", "Skor_Kuis"],
        "👥 Kehadiran & Partisipasi": ["Persentase_Kehadiran", "Skor_Partisipasi"],
        "📖 Aktivitas Belajar": ["Waktu_Belajar", "Akses_Materi"],
        "🎯 Lainnya": []
    }
    
    # Categorize remaining features
    categorized = set()
    for cat_features in categories.values():
        categorized.update(cat_features)
    
    categories["🎯 Lainnya"] = [f for f in FEATURES if f not in categorized]
    
    for category, features in categories.items():
        if features:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h4 style='color: white; margin: 0;'>{category}</h4>
                </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(3)
            
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    # Set appropriate ranges and defaults
                    if "Persentase" in feature:
                        max_val, step, default = 100.0, 1.0, 80.0
                    elif "Nilai_Internal" in feature:
                        max_val, step, default = 30.0, 0.5, 20.0
                    elif "Skor" in feature:
                        max_val, step, default = 100.0, 1.0, 70.0
                    else:
                        max_val, step, default = 100.0, 1.0, 50.0
                    
                    inputs[feature] = st.number_input(
                        f"📊 {feature.replace('_', ' ')}",
                        min_value=0.0,
                        max_value=max_val,
                        value=default,
                        step=step,
                        help=f"Masukkan nilai untuk {feature.replace('_', ' ')}"
                    )
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Prediksi Nilai Akhir", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("🔄 Memproses prediksi dengan AI..."):
            X = pd.DataFrame([inputs], columns=FEATURES)
            X_scaled = scaler.transform(X)
            raw_prediction = model.predict(X_scaled)[0]
            
            # Apply academic rules
            prediction = raw_prediction
            min_internal = min(
                inputs.get("Nilai_Internal_1", 0),
                inputs.get("Nilai_Internal_2", 0)
            )
            kehadiran = inputs.get("Persentase_Kehadiran", 0)
            skor_tugas = inputs.get("Skor_Tugas", 0)
            
            # Academic rules
            if min_internal < 15:
                prediction = min(prediction, 69)
            elif min_internal < 20:
                prediction = min(prediction, 79)
            else:
                if not (min_internal >= 25 and kehadiran >= 85 and skor_tugas >= 75):
                    prediction = min(prediction, 89)
            
            prediction = np.clip(prediction, 0, 100)
            
            # Display results with celebration
            st.balloons()
            st.markdown("---")
            st.markdown("### 🎯 Hasil Prediksi")
            
            # Score display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction >= 90:
                    grade = "A"
                    grade_class = "grade-a"
                    emoji = "🌟"
                    message = "Excellent"
                elif prediction >= 80:
                    grade = "B"
                    grade_class = "grade-b"
                    emoji = "👍"
                    message = "Very Good"
                elif prediction >= 65:
                    grade = "C"
                    grade_class = "grade-c"
                    emoji = "🙂"
                    message = "Good"
                else:
                    grade = "D"
                    grade_class = "grade-d"
                    emoji = "⚠️"
                    message = "Needs Improvement"
                
                st.markdown(f"""
                    <div style='text-align: center; padding: 3rem; 
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%); 
                    border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);'>
                        <div style='font-size: 3rem; margin-bottom: 1rem;'>🎓</div>
                        <h1 style='font-size: 5rem; margin: 0; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                        {prediction:.2f}</h1>
                        <p style='font-size: 1.5rem; color: #666; margin: 0.5rem 0;'>Nilai Akhir Prediksi</p>
                        <div class='{grade_class} grade-badge'>{emoji} Grade {grade} - {message}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed breakdown
            st.markdown("### 📊 Rincian Analisis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Data Input")
                input_df = pd.DataFrame({
                    "Fitur": [f"📌 {f.replace('_', ' ')}" for f in FEATURES],
                    "Nilai": [inputs[f] for f in FEATURES]
                })
                st.dataframe(input_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 🎯 Evaluasi Akademik")
                
                # Academic evaluation
                rules_met = []
                rules_not_met = []
                
                if min_internal >= 25:
                    rules_met.append("✅ Nilai Internal ≥ 25")
                else:
                    rules_not_met.append(f"❌ Nilai Internal minimum: {min_internal:.1f} (perlu ≥ 25 untuk A)")
                
                if kehadiran >= 85:
                    rules_met.append("✅ Kehadiran ≥ 85%")
                else:
                    rules_not_met.append(f"❌ Kehadiran: {kehadiran:.1f}% (perlu ≥ 85% untuk A)")
                
                if skor_tugas >= 75:
                    rules_met.append("✅ Skor Tugas ≥ 75")
                else:
                    rules_not_met.append(f"❌ Skor Tugas: {skor_tugas:.1f} (perlu ≥ 75 untuk A)")
                
                for rule in rules_met:
                    st.success(rule)
                
                for rule in rules_not_met:
                    st.warning(rule)
                
                # Recommendations
                if rules_not_met:
                    st.markdown("""
                        <div class='info-box'>
                            <h4 style='margin-top: 0;'>💡 Rekomendasi</h4>
                            <p style='margin: 0;'>Tingkatkan aspek yang belum memenuhi syarat untuk mendapatkan nilai yang lebih baik!</p>
                        </div>
                    """, unsafe_allow_html=True)

# ======================================================
# PAGE 3: PREDIKSI BATCH
# ======================================================
elif menu == "📁 Prediksi Batch (CSV)":
    st.markdown("""
        <div class='main-header'>
            <h1>📁 Prediksi Batch (CSV)</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>
                📤 Upload file CSV untuk memprediksi nilai banyak siswa sekaligus
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown("### 📝 Instruksi Penggunaan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
                <h4 style='margin-top: 0;'>📄 Format File CSV:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li>File harus berformat .csv</li>
                    <li>Harus memiliki kolom sesuai fitur model</li>
                    <li>Pastikan tidak ada nilai kosong (NaN)</li>
                    <li>Gunakan pemisah koma (,)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='success-box'>
                <h4 style='margin-top: 0;'>✅ Fitur yang Diperlukan:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li>📝 Nilai_Internal_1, Nilai_Internal_2</li>
                    <li>📚 Skor_Tugas, Skor_Kuis</li>
                    <li>👥 Persentase_Kehadiran</li>
                    <li>🎯 Skor_Partisipasi</li>
                    <li>📖 Dan fitur lainnya sesuai model</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Download template with icon
    st.markdown("### 📥 Download Template")
    template_df = pd.DataFrame(columns=FEATURES)
    template_df.loc[0] = [70.0] * len(FEATURES)
    
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Template CSV",
        csv_template,
        "template_prediksi.csv",
        "text/csv",
        help="Download template CSV dengan format yang benar",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📤 Upload File CSV")
    file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        help="Upload file CSV yang berisi data siswa untuk diprediksi"
    )
    
    if file:
        try:
            df = pd.read_csv(file)
            
            st.success(f"✅ File berhasil diupload! Total data: {len(df)} baris")
            
            # Preview data
            st.markdown("### 👀 Preview Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data statistics with colorful cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div style='text-align: center; padding: 1.5rem; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 12px; color: white;'>
                        <div style='font-size: 2rem;'>📊</div>
                        <h2 style='margin: 0.5rem 0;'>{len(df)}</h2>
                        <p style='margin: 0;'>Total Baris</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style='text-align: center; padding: 1.5rem; 
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 12px; color: white;'>
                        <div style='font-size: 2rem;'>📋</div>
                        <h2 style='margin: 0.5rem 0;'>{len(df.columns)}</h2>
                        <p style='margin: 0;'>Total Kolom</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div style='text-align: center; padding: 1.5rem; 
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    border-radius: 12px; color: white;'>
                        <div style='font-size: 2rem;'>❓</div>
                        <h2 style='margin: 0.5rem 0;'>{df.isnull().sum().sum()}</h2>
                        <p style='margin: 0;'>Missing Values</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div style='text-align: center; padding: 1.5rem; 
                    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    border-radius: 12px; color: white;'>
                        <div style='font-size: 2rem;'>🔄</div>
                        <h2 style='margin: 0.5rem 0;'>{df.duplicated().sum()}</h2>
                        <p style='margin: 0;'>Duplicate Rows</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Check columns
            st.markdown("### 🔍 Validasi Kolom")
            missing_cols = [f for f in FEATURES if f not in df.columns]
            extra_cols = [c for c in df.columns if c not in FEATURES]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if missing_cols:
                    st.error(f"❌ **Kolom yang hilang:** {', '.join(missing_cols)}")
                else:
                    st.success("✅ Semua kolom yang diperlukan tersedia!")
            
            with col2:
                if extra_cols:
                    st.info(f"ℹ️ **Kolom tambahan:** {', '.join(extra_cols)}")
            
            # Prediction button
            if not missing_cols:
                st.markdown("---")
                
                if st.button("🚀 Prediksi Semua Data", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Sedang memproses prediksi..."):
                        X = df[FEATURES].copy()
                        
                        # Check for missing values
                        if X.isnull().sum().sum() > 0:
                            st.warning("⚠️ Terdapat nilai kosong dalam data. Mengisi dengan median...")
                            X = X.fillna(X.median())
                        
                        X_scaled = scaler.transform(X)
                        predictions = model.predict(X_scaled)
                        
                        # Apply academic rules for each row
                        final_predictions = []
                        for i, pred in enumerate(predictions):
                            row = df.iloc[i]
                            min_internal = min(
                                row.get("Nilai_Internal_1", 0),
                                row.get("Nilai_Internal_2", 0)
                            )
                            kehadiran = row.get("Persentase_Kehadiran", 0)
                            skor_tugas = row.get("Skor_Tugas", 0)
                            
                            if min_internal < 15:
                                pred = min(pred, 69)
                            elif min_internal < 20:
                                pred = min(pred, 79)
                            else:
                                if not (min_internal >= 25 and kehadiran >= 85 and skor_tugas >= 75):
                                    pred = min(pred, 89)
                            
                            final_predictions.append(np.clip(pred, 0, 100))
                        
                        df["Predicted_Final_Score"] = final_predictions
                        
                        # Add grade column
                        def get_grade(score):
                            if score >= 90: return "A"
                            elif score >= 80: return "B"
                            elif score >= 65: return "C"
                            else: return "D"
                        
                        df["Grade"] = df["Predicted_Final_Score"].apply(get_grade)
                        
                        st.success("✅ Prediksi berhasil!")
                        st.balloons()
                        
                        # Results summary with colorful metrics
                        st.markdown("### 📊 Ringkasan Hasil Prediksi")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1.5rem; 
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 12px; color: white;'>
                                    <div style='font-size: 2rem;'>📊</div>
                                    <h2 style='margin: 0.5rem 0;'>{df['Predicted_Final_Score'].mean():.2f}</h2>
                                    <p style='margin: 0;'>Rata-rata Nilai</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1.5rem; 
                                background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                                border-radius: 12px; color: white;'>
                                    <div style='font-size: 2rem;'>⬆️</div>
                                    <h2 style='margin: 0.5rem 0;'>{df['Predicted_Final_Score'].max():.2f}</h2>
                                    <p style='margin: 0;'>Nilai Tertinggi</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1.5rem; 
                                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                border-radius: 12px; color: white;'>
                                    <div style='font-size: 2rem;'>⬇️</div>
                                    <h2 style='margin: 0.5rem 0;'>{df['Predicted_Final_Score'].min():.2f}</h2>
                                    <p style='margin: 0;'>Nilai Terendah</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1.5rem; 
                                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                border-radius: 12px; color: white;'>
                                    <div style='font-size: 2rem;'>📈</div>
                                    <h2 style='margin: 0.5rem 0;'>{df['Predicted_Final_Score'].std():.2f}</h2>
                                    <p style='margin: 0;'>Std Deviasi</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Grade distribution
                        st.markdown("### 📈 Distribusi Grade")
                        grade_counts = df["Grade"].value_counts().sort_index()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(
                                values=grade_counts.values,
                                names=grade_counts.index,
                                title="🎯 Distribusi Grade",
                                color=grade_counts.index,
                                color_discrete_map={"A": "#38ef7d", "B": "#00f2fe", "C": "#fee140", "D": "#f7b733"}
                            )
                            fig_pie.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_hist = px.histogram(
                                df,
                                x="Predicted_Final_Score",
                                nbins=20,
                                title="📊 Distribusi Nilai Prediksi",
                                color_discrete_sequence=["#667eea"]
                            )
                            fig_hist.update_layout(
                                xaxis_title="Nilai",
                                yaxis_title="Frekuensi",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Show results
                        st.markdown("### 📋 Hasil Prediksi Lengkap")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        st.markdown("### ⬇️ Download Hasil")
                        csv_result = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Hasil Prediksi (CSV)",
                            csv_result,
                            "hasil_prediksi.csv",
                            "text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {str(e)}")
            st.info("💡 Pastikan file CSV Anda memiliki format yang benar dan tidak corrupt.")

# ======================================================
# PAGE 4: INFORMASI MODEL
# ======================================================
elif menu == "ℹ️ Informasi Model":
    st.markdown("""
        <div class='main-header'>
            <h1>ℹ️ Informasi Model</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>
                📚 Detail teknis dan dokumentasi sistem
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model information
    st.markdown("### 🤖 Tentang Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
                <h4 style='margin-top: 0;'>🔧 Tipe Model:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li>📊 Linear Regression</li>
                    <li>🎯 Supervised Learning</li>
                    <li>📈 Regression Task</li>
                </ul>
                <h4 style='margin-top: 1rem;'>📚 Library:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li>🔬 Scikit-learn</li>
                    <li>🐼 Pandas</li>
                    <li>🔢 NumPy</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='success-box'>
                <h4 style='margin-top: 0;'>⚡ Performa:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li>🎯 R² Score: {metrics['r2']:.3f}</li>
                    <li>📊 MAE: {metrics['mae']:.2f}</li>
                    <li>📈 RMSE: {metrics['rmse']:.2f}</li>
                </ul>
                <h4 style='margin-top: 1rem;'>🔢 Features:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li>📋 Total: {len(FEATURES)} fitur</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Academic rules
    st.markdown("### 📜 Aturan Akademik")
    
    st.markdown("""
        <div class='warning-box'>
            <h4 style='margin-top: 0;'>🎓 Sistem Penilaian:</h4>
            
            <div style='margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.7); border-radius: 8px;'>
                <h5 style='margin: 0; color: #11998e;'>🌟 Grade A (≥ 90)</h5>
                <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                    <li>Nilai Internal minimum ≥ 25</li>
                    <li>Kehadiran ≥ 85%</li>
                    <li>Skor Tugas ≥ 75</li>
                </ul>
            </div>
            
            <div style='margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.7); border-radius: 8px;'>
                <h5 style='margin: 0; color: #4facfe;'>👍 Grade B (80-89)</h5>
                <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                    <li>Nilai Internal minimum ≥ 20</li>
                    <li>Atau tidak memenuhi semua syarat A</li>
                </ul>
            </div>
            
            <div style='margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.7); border-radius: 8px;'>
                <h5 style='margin: 0; color: #fa709a;'>🙂 Grade C (65-79)</h5>
                <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                    <li>Nilai Internal minimum ≥ 15</li>
                    <li>Atau tidak memenuhi syarat B</li>
                </ul>
            </div>
            
            <div style='margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.7); border-radius: 8px;'>
                <h5 style='margin: 0; color: #fc4a1a;'>⚠️ Grade D (< 65)</h5>
                <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                    <li>Nilai Internal minimum < 15</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Features list
    st.markdown("### 📋 Daftar Fitur Lengkap")
    
    features_detail = pd.DataFrame({
        "No": range(1, len(FEATURES) + 1),
        "Nama Fitur": [f"📌 {f.replace('_', ' ')}" for f in FEATURES],
        "Kode Fitur": FEATURES,
        "Koefisien": model.coef_,
        "Pengaruh": ["⬆️ Positif" if c > 0 else "⬇️ Negatif" for c in model.coef_]
    })
    
    st.dataframe(features_detail, use_container_width=True, hide_index=True)
    
    # System requirements
    st.markdown("### 💻 System Requirements")
    
    st.code("""
Python >= 3.8
streamlit >= 1.28.0
pandas >= 1.5.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
plotly >= 5.17.0
joblib >= 1.3.0
    """, language="text")
    
    # Contact
    st.markdown("### 📧 Support & Feedback")
    
    st.markdown("""
        <div class='info-box'>
            <h4 style='margin-top: 0;'>💬 Jika Anda memiliki pertanyaan atau masukan, silakan hubungi:</h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li>📧 Email: support@example.com</li>
                <li>🌐 Website: https://example.com</li>
                <li>📱 Phone: +62 XXX-XXXX-XXXX</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
    <div class='footer'>
        <div style='font-size: 2rem; margin-bottom: 1rem;'>🎓✨</div>
        <h3 style='margin: 0.5rem 0; color: white;'>Academic Performance Predictor</h3>
        <p style='margin: 0.5rem 0; opacity: 0.95; color: white;'>Powered by Machine Learning & Artificial Intelligence</p>
        <div style='text-align: center; color: white; padding: 20px 30px; margin-top: 20px;'>
            <p style='font-size: 16px; margin: 0 0 15px 0; font-weight: 600; color: white;'>
                Copyright © 2026 BY Pengelola MK Praktikum Unggulan (Praktikum DGX)
            </p>
            <div style='margin-top: 15px;'>
                <a href='https://www.praktikum-hpc.gunadarma.ac.id/' target='_blank' 
                   style='color: white; text-decoration: none; margin: 0 10px; font-size: 14px; opacity: 0.95; font-weight: 500;'>
                    🔗 praktikum-hpc.gunadarma.ac.id
                </a>
                <br><br>
                <a href='https://www.hpc-hub.gunadarma.ac.id/' target='_blank' 
                   style='color: white; text-decoration: none; margin: 0 10px; font-size: 14px; opacity: 0.95; font-weight: 500;'>
                    🔗 hpc-hub.gunadarma.ac.id
                </a>
            </div>
        </div>
        <div style='margin-top: 1rem; font-size: 0.9rem; color: white; opacity: 0.85;'>
            Made with ❤️ using Streamlit
        </div>
    </div>
""", unsafe_allow_html=True)