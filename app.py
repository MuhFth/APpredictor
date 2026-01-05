import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import joblib
from io import StringIO

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: white;
        font-size: 1.3em;
        margin-top: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    .category-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL AT STARTUP
# =========================================
@st.cache_resource
def load_saved_model():
    """Load the pre-trained model, scaler, and feature names"""
    try:
        # Load the model file
        data = joblib.load("academic_predictor_model.pkl")
        
        # Check if it's a dictionary (from your Colab notebook) or direct model
        if isinstance(data, dict):
            model = data.get('model', None)
            scaler = data.get('scaler', None)
            feature_names = data.get('feature_names', None)
        else:
            # If it's a direct model object
            model = data
            scaler = None
            feature_names = None
        
        # If no feature names in file, use default
        if feature_names is None:
            feature_names = ['ID_Siswa', 'Persentase_Kehadiran', 'Nilai_Internal_1', 
                            'Nilai_Internal_2', 'Skor_Tugas', 'Jam_Belajar_Harian', 
                            'Nilai_Ujian_Akhir']
        
        return model, scaler, feature_names
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load model automatically
model, scaler, feature_names = load_saved_model()

# =========================================
# SESSION STATE INITIALIZATION
# =========================================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = (model is not None)

# =========================================
# TITLE
# =========================================
st.markdown("<h1>🎓 Academic Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>✨ Prediksi Nilai Akhir Siswa dengan Machine Learning ✨</p>", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/student-male.png", width=150)
    st.title("📊 Model Information")
    
    # Model status indicator
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded Successfully")
        if model is not None:
            st.info(f"**Model Type:** {type(model).__name__}")
            if feature_names:
                st.info(f"**Total Features:** {len(feature_names)}")
                with st.expander("📋 View Feature Names"):
                    for i, feat in enumerate(feature_names, 1):
                        st.write(f"{i}. {feat.replace('_', ' ').title()}")
    else:
        st.error("❌ Model Not Loaded")
    
    st.markdown("---")
    st.markdown("### 💡 About")
    st.info("Aplikasi ini menggunakan model Machine Learning untuk memprediksi performa akademik siswa berdasarkan berbagai faktor.")
    
    st.markdown("---")
    st.markdown("### 📈 How It Works")
    st.markdown("""
    1. **Input Data** - Masukkan data siswa
    2. **Predict** - Klik tombol prediksi
    3. **Get Results** - Lihat hasil prediksi
    """)

# =========================================
# MAIN PAGE
# =========================================
if not st.session_state.model_loaded:
    st.error("❌ Model gagal dimuat. Periksa file model Anda!")
    st.info("💡 Pastikan file 'academic_predictor_model.pkl' ada di folder yang sama dengan app.py")
    st.stop()

# Success message
st.markdown("""
<div style='background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; 
text-align: center; color: white; margin: 20px 0;'>
    ✅ <strong>Model siap digunakan!</strong> Silakan masukkan data siswa di bawah ini.
</div>
""", unsafe_allow_html=True)

# =========================================
# INPUT SECTION
# =========================================
st.markdown("<div class='input-section'>", unsafe_allow_html=True)
st.markdown("### 📝 Masukkan Data Siswa")

if feature_names is None:
    st.error("❌ Feature names tidak terdefinisi!")
    st.stop()

# Create input fields with better labels
feature_labels = {
    'ID_Siswa': 'ID Siswa',
    'Persentase_Kehadiran': 'Persentase Kehadiran (%)',
    'Nilai_Internal_1': 'Nilai Internal 1',
    'Nilai_Internal_2': 'Nilai Internal 2',
    'Skor_Tugas': 'Skor Tugas',
    'Jam_Belajar_Harian': 'Jam Belajar Harian',
    'Nilai_Ujian_Akhir': 'Nilai Ujian Akhir'
}

# Create 3 columns for better layout
col1, col2, col3 = st.columns(3)
feature_values = []

for idx, feature in enumerate(feature_names):
    label = feature_labels.get(feature, feature.replace('_', ' ').title())
    
    # Distribute inputs across 3 columns
    if idx % 3 == 0:
        with col1:
            if feature == 'ID_Siswa':
                value = st.number_input(
                    f"🆔 {label}",
                    min_value=1,
                    max_value=10000,
                    value=1001,
                    step=1,
                    key=f"input_{feature}"
                )
            elif feature == 'Persentase_Kehadiran':
                value = st.number_input(
                    f"📊 {label}",
                    min_value=0.0,
                    max_value=100.0,
                    value=85.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
            else:
                value = st.number_input(
                    f"📚 {label}",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
    elif idx % 3 == 1:
        with col2:
            if feature == 'Jam_Belajar_Harian':
                value = st.number_input(
                    f"⏰ {label}",
                    min_value=0.0,
                    max_value=24.0,
                    value=5.0,
                    step=0.5,
                    key=f"input_{feature}"
                )
            else:
                value = st.number_input(
                    f"📝 {label}",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
    else:
        with col3:
            value = st.number_input(
                f"🎯 {label}",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=0.1,
                key=f"input_{feature}"
            )
    
    feature_values.append(value)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# PREDICTION SECTION
# =========================================
st.markdown("<br>", unsafe_allow_html=True)

# Center the predict button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🎯 PREDIKSI NILAI AKHIR", use_container_width=True)

if predict_button:
    try:
        # Prepare input
        input_data = np.array([feature_values])
        
        # Scale if scaler is available
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Display prediction with animation
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎊 Hasil Prediksi")
        st.markdown(f"""
        <div class="prediction-box">
            Nilai Akhir Prediksi: {prediction:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        # Performance category with colors
        if prediction >= 90:
            category = "🌟 Excellent (A)"
            color = "#00C851"
            emoji = "🎉"
            message = "Luar biasa! Performa sangat memuaskan!"
        elif prediction >= 80:
            category = "👍 Very Good (B+)"
            color = "#33b5e5"
            emoji = "😊"
            message = "Sangat bagus! Pertahankan prestasi ini!"
        elif prediction >= 70:
            category = "✅ Good (B)"
            color = "#ffbb33"
            emoji = "👏"
            message = "Bagus! Terus tingkatkan!"
        elif prediction >= 60:
            category = "⚠️ Average (C)"
            color = "#ff8800"
            emoji = "💪"
            message = "Cukup baik, masih bisa lebih baik lagi!"
        else:
            category = "❌ Needs Improvement (D)"
            color = "#ff4444"
            emoji = "📚"
            message = "Perlu usaha lebih keras!"
        
        st.markdown(f"""
        <div style="background: {color}; padding: 25px; border-radius: 15px; 
        text-align: center; color: white; margin-top: 20px; box-shadow: 0 6px 12px rgba(0,0,0,0.2);">
            <div style="font-size: 28px; margin-bottom: 10px;">{emoji}</div>
            <div style="font-size: 22px; font-weight: bold; margin-bottom: 5px;">{category}</div>
            <div style="font-size: 16px;">{message}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display input summary in expandable section
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Lihat Ringkasan Input Data"):
            input_df = pd.DataFrame({
                'Fitur': [feature_labels.get(f, f.replace('_', ' ').title()) for f in feature_names],
                'Nilai': feature_values
            })
            
            # Style the dataframe
            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Fitur": st.column_config.TextColumn("Fitur", width="medium"),
                    "Nilai": st.column_config.NumberColumn("Nilai", format="%.2f", width="medium")
                }
            )
        
        # Additional insights
        with st.expander("📊 Analisis Data"):
            col_a, col_b, col_c = st.columns(3)
            
            # Safely get indices based on feature names
            try:
                # Find indices dynamically
                id_idx = feature_names.index('ID_Siswa') if 'ID_Siswa' in feature_names else None
                kehadiran_idx = feature_names.index('Persentase_Kehadiran') if 'Persentase_Kehadiran' in feature_names else None
                nilai1_idx = feature_names.index('Nilai_Internal_1') if 'Nilai_Internal_1' in feature_names else None
                nilai2_idx = feature_names.index('Nilai_Internal_2') if 'Nilai_Internal_2' in feature_names else None
                jam_idx = feature_names.index('Jam_Belajar_Harian') if 'Jam_Belajar_Harian' in feature_names else None
                
                with col_a:
                    if nilai1_idx is not None and nilai2_idx is not None:
                        avg_internal = (feature_values[nilai1_idx] + feature_values[nilai2_idx]) / 2
                        st.metric("Rata-rata Nilai Internal", 
                                 f"{avg_internal:.2f}",
                                 delta=None)
                    else:
                        st.metric("Rata-rata Nilai Internal", "N/A")
                
                with col_b:
                    if kehadiran_idx is not None:
                        kehadiran = feature_values[kehadiran_idx]
                        st.metric("Kehadiran", 
                                 f"{kehadiran:.1f}%",
                                 delta="Baik" if kehadiran >= 80 else "Perlu Ditingkatkan")
                    else:
                        st.metric("Kehadiran", "N/A")
                
                with col_c:
                    if jam_idx is not None:
                        jam_belajar = feature_values[jam_idx]
                        st.metric("Jam Belajar/Hari", 
                                 f"{jam_belajar:.1f} jam",
                                 delta="Optimal" if jam_belajar >= 4 else "Kurang")
                    else:
                        st.metric("Jam Belajar/Hari", "N/A")
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan analisis detail: {str(e)}")
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
        st.error("Pastikan input data sesuai dengan format yang diharapkan.")
        st.info("💡 Tip: Cek apakah semua nilai yang dimasukkan valid dan model sudah ter-load dengan benar.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p style='font-size: 16px; margin: 0;'>Made with ❤️ using Streamlit</p>
    <p style='font-size: 14px; margin: 5px 0 0 0; opacity: 0.8;'>🎓 Academic Performance Predictor © 2024</p>
</div>
""", unsafe_allow_html=True)