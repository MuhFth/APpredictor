import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings

warnings.filterwarnings('ignore')

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
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 32px;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3.5em;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    .subtitle {
        text-align: center;
        color: white;
        font-size: 1.4em;
        margin-top: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        padding: 18px 40px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.4);
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.6);
    }
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        margin: 20px 0;
    }
    .success-box {
        background: rgba(0, 200, 81, 0.2);
        border-left: 5px solid #00C851;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
    .info-box {
        background: rgba(33, 150, 243, 0.2);
        border-left: 5px solid #2196F3;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        data = joblib.load("academic_predictor_model_pt3.pkl")
        
        if isinstance(data, dict):
            return (
                data.get('model'),
                data.get('scaler'),
                data.get('feature_names'),
                data.get('metrics', {}),
                data.get('feature_labels', {})
            )
        return None, None, None, {}, {}
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, {}, {}

model, scaler, feature_names, metrics, feature_labels = load_model()

# =========================================
# SESSION STATE
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
    st.image("https://img.icons8.com/clouds/200/student-male.png", width=180)
    st.title("📊 Model Info")
    
    if st.session_state.model_loaded:
        st.success("✅ Model Ready")
        
        if metrics:
            st.markdown("### 📈 Performance")
            if 'test_r2' in metrics:
                r2_pct = metrics['test_r2'] * 100
                st.metric("Accuracy (R²)", f"{r2_pct:.1f}%")
            if 'mae' in metrics:
                st.metric("Avg Error (MAE)", f"{metrics['mae']:.2f} pts")
        
        if feature_names:
            st.markdown("### 📋 Features")
            st.info(f"{len(feature_names)} input features")
            with st.expander("View Details"):
                for i, feat in enumerate(feature_names, 1):
                    label = feature_labels.get(feat, feat.replace('_', ' ').title())
                    st.write(f"{i}. {label}")
    else:
        st.error("❌ Model Not Loaded")
    
    st.markdown("---")
    st.markdown("### 💡 About")
    st.info("""
    Aplikasi ini memprediksi **Final Score** siswa berdasarkan:
    - Kehadiran
    - Nilai Internal
    - Skor Tugas  
    - Jam Belajar
    """)

# =========================================
# MAIN CONTENT
# =========================================

if not st.session_state.model_loaded:
    st.error("❌ Model tidak dapat dimuat!")
    st.info("💡 Pastikan file 'academic_predictor_model.pkl' ada di folder yang sama dengan app.py")
    st.stop()

# Success message
st.markdown("""
<div class='success-box'>
    ✅ <strong>Model siap digunakan!</strong> Masukkan data siswa untuk mendapatkan prediksi nilai akhir.
</div>
""", unsafe_allow_html=True)

# Model info
if metrics:
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'test_r2' in metrics:
            r2_val = metrics['test_r2'] * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">🎯 Accuracy</h3>
                <p style="font-size: 2em; font-weight: bold; color: #333; margin: 10px 0;">{r2_val:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'mae' in metrics:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">📊 Avg Error</h3>
                <p style="font-size: 2em; font-weight: bold; color: #333; margin: 10px 0;">±{metrics['mae']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">📝 Features</h3>
            <p style="font-size: 2em; font-weight: bold; color: #333; margin: 10px 0;">{len(feature_names)}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# INPUT SECTION
# =========================================
st.markdown("<div class='input-section'>", unsafe_allow_html=True)
st.markdown("### 📝 Input Data Siswa")

# Create input fields
n_cols = 3
cols = st.columns(n_cols)
inputs = {}

for idx, feature in enumerate(feature_names):
    label = feature_labels.get(feature, feature.replace('_', ' ').title())
    col_idx = idx % n_cols
    
    with cols[col_idx]:
        if 'Kehadiran' in feature:
            value = st.number_input(
                f"📊 {label}",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
                key=feature,
                help="Persentase kehadiran siswa (0-100%)"
            )
        elif 'Jam' in feature:
            value = st.number_input(
                f"⏰ {label}",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5,
                key=feature,
                help="Jam belajar per hari"
            )
        else:
            value = st.number_input(
                f"📚 {label}",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                key=feature,
                help="Nilai 0-100"
            )
        
        inputs[feature] = value

st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# PREDICT BUTTON
# =========================================
st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🎯 PREDIKSI NILAI AKHIR", use_container_width=True)

# =========================================
# PREDICTION
# =========================================
if predict_button:
    try:
        # Prepare input as DataFrame
        input_values = [inputs[feat] for feat in feature_names]
        input_df = pd.DataFrame([input_values], columns=feature_names)
        
        # Scale input
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Clip to 0-100 range
        prediction = np.clip(prediction, 0, 100)
        
        # Display prediction
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎊 Hasil Prediksi")
        
        st.markdown(f"""
        <div class="prediction-box">
            Final Score: {prediction:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        # Performance category
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
            color = "#dd0404ef"
            emoji = "📚"
            message = "Perlu usaha lebih keras!"
        
        st.markdown(f"""
        <div style="background: {color}; padding: 30px; border-radius: 20px; 
        text-align: center; color: white; margin-top: 25px; box-shadow: 0 8px 20px rgba(0,0,0,0.25);">
            <div style="font-size: 40px; margin-bottom: 15px;">{emoji}</div>
            <div style="font-size: 26px; font-weight: bold; margin-bottom: 10px;">{category}</div>
            <div style="font-size: 18px;">{message}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Input summary
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Lihat Detail Input"):
            summary_data = []
            for feat in feature_names:
                label = feature_labels.get(feat, feat.replace('_', ' ').title())
                summary_data.append({
                    'Parameter': label,
                    'Nilai': f"{inputs[feat]:.1f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Analysis
        with st.expander("📊 Analisis Performa"):
            col_a, col_b, col_c = st.columns(3)
            
            # Get feature indices
            kehadiran_idx = next((i for i, f in enumerate(feature_names) if 'Kehadiran' in f), None)
            nilai1_idx = next((i for i, f in enumerate(feature_names) if 'Internal_1' in f), None)
            nilai2_idx = next((i for i, f in enumerate(feature_names) if 'Internal_2' in f), None)
            jam_idx = next((i for i, f in enumerate(feature_names) if 'Jam' in f), None)
            
            with col_a:
                if nilai1_idx is not None and nilai2_idx is not None:
                    avg_internal = (input_values[nilai1_idx] + input_values[nilai2_idx]) / 2
                    st.metric(
                        "Rata-rata Nilai Internal",
                        f"{avg_internal:.1f}",
                        delta="Baik" if avg_internal >= 70 else "Perlu Ditingkatkan"
                    )
            
            with col_b:
                if kehadiran_idx is not None:
                    kehadiran = input_values[kehadiran_idx]
                    st.metric(
                        "Status Kehadiran",
                        f"{kehadiran:.0f}%",
                        delta="Baik" if kehadiran >= 80 else "Kurang"
                    )
            
            with col_c:
                if jam_idx is not None:
                    jam = input_values[jam_idx]
                    st.metric(
                        "Jam Belajar/Hari",
                        f"{jam:.1f} jam",
                        delta="Optimal" if jam >= 5 else "Kurang"
                    )
        
        # Visualization
        with st.expander("📈 Visualisasi Kontribusi"):
            # Create radar chart
            fig = go.Figure()
            
            # Normalize values to 0-100 scale
            normalized_values = []
            labels = []
            
            for feat in feature_names:
                label = feature_labels.get(feat, feat.replace('_', ' ').title())
                value = inputs[feat]
                
                # Normalize to 100 scale
                if 'Jam' in feat:
                    norm_value = (value / 12) * 100  # Assuming max 12 hours
                else:
                    norm_value = value
                
                normalized_values.append(norm_value)
                labels.append(label)
            
            # Add closing point
            normalized_values.append(normalized_values[0])
            labels.append(labels[0])
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=labels,
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='#667eea', width=2),
                name='Student Performance'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                title="Performance Radar Chart",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 25px;'>
    <p style='font-size: 18px; margin: 0; font-weight: 500;'>Made with ❤️ using Streamlit & Machine Learning</p>
    <p style='font-size: 14px; margin: 10px 0 0 0; opacity: 0.8;'>🎓 Academic Performance Predictor © 2024</p>
</div>
""", unsafe_allow_html=True)