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
    .probability-bar {
        background: linear-gradient(90deg, #ff4444 0%, #ffbb33 50%, #00C851 100%);
        height: 30px;
        border-radius: 15px;
        position: relative;
        margin: 20px 0;
    }
    .probability-indicator {
        position: absolute;
        top: -5px;
        width: 4px;
        height: 40px;
        background: white;
        box-shadow: 0 0 10px rgba(255,255,255,0.8);
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL AND SCALER
# =========================================
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler from separate files"""
    try:
        # Load model
        model = joblib.load("model_kelulusan.pkl")
        
        # Load scaler
        scaler = joblib.load("scaler_kelulusan.pkl")
        
        # Define feature names (MUST match the training data exactly)
        feature_names = [
            'Attendance (%)',
            'Internal Test 1 (out of 40)',
            'Internal Test 2 (out of 40)',
            'Assignment Score (out of 10)',
            'Final Exam Marks (out of 100)'
        ]
        
        # Define feature labels for better display
        feature_labels = {
            'Attendance (%)': 'Persentase Kehadiran (%)',
            'Internal Test 1 (out of 40)': 'Nilai Internal 1 (dari 40)',
            'Internal Test 2 (out of 40)': 'Nilai Internal 2 (dari 40)',
            'Assignment Score (out of 10)': 'Skor Tugas (dari 10)',
            'Final Exam Marks (out of 100)': 'Nilai Ujian Akhir (dari 100)'
        }
        
        # Mock metrics (you can calculate these from your training data)
        metrics = {
            'accuracy': 0.92,  # Replace with actual accuracy
            'model_type': 'Classification'
        }
        
        return model, scaler, feature_names, metrics, feature_labels
        
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        st.info("Make sure 'model.pkl' and 'scaler.pkl' are in the same folder as app.py")
        return None, None, None, {}, {}
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None, {}, {}

model, scaler, feature_names, metrics, feature_labels = load_model_and_scaler()

# =========================================
# SESSION STATE
# =========================================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = (model is not None and scaler is not None)

# =========================================
# TITLE
# =========================================
st.markdown("<h1>🎓 Academic Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>✨ Prediksi Status Kelulusan Siswa dengan Machine Learning ✨</p>", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/student-male.png", width=180)
    st.title("📊 Model Info")
    
    if st.session_state.model_loaded:
        st.success("✅ Model Ready")
        
        # Display model type
        st.markdown("### 🤖 Model Type")
        st.info("**Logistic Regression** (Classification)")
        
        if metrics:
            st.markdown("### 📈 Performance")
            if 'accuracy' in metrics:
                acc_pct = metrics['accuracy'] * 100
                st.metric("Accuracy", f"{acc_pct:.1f}%")
        
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
    Aplikasi ini memprediksi **Status Kelulusan** siswa berdasarkan:
    - Kehadiran (%)
    - Nilai Internal 1 & 2 (dari 40)
    - Skor Tugas (dari 10)
    - Nilai Ujian Akhir (dari 100)
    
    **Output:**
    - 🎉 **PASS** (Lulus)
    - ❌ **FAIL** (Tidak Lulus)
    """)

# =========================================
# MAIN CONTENT
# =========================================

if not st.session_state.model_loaded:
    st.error("❌ Model atau Scaler tidak dapat dimuat!")
    st.info("💡 Pastikan file 'model.pkl' dan 'scaler.pkl' ada di folder yang sama dengan app.py")
    st.stop()

# Success message
st.markdown("""
<div class='success-box'>
    ✅ <strong>Model siap digunakan!</strong> Masukkan data siswa untuk memprediksi status kelulusan.
</div>
""", unsafe_allow_html=True)

# Model info
if metrics:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">🤖 Model Type</h3>
            <p style="font-size: 1.5em; font-weight: bold; color: #333; margin: 10px 0;">Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'accuracy' in metrics:
            acc_val = metrics['accuracy'] * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">🎯 Accuracy</h3>
                <p style="font-size: 2em; font-weight: bold; color: #333; margin: 10px 0;">{acc_val:.1f}%</p>
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
        if 'Attendance' in feature:
            value = st.number_input(
                f"📊 {label}",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
                key=feature,
                help="Persentase kehadiran siswa (0-100%)"
            )
        elif 'Internal Test 1' in feature or 'Internal Test 2' in feature:
            value = st.number_input(
                f"📚 {label}",
                min_value=0.0,
                max_value=40.0,
                value=30.0,
                step=1.0,
                key=feature,
                help="Nilai ujian internal (0-40)"
            )
        elif 'Assignment Score' in feature:
            value = st.number_input(
                f"📝 {label}",
                min_value=0.0,
                max_value=10.0,
                value=8.0,
                step=0.5,
                key=feature,
                help="Skor tugas (0-10)"
            )
        elif 'Final Exam' in feature:
            value = st.number_input(
                f"📖 {label}",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                key=feature,
                help="Nilai ujian akhir (0-100)"
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
    predict_button = st.button("🎯 PREDIKSI STATUS KELULUSAN", use_container_width=True)

# =========================================
# PREDICTION
# =========================================
if predict_button:
    try:
        # Prepare input as DataFrame
        input_values = [inputs[feat] for feat in feature_names]
        input_df = pd.DataFrame([input_values], columns=feature_names)
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Predict class and probability
        prediction_class = model.predict(input_scaled)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(input_scaled)[0]
            pass_probability = prediction_proba[1] * 100  # Probability of class 1 (PASS)
            fail_probability = prediction_proba[0] * 100  # Probability of class 0 (FAIL)
        else:
            # If no predict_proba, use decision function
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(input_scaled)[0]
                # Convert to probability-like score (0-100)
                pass_probability = 1 / (1 + np.exp(-decision)) * 100
                fail_probability = 100 - pass_probability
            else:
                pass_probability = 100 if prediction_class == 1 else 0
                fail_probability = 100 if prediction_class == 0 else 0
        
        # Display prediction
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎊 Hasil Prediksi")
        
        # Determine result
        if prediction_class == 1:
            result_text = "PASS (LULUS) ✅"
            result_color = "#00C851"
            emoji = "🎉"
            message = "Selamat! Siswa diprediksi LULUS!"
        else:
            result_text = "FAIL (TIDAK LULUS) ❌"
            result_color = "#dd0404ef"
            emoji = "📚"
            message = "Siswa diprediksi TIDAK LULUS. Perlu usaha lebih keras!"
        
        st.markdown(f"""
        <div class="prediction-box">
            Status: {result_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Show probability
        st.markdown("<br>", unsafe_allow_html=True)
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #00C851; margin: 0;">✅ Probabilitas PASS</h3>
                <p style="font-size: 2.5em; font-weight: bold; color: #00C851; margin: 10px 0;">{pass_probability:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_prob2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #dd0404ef; margin: 0;">❌ Probabilitas FAIL</h3>
                <p style="font-size: 2.5em; font-weight: bold; color: #dd0404ef; margin: 10px 0;">{fail_probability:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Visualisasi Probabilitas")
        
        indicator_position = pass_probability
        st.markdown(f"""
        <div style="background: white; padding: 30px; border-radius: 20px; margin: 10px 0;">
            <div class="probability-bar">
                <div class="probability-indicator" style="left: {indicator_position}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; color: #333;">
                <span style="font-weight: bold;">❌ FAIL (0%)</span>
                <span style="font-weight: bold;">⚠️ BORDER (50%)</span>
                <span style="font-weight: bold;">✅ PASS (100%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Result box
        st.markdown(f"""
        <div style="background: {result_color}; padding: 30px; border-radius: 20px; 
        text-align: center; color: white; margin-top: 25px; box-shadow: 0 8px 20px rgba(0,0,0,0.25);">
            <div style="font-size: 40px; margin-bottom: 15px;">{emoji}</div>
            <div style="font-size: 26px; font-weight: bold; margin-bottom: 10px;">{result_text}</div>
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
            attendance_idx = next((i for i, f in enumerate(feature_names) if 'Attendance' in f), None)
            internal1_idx = next((i for i, f in enumerate(feature_names) if 'Internal Test 1' in f), None)
            internal2_idx = next((i for i, f in enumerate(feature_names) if 'Internal Test 2' in f), None)
            assignment_idx = next((i for i, f in enumerate(feature_names) if 'Assignment' in f), None)
            
            with col_a:
                if internal1_idx is not None and internal2_idx is not None:
                    # Calculate percentage (out of 40 each)
                    avg_internal = ((input_values[internal1_idx] + input_values[internal2_idx]) / 80) * 100
                    st.metric(
                        "Rata-rata Nilai Internal",
                        f"{avg_internal:.1f}%",
                        delta="Baik" if avg_internal >= 70 else "Perlu Ditingkatkan"
                    )
            
            with col_b:
                if attendance_idx is not None:
                    attendance = input_values[attendance_idx]
                    st.metric(
                        "Status Kehadiran",
                        f"{attendance:.0f}%",
                        delta="Baik" if attendance >= 80 else "Kurang"
                    )
            
            with col_c:
                if assignment_idx is not None:
                    assignment = input_values[assignment_idx]
                    assignment_pct = (assignment / 10) * 100
                    st.metric(
                        "Skor Tugas",
                        f"{assignment:.1f}/10",
                        delta="Baik" if assignment_pct >= 70 else "Kurang"
                    )
        
        # Visualization
        with st.expander("📈 Visualisasi Performa"):
            # Create radar chart
            fig = go.Figure()
            
            # Normalize values to 0-100 scale
            normalized_values = []
            labels = []
            
            for feat in feature_names:
                label = feature_labels.get(feat, feat.replace('_', ' ').title())
                value = inputs[feat]
                
                # Normalize to 100 scale based on feature type
                if 'Internal Test' in feat:
                    norm_value = (value / 40) * 100  # Out of 40
                elif 'Assignment' in feat:
                    norm_value = (value / 10) * 100  # Out of 10
                elif 'Attendance' in feat or 'Final Exam' in feat:
                    norm_value = value  # Already 0-100
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
        
        if prediction_class == 1:
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