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
    .viz-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin: 15px 0;
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
        data = joblib.load("academic_predictor_model_pt5.pkl")
        
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
# SIDEBAR - NAVIGATION
# =========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/student-male.png", width=180)
    st.title("📊 Navigation")
    
    # Navigation menu
    page = st.radio(
        "Select Page:",
        ["📈 Visualisasi Model", "🔮 Prediksi Satuan", "📂 Prediksi Batch (CSV)"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.title("ℹ️ Model Info")
    
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
# CHECK MODEL
# =========================================
if not st.session_state.model_loaded:
    st.error("❌ Model tidak dapat dimuat!")
    st.info("💡 Pastikan file 'academic_predictor_model.pkl' ada di folder yang sama dengan app.py")
    st.stop()

# =========================================
# PAGE 1: VISUALISASI MODEL
# =========================================
if page == "📈 Visualisasi Model":
    st.markdown("## 📈 Visualisasi & Analisis Model")
    
    # Model Performance Metrics
    st.markdown("<div class='success-box'>✅ <strong>Model Performance Overview</strong></div>", unsafe_allow_html=True)
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'test_r2' in metrics:
                r2_val = metrics['test_r2'] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin: 0;">🎯 R² Score</h3>
                    <p style="font-size: 2em; font-weight: bold; color: #333; margin: 10px 0;">{r2_val:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'mae' in metrics:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin: 0;">📊 MAE</h3>
                    <p style="font-size: 2em; font-weight: bold; color: #333; margin: 10px 0;">±{metrics['mae']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'rmse' in metrics:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin: 0;">📉 RMSE</h3>
                    <p style="font-size: 2em; font-weight: bold; color: #333; margin: 10px 0;">{metrics['rmse']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">📝 Features</h3>
                <p style="font-size: 2em; font-weight: bold; color: #333; margin: 10px 0;">{len(feature_names)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("### 🔍 Feature Importance")
    st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
    
    if model and hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': [feature_labels.get(f, f) for f in feature_names],
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        fig_importance = go.Figure()
        
        colors = ['#00C851' if c > 0 else '#ff4444' for c in coef_df['Coefficient']]
        
        fig_importance.add_trace(go.Bar(
            x=coef_df['Coefficient'],
            y=coef_df['Feature'],
            orientation='h',
            marker=dict(color=colors),
            text=coef_df['Coefficient'].round(2),
            textposition='outside'
        ))
        
        fig_importance.update_layout(
            title="Model Coefficients (Feature Importance)",
            xaxis_title="Coefficient Value",
            yaxis_title="Features",
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Interpretation
        st.markdown("#### 📊 Interpretasi:")
        st.markdown(f"""
        - **Fitur Paling Berpengaruh**: {coef_df.iloc[0]['Feature']} (Koefisien: {coef_df.iloc[0]['Coefficient']:.2f})
        - **Pengaruh Positif**: Hijau menunjukkan fitur yang meningkatkan final score
        - **Pengaruh Negatif**: Merah menunjukkan fitur yang menurunkan final score
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Formula
    st.markdown("### 📐 Model Formula")
    st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
    
    if model and hasattr(model, 'coef_'):
        formula = f"**Final Score = {model.intercept_:.2f}**"
        for feat, coef in zip(feature_names, model.coef_):
            label = feature_labels.get(feat, feat)
            sign = '+' if coef >= 0 else ''
            formula += f" {sign} {coef:.2f} × {label}"
        
        st.markdown(formula)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Distribution Analysis
    st.markdown("### 📊 Feature Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
        st.markdown("#### Expected Input Ranges")
        
        ranges_data = []
        for feat in feature_names:
            label = feature_labels.get(feat, feat)
            if 'Kehadiran' in feat:
                ranges_data.append({'Feature': label, 'Min': 0, 'Max': 100, 'Unit': '%'})
            elif 'Internal' in feat:
                ranges_data.append({'Feature': label, 'Min': 0, 'Max': 40, 'Unit': 'poin'})
            elif 'Tugas' in feat:
                ranges_data.append({'Feature': label, 'Min': 0, 'Max': 100, 'Unit': 'poin'})
            elif 'Jam' in feat:
                ranges_data.append({'Feature': label, 'Min': 0, 'Max': 12, 'Unit': 'jam'})
        
        ranges_df = pd.DataFrame(ranges_data)
        st.dataframe(ranges_df, use_container_width=True, hide_index=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
        st.markdown("#### Model Performance Category")
        
        if 'test_r2' in metrics:
            r2 = metrics['test_r2']
            
            if r2 >= 0.9:
                category = "⭐⭐⭐ Excellent"
                color = "#00C851"
            elif r2 >= 0.8:
                category = "⭐⭐ Very Good"
                color = "#33b5e5"
            elif r2 >= 0.7:
                category = "⭐ Good"
                color = "#ffbb33"
            else:
                category = "✅ Acceptable"
                color = "#ff8800"
            
            st.markdown(f"""
            <div style="background: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h2 style="margin: 0;">{category}</h2>
                <p style="margin: 10px 0 0 0;">Model dapat menjelaskan {r2*100:.1f}% variasi data</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# PAGE 2: PREDIKSI SATUAN
# =========================================
elif page == "🔮 Prediksi Satuan":
    st.markdown("## 🔮 Prediksi Individual")
    
    st.markdown("""
    <div class='success-box'>
        ✅ <strong>Input Data Siswa</strong> - Masukkan data untuk mendapatkan prediksi nilai akhir
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    st.markdown("### 📝 Input Data Siswa")
    
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
                    max_value=12.0,
                    value=5.0,
                    step=0.5,
                    key=feature,
                    help="Jam belajar per hari (max 12 jam)"
                )
            elif 'Internal_1' in feature or 'Internal_2' in feature:
                value = st.number_input(
                    f"📚 {label}",
                    min_value=0.0,
                    max_value=40.0,
                    value=30.0,
                    step=1.0,
                    key=feature,
                    help="Nilai Internal (0-40)"
                )
            elif 'Skor_Tugas' in feature:
                value = st.number_input(
                    f"📝 {label}",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=1.0,
                    key=feature,
                    help="Skor Tugas (0-100)"
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
    
    # Predict Button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🎯 PREDIKSI NILAI AKHIR", use_container_width=True)
    
    # Prediction
    if predict_button:
        try:
            input_values = [inputs[feat] for feat in feature_names]
            input_df = pd.DataFrame([input_values], columns=feature_names)
            
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
            
            prediction = model.predict(input_scaled)[0]
            prediction = np.clip(prediction, 0, 100)
            
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
                color = "#ff4444"
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
                
                kehadiran_idx = next((i for i, f in enumerate(feature_names) if 'Kehadiran' in f), None)
                nilai1_idx = next((i for i, f in enumerate(feature_names) if 'Internal_1' in f), None)
                nilai2_idx = next((i for i, f in enumerate(feature_names) if 'Internal_2' in f), None)
                jam_idx = next((i for i, f in enumerate(feature_names) if 'Jam' in f), None)
                
                with col_a:
                    if nilai1_idx is not None and nilai2_idx is not None:
                        avg_internal = (input_values[nilai1_idx] + input_values[nilai2_idx]) / 2
                        st.metric(
                            "Rata-rata Nilai Internal",
                            f"{avg_internal:.1f}/40",
                            delta="Baik" if avg_internal >= 28 else "Perlu Ditingkatkan"
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
                fig = go.Figure()
                
                normalized_values = []
                labels = []
                
                for feat in feature_names:
                    label = feature_labels.get(feat, feat.replace('_', ' ').title())
                    value = inputs[feat]
                    
                    if 'Jam' in feat:
                        norm_value = (value / 12) * 100
                    elif 'Internal' in feat:
                        norm_value = (value / 40) * 100
                    else:
                        norm_value = value
                    
                    normalized_values.append(norm_value)
                    labels.append(label)
                
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
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
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

# =========================================
# PAGE 3: PREDIKSI BATCH (CSV)
# =========================================
elif page == "📂 Prediksi Batch (CSV)":
    st.markdown("## 📂 Prediksi Batch dari File CSV")
    
    st.markdown("""
    <div class='info-box'>
        📋 <strong>Upload File CSV</strong> - File harus memiliki kolom sesuai dengan features model
    </div>
    """, unsafe_allow_html=True)
    
    # Required columns info
    st.markdown("### 📋 Format File CSV yang Dibutuhkan")
    st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
    
    st.markdown("**Kolom yang harus ada dalam CSV:**")
    required_cols = []
    for feat in feature_names:
        label = feature_labels.get(feat, feat)
        required_cols.append(f"- `{feat}` ({label})")
    
    st.markdown("\n".join(required_cols))
    
    # Download template
    template_data = {}
    for feat in feature_names:
        if 'Kehadiran' in feat:
            template_data[feat] = [85, 90, 75]
        elif 'Internal' in feat:
            template_data[feat] = [30, 35, 28]
        elif 'Tugas' in feat:
            template_data[feat] = [75, 80, 70]
        elif 'Jam' in feat:
            template_data[feat] = [5, 6, 4]
        else:
            template_data[feat] = [75, 80, 70]
    
    template_df = pd.DataFrame(template_data)
    
    st.download_button(
        label="📥 Download Template CSV",
        data=template_df.to_csv(index=False),
        file_name="template_input.csv",
        mime="text/csv",
        help="Download template CSV dengan format yang benar"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # File upload
    st.markdown("### 📤 Upload File CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            input_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File berhasil diupload! Total data: {len(input_data)} rows")
            
            # Display preview
            st.markdown("#### 👀 Preview Data (5 baris pertama)")
            st.dataframe(input_data.head(), use_container_width=True)
            
            # Check for required columns
            missing_cols = [col for col in feature_names if col not in input_data.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan dalam CSV: {', '.join(missing_cols)}")
                st.info("💡 Pastikan nama kolom sama persis dengan yang tertera di atas")
            else:
                st.success("✅ Semua kolom yang dibutuhkan tersedia!")
                
                # Predict button
                if st.button("🎯 PREDIKSI SEMUA DATA", use_container_width=True):
                    with st.spinner("⏳ Sedang memproses prediksi..."):
                        try:
                            # Extract features
                            X_input = input_data[feature_names]
                            
                            # Scale
                            if scaler is not None:
                                X_scaled = scaler.transform(X_input)
                            else:
                                X_scaled = X_input.values
                            
                            # Predict
                            predictions = model.predict(X_scaled)
                            predictions = np.clip(predictions, 0, 100)
                            
                            # Add predictions to dataframe
                            result_df = input_data.copy()
                            result_df['Predicted_Final_Score'] = predictions
                            
                            # Add category
                            def get_category(score):
                                if score >= 90:
                                    return "Excellent (A)"
                                elif score >= 80:
                                    return "Very Good (B+)"
                                elif score >= 70:
                                    return "Good (B)"
                                elif score >= 60:
                                    return "Average (C)"
                                else:
                                    return "Needs Improvement (D)"
                            
                            result_df['Category'] = result_df['Predicted_Final_Score'].apply(get_category)
                            
                            st.success(f"✅ Prediksi berhasil! Total {len(result_df)} data telah diprediksi")
                            
                            # Display results
                            st.markdown("### 📊 Hasil Prediksi")
                            st.dataframe(result_df, use_container_width=True, height=400)
                            
                            # Statistics
                            st.markdown("### 📈 Statistik Hasil Prediksi")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                avg_score = result_df['Predicted_Final_Score'].mean()
                                st.metric("Rata-rata Score", f"{avg_score:.2f}")
                            
                            with col2:
                                max_score = result_df['Predicted_Final_Score'].max()
                                st.metric("Score Tertinggi", f"{max_score:.2f}")
                            
                            with col3:
                                min_score = result_df['Predicted_Final_Score'].min()
                                st.metric("Score Terendah", f"{min_score:.2f}")
                            
                            with col4:
                                excellent_count = len(result_df[result_df['Predicted_Final_Score'] >= 90])
                                st.metric("Excellent (≥90)", f"{excellent_count}")
                            
                            # Distribution chart
                            st.markdown("### 📊 Distribusi Prediksi")
                            
                            col_chart1, col_chart2 = st.columns(2)
                            
                            with col_chart1:
                                # Histogram
                                fig_hist = px.histogram(
                                    result_df,
                                    x='Predicted_Final_Score',
                                    nbins=20,
                                    title='Distribusi Nilai Final Score',
                                    labels={'Predicted_Final_Score': 'Final Score'},
                                    color_discrete_sequence=['#667eea']
                                )
                                fig_hist.update_layout(
                                    showlegend=False,
                                    height=400,
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            with col_chart2:
                                # Category pie chart
                                category_counts = result_df['Category'].value_counts()
                                
                                fig_pie = px.pie(
                                    values=category_counts.values,
                                    names=category_counts.index,
                                    title='Distribusi Kategori Performa',
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                fig_pie.update_layout(height=400)
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Download results
                            st.markdown("### 💾 Download Hasil")
                            
                            csv = result_df.to_csv(index=False)
                            
                            col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
                            
                            with col_dl2:
                                st.download_button(
                                    label="📥 Download Hasil Prediksi (CSV)",
                                    data=csv,
                                    file_name="hasil_prediksi.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
            st.info("💡 Pastikan file CSV Anda valid dan sesuai format")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 30px;'>
    <p style='font-size: 16px; margin: 0 0 15px 0; font-weight: 600;'>
        Copyright © 2026 BY Pengelola MK Praktikum Unggulan (Praktikum DGX)
    </p>
    <div style='margin-top: 15px;'>
        <a href='https://www.praktikum-hpc.gunadarma.ac.id/' target='_blank' 
           style='color: white; text-decoration: none; margin: 0 10px; font-size: 14px; opacity: 0.9;'>
            🔗 praktikum-hpc.gunadarma.ac.id
        </a>
        <br>
        <a href='https://www.hpc-hub.gunadarma.ac.id/' target='_blank' 
           style='color: white; text-decoration: none; margin: 0 10px; font-size: 14px; opacity: 0.9;'>
            🔗 hpc-hub.gunadarma.ac.id
        </a>
    </div>
</div>
""", unsafe_allow_html=True)