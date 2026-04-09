"""
Cancer Risk Level Predictor - Main Application
================================================
Author  : Lokesh
Model   : Optuna-Tuned Class-Weighted XGBoost
Accuracy: 86% | High-Risk Recall: 45% (9/20 correct)
Deploy  : Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700; color: #1B4332;
        text-align: center; padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1.1rem; color: #52796F;
        text-align: center; margin-bottom: 2rem;
    }
    .insight-box {
        background: #F0FFF4; border-left: 5px solid #2D6A4F;
        padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0;
        font-size: 0.95rem; color: #1B4332;
    }
    .warning-box {
        background: #FFF5F5; border-left: 5px solid #D00000;
        padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0;
        font-size: 0.95rem; color: #8B0000;
    }
    .metric-row {
        display: flex; gap: 12px; margin: 12px 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stTabs [data-baseweb="tab"] {
        background-color: #D8F3DC; border-radius: 8px 8px 0 0;
        padding: 10px 16px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helper: Load Artifacts ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_feature_names():
    # Home page only needs feature count; avoid loading the full model during startup.
    import joblib
    return joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, 'data', 'cancer-risk-factors.csv'))

FEATURE_NAMES = load_feature_names()
df = load_data()

# ─── Sidebar Navigation ──────────────────────────────────────────
st.sidebar.markdown("## 🏥 Cancer Risk Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project by:** Lokesh 
**Model:** XGBoost (Class-Weighted)  
**Accuracy:** 86%  
**High-Risk Recall:** 45% (9/20)  
**Dataset:** 2,000 patients  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Pages")
st.sidebar.markdown("""
- **Home** → Overview & key stats  
- **EDA Dashboard** → 15+ visualizations  
- **Predict Risk** → Single & batch prediction  
- **Model Performance** → Metrics & pipeline  
""")

# ═══════════════════════════════════════════════════════════════════
#  HOME PAGE
# ═══════════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">🏥 Cancer Risk Level Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-Powered Risk Assessment using Demographic, Behavioral & Health Features</p>', unsafe_allow_html=True)

# ─── Key Metrics Row ──────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Patients", f"{len(df):,}")
m2.metric("Features", len(FEATURE_NAMES))
m3.metric("Model Accuracy", "86%")
m4.metric("High-Risk Recall", "45%")
m5.metric("Cancer Types", df['Cancer_Type'].nunique())

st.markdown("---")

# ─── Two Column Layout ────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("📌 Project Overview")
    st.markdown("""
    This project builds an end-to-end **Machine Learning Pipeline** to predict cancer risk 
    levels (**Low / Medium / High**) based on 16 features spanning demographics, lifestyle, 
    environmental exposure, and genetic markers.

    The final model — an **Optuna-Tuned Class-Weighted XGBoost** — was selected after 
    a rigorous **10-step iterative experimentation** process that addressed:
    
    - **Data Leakage Detection** — removed `Overall_Risk_Score` which was leaking target info  
    - **Class Imbalance** — Medium class dominates (78.7%), High is only 5.1%  
    - **High-Risk Recall Optimization** — the model correctly identifies 9 out of 20 high-risk patients  
    - **Hyperparameter Tuning** — 40 Optuna trials with 3-fold stratified CV  
    - **Current Final Output Insight** — class recalls are High: 0.45, Low: 0.77, Medium: 0.91  
    """)

    st.subheader("📊 Dataset at a Glance")
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | **Rows** | {len(df):,} patients |
    | **Columns** | {len(df.columns)} (16 features + 5 meta) |
    | **Cancer Types** | Lung (527), Breast (460), Colon (418), Prostate (305), Skin (290) |
    | **Age Range** | {df['Age'].min()} – {df['Age'].max()} years (mean: {df['Age'].mean():.1f}) |
    | **Gender Split** | Female: {len(df[df['Gender']==0]):,} ({len(df[df['Gender']==0])/len(df)*100:.1f}%) · Male: {len(df[df['Gender']==1]):,} ({len(df[df['Gender']==1])/len(df)*100:.1f}%) |
    | **Null Values** | 0 (clean dataset) |
    """)

with right:
    st.subheader("🎯 Risk Level Distribution")
    risk_counts = df['Risk_Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Level', 'Count']
    risk_counts['Percentage'] = (risk_counts['Count'] / risk_counts['Count'].sum() * 100).round(1)
    
    fig = px.pie(
        risk_counts, values='Count', names='Risk_Level',
        color='Risk_Level',
        color_discrete_map={'Low': '#2D6A4F', 'Medium': '#E85D04', 'High': '#D00000'},
        hole=0.45
    )
    fig.update_traces(textinfo='label+percent+value', textfont_size=13)
    fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>🔍 Key Insight:</b> The dataset is <b>heavily imbalanced</b> — Medium (78.7%) dominates, 
    Low (16.2%) is moderate, and High (5.1%) is the minority. This is clinically realistic 
    but requires special handling (SMOTE, class weighting) to avoid ignoring high-risk patients.
    </div>
    """, unsafe_allow_html=True)

# ─── Model Pipeline Summary ──────────────────────────────────────
st.markdown("---")
st.subheader("🔬 10-Step Model Development Pipeline")

pipeline_data = pd.DataFrame({
    'Step': ['1. Basic Models', '2. Leakage Fix', '3. Random Forest', '4. Optuna RF',
             '5. SMOTE + Optuna', '6. High-Risk Focus', '7. XGBoost + SMOTE',
             '8. Optuna + XGB (Recall)', '9. Class-Weighted XGB', '10. Final Model'],
    'Accuracy': ['99%*', '—', '84%', '83%', '83%', '80%', '85%', '61%', '64%', '86%'],
    'High-Risk Correct': ['—', '—', '7/20', '5/20', '9/20', '12/20', '9/20', '13/20', '15/20', '9/20'],
    'Key Action': [
        'Logistic + RF (data leakage detected)',
        'Removed Overall_Risk_Score column',
        'Baseline after fix — poor High recall',
        'Optuna hyperparameter tuning',
        'SMOTE for class balancing',
        'SMOTE + RF targeting High class',
        'Switched to XGBoost',
        'Optimized recall — accuracy dropped',
        'Class weights without SMOTE',
        'Optuna + class weights = best balance'
    ]
})
st.dataframe(pipeline_data, use_container_width=True, hide_index=True)

st.markdown("""
<div class="insight-box">
<b>🔍 Pipeline Insight:</b> Step 1 gave 99% accuracy — too good to be true! 
The <code>Overall_Risk_Score</code> column was directly derived from the target, causing 
<b>data leakage</b>. After removing it, the true challenge of predicting high-risk patients 
became apparent. The final model currently reports <b>86% overall accuracy</b> with
<b>45% High recall (9/20)</b>, <b>77% Low recall</b>, and <b>91% Medium recall</b>.
</div>
""", unsafe_allow_html=True)

# ─── Quick Dataset Preview ────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Preview Raw Dataset (first 10 rows)", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("📈 Feature Statistics Summary", expanded=False):
    st.dataframe(df.describe().round(2), use_container_width=True)

st.markdown("---")
st.caption("Navigate to **EDA Dashboard** for detailed analysis, **Predict Risk** for predictions, or **Model Performance** for evaluation metrics.")

