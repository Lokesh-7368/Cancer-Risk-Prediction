"""
Page 3: Model Performance & Evaluation
========================================
Complete model evaluation metrics, confusion matrix, and pipeline details.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Performance", page_icon="📋", layout="wide")

st.markdown("# 📋 Model Performance & Evaluation")
st.markdown("*Detailed metrics, pipeline history, and deployment information*")
st.markdown("---")

st.markdown("""
<div style="background:#FFF9E6; border-left:5px solid #E6A700; padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0;">
<b>🗣️ Presentation Note:</b> If you are explaining this to a teacher or panel, focus on <b>why</b> each model choice was made,
not only on final accuracy. Emphasize the trade-off between overall accuracy and minority-class (High-risk) recall.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════
st.subheader("1. Complete Model Comparison Table")
st.markdown("*All experiments conducted during the 10-step iterative process:*")

model_data = pd.DataFrame({
    'Model': [
        'Random Forest (Balanced, No Leakage)',
        'Optuna-Tuned RF (Macro F1 focus)',
        'Optuna-Tuned RF (High Recall focus)',
        'Baseline XGBoost (SMOTE)',
        'Class-weighted XGBoost',
        'Optuna-Tuned Class-weighted XGBoost '
    ],
    'Accuracy': [0.84, 0.83, 0.80, 0.85, 0.64, 0.86],
    'Macro F1': [0.64, 0.65, 0.65, 0.68, 0.54, 0.71],
    'Weighted F1': [0.83, 0.83, 0.81, 0.85, 0.67, 0.86],
    'Recall (High)': [0.35, 0.45, 0.60, 0.45, 0.75, 0.45],
    'Recall (Low)': [0.58, 0.63, 0.74, 0.68, 0.82, 0.77],
    'Recall (Medium)': [0.92, 0.89, 0.82, 0.92, 0.59, 0.91],
    'Key Notes': [
        'Basic RF with SMOTE; decent baseline',
        'Optuna tuning improved High recall slightly',
        'Strong recall improvement for minority classes',
        'Higher precision and overall accuracy than RF',
        'Boosted minority recall, but accuracy dropped',
        'Best current overall balance in latest verified run'
    ]
})

st.dataframe(model_data, use_container_width=True, hide_index=True)

st.markdown("""
<div style="background:#F0FFF4; border-left:5px solid #2D6A4F; padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0;">
<b>🔍 Model Selection Rationale:</b> The final <b>Optuna-Tuned Class-Weighted XGBoost</b> was chosen because it 
achieves the best <b>current balance</b> — 86% accuracy with 0.71 macro F1. While the Class-Weighted XGBoost (Step 9) 
had higher High-risk recall (0.75), its overall accuracy was only 64%. The final model provides 
<b>reliable predictions across ALL classes</b> without sacrificing too much on any single class.
</div>
""", unsafe_allow_html=True)

# ─── Visual Comparison ────────────────────────────────────────────
st.markdown("---")
st.subheader("2. Visual Model Comparison")

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(
        model_data, x='Model', y='Accuracy',
        color='Accuracy', color_continuous_scale='Greens',
        text=model_data['Accuracy'].apply(lambda x: f"{x:.0%}"),
        title="Accuracy Comparison Across Models"
    )
    fig.update_layout(height=400, xaxis_tickangle=-30, coloraxis_showscale=False)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    fig.add_trace(go.Bar(name='High', x=model_data['Model'], y=model_data['Recall (High)'],
                         marker_color='#D00000', text=model_data['Recall (High)'].apply(lambda x: f"{x:.0%}"),
                         textposition='outside'))
    fig.add_trace(go.Bar(name='Low', x=model_data['Model'], y=model_data['Recall (Low)'],
                         marker_color='#2D6A4F', text=model_data['Recall (Low)'].apply(lambda x: f"{x:.0%}"),
                         textposition='outside'))
    fig.update_layout(height=400, barmode='group', title="High vs Low Recall Comparison",
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="background:#F0FFF4; border-left:5px solid #2D6A4F; padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0;">
<b>🔍 Insight:</b> In healthcare risk systems, a model with slightly lower accuracy can be preferred if it catches more High-risk
patients. This is why recall on the High class is treated as a primary safety metric.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# FINAL MODEL DETAILS
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("3. Final Model: Optuna-Tuned Class-Weighted XGBoost")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Accuracy", "86%")
c2.metric("Macro F1 Score", "0.71")
c3.metric("Weighted F1", "0.86")
c4.metric("High-Risk Recall", "45%")

st.markdown("""
### Classification Report (Final Model)

| Class | Precision | Recall | F1-Score | Support |
|:------|----------:|-------:|---------:|--------:|
| **High** | 0.53 | 0.45 | 0.49 | 20 |
| **Low** | 0.71 | 0.77 | 0.74 | 65 |
| **Medium** | 0.92 | 0.91 | 0.91 | 315 |
| **Weighted Avg** | 0.86 | 0.86 | 0.86 | 400 |
""")

st.markdown("""
<div style="background:#F0FFF4; border-left:5px solid #2D6A4F; padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0;">
<b>🔍 Key Achievement:</b> The model correctly identifies <b>9 out of 20 high-risk patients</b> in the test set. 
In a medical context, this is crucial — missing a high-risk patient (false negative) is far more dangerous 
than a false alarm. The class-weighting approach penalizes the model more for misclassifying minority classes, 
resulting in better recall without SMOTE's synthetic data generation.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONFUSION MATRIX VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("4. Confusion Matrix (Illustrative)")

# Based on final model results from the report
cm = np.array([
    [9, 0, 11],
    [0, 50, 15],
    [8, 20, 287]
])
labels = ['High', 'Low', 'Medium']

fig = px.imshow(
    cm, text_auto=True, x=labels, y=labels,
    color_continuous_scale='Greens',
    labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'}
)
fig.update_layout(height=450, width=500, title="Confusion Matrix (Test Set)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="background:#F0FFF4; border-left:5px solid #2D6A4F; padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0;">
<b>🔍 Reading the Confusion Matrix:</b><br>
• <b>High Risk:</b> 9 correctly predicted, 11 misclassified as Medium.<br>
• <b>Low Risk:</b> 50 correct, 15 shifted to Medium.<br>
• <b>Medium Risk:</b> 287 correct, with spillover to both High and Low.<br>
• The biggest confusion pattern is movement toward <b>Medium</b>, indicating the model is conservative around minority extremes.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 10-STEP PIPELINE
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("5. Complete 10-Step Model Development Pipeline")

pipeline_steps = [
    ("1. Basic Models", "Built Logistic Regression and Random Forest as baselines. Got suspiciously high accuracy (~99%) — triggered investigation.", "🏗️"),
    ("2. Data Leakage Fix", "Discovered 'Overall_Risk_Score' was a composite directly derived from target variable. Removing it dropped accuracy to realistic levels.", "🔍"),
    ("3. Random Forest (No Leakage)", "Accuracy: 84%. But only 7/20 High-risk patients predicted correctly — unacceptable for medical use.", "🌲"),
    ("4. Hyperparameter Tuning (Optuna)", "Optuna with 50 trials for RF. Accuracy: 83%, High recall barely improved to 5/20. Tuning alone isn't enough.", "⚙️"),
    ("5. SMOTE + Optuna RF", "Applied SMOTE to balance classes. Accuracy: 83%, High recall improved to 9/20. Better but still insufficient.", "⚖️"),
    ("6. Focus on High-Risk (SMOTE + RF)", "Explicitly optimized for High recall. Accuracy: 80%, High recall: 12/20. Progress but overall accuracy suffered.", "🎯"),
    ("7. XGBoost + SMOTE", "Switched to XGBoost. Accuracy: 85%, High recall around 9/20 in evaluated runs. XGBoost is more precise overall.", "🚀"),
    ("8. Optuna + XGBoost (Recall Focus)", "Optimized recall for High class. High recall: 13/20 (great!), but overall accuracy dropped to 61%.", "📊"),
    ("9. Class-Weighted XGBoost (No SMOTE)", "Used inverse-frequency class weights instead of SMOTE. High recall: 15/20 (best!), but accuracy only 64%.", "⚖️"),
    ("10. Final: Optuna + Class-Weighted XGB", "Combined Optuna tuning with class weights. Accuracy: 86%, High recall: 9/20 (45%). Best current balance in latest run. ✅", "🏆")
]

for step, desc, icon in pipeline_steps:
    with st.expander(f"{icon} {step}"):
        st.write(desc)

# ═══════════════════════════════════════════════════════════════════
# TECHNICAL DETAILS
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("6. Technical Configuration")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    #### Hyperparameters (Optuna Best)
    | Parameter | Value |
    |:----------|:------|
    | n_estimators | ~300 |
    | max_depth | 6-8 |
    | learning_rate | ~0.05 |
    | subsample | ~0.8 |
    | colsample_bytree | ~0.7 |
    | gamma | ~1.5 |
    | reg_alpha | ~2.0 |
    | reg_lambda | ~3.0 |
    | eval_metric | mlogloss |
    | Optuna trials | 40 |
    | CV folds | 3 (Stratified) |
    """)

with c2:
    st.markdown("""
    #### Data Preprocessing
    | Step | Details |
    |:-----|:--------|
    | Removed Columns | Patient_ID, Cancer_Type, Overall_Risk_Score |
    | Target Encoding | LabelEncoder (High=0, Low=1, Medium=2) |
    | Train/Test Split | 80/20, stratified, random_state=42 |
    | Class Weighting | Inverse frequency (sample_weight) |
    | Feature Scaling | None needed (XGBoost is tree-based) |
    | Missing Values | 0 (dataset is clean) |
    """)

st.markdown("""
#### Tools & Technologies
| Category | Tools |
|:---------|:------|
| **Language** | Python 3.10+ |
| **ML Libraries** | scikit-learn, XGBoost, Optuna, imbalanced-learn (SMOTE) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git + GitHub |
| **Notebook** | Jupyter / Google Colab |
""")

st.markdown("""
### Viva-Ready Closing Summary
1. Business objective: identify high-risk patients early for proactive intervention.
2. Technical challenge: severe class imbalance and leakage risk.
3. Solution strategy: iterative experimentation, leakage removal, class weighting, and Optuna tuning.
4. Final value: strong overall performance with clinically meaningful High-risk detection.
""")
