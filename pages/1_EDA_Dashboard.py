"""
Page 1: Comprehensive EDA Dashboard
=====================================
15+ interactive visualizations with detailed insights at every point.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

# ─── Load Data ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, 'data', 'cancer-risk-factors.csv'))

df = load_data()
df['Gender_Label'] = df['Gender'].map({0: 'Female', 1: 'Male'})

RISK_COLORS = {'Low': '#2D6A4F', 'Medium': '#E85D04', 'High': '#D00000'}
GENDER_COLORS = {'Female': '#1B4332', 'Male': '#E85D04'}

def insight_box(text):
    st.markdown(f'<div style="background:#F0FFF4; border-left:5px solid #2D6A4F; '
                f'padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0; '
                f'font-size:0.95rem; color:#1B4332;"><b>🔍 Insight:</b> {text}</div>',
                unsafe_allow_html=True)

def warning_box(text):
    st.markdown(f'<div style="background:#FFF5F5; border-left:5px solid #D00000; '
                f'padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0; '
                f'font-size:0.95rem; color:#8B0000;"><b>⚠️ Finding:</b> {text}</div>',
                unsafe_allow_html=True)

# ─── Page Title ───────────────────────────────────────────────────
st.markdown("# 📊 Exploratory Data Analysis Dashboard")
st.markdown("*15+ interactive visualizations with detailed insights at each step*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# TAB STRUCTURE
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Distributions", "👥 Gender & Cancer", "⚠️ Risk Factors",
    "🔗 Correlations", "👴 Age Analysis", "🔬 Feature Deep-Dive",
    "📐 Statistical Summary"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("1️⃣ Risk Level Distribution — Class Imbalance Analysis")
    
    risk_counts = df['Risk_Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Level', 'Count']
    risk_counts['Pct'] = (risk_counts['Count'] / len(df) * 100).round(1)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(
            risk_counts, x='Risk_Level', y='Count', color='Risk_Level',
            color_discrete_map=RISK_COLORS, text=risk_counts.apply(
                lambda r: f"{r['Count']} ({r['Pct']}%)", axis=1
            )
        )
        fig.update_layout(height=420, showlegend=False,
                          xaxis_title="Risk Level", yaxis_title="Number of Patients")
        fig.update_traces(textposition='outside', textfont_size=14)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Class Breakdown")
        st.markdown(f"""
        | Risk Level | Count | Percentage |
        |:-----------|------:|-----------:|
        | **Medium** | {risk_counts[risk_counts['Risk_Level']=='Medium']['Count'].values[0]:,} | {risk_counts[risk_counts['Risk_Level']=='Medium']['Pct'].values[0]}% |
        | **Low** | {risk_counts[risk_counts['Risk_Level']=='Low']['Count'].values[0]:,} | {risk_counts[risk_counts['Risk_Level']=='Low']['Pct'].values[0]}% |
        | **High** | {risk_counts[risk_counts['Risk_Level']=='High']['Count'].values[0]:,} | {risk_counts[risk_counts['Risk_Level']=='High']['Pct'].values[0]}% |
        """)
        st.markdown(f"**Imbalance Ratio:** Medium is **{risk_counts[risk_counts['Risk_Level']=='Medium']['Count'].values[0] / risk_counts[risk_counts['Risk_Level']=='High']['Count'].values[0]:.1f}x** larger than High")
    
    insight_box(
        "The dataset is <b>heavily imbalanced</b> — Medium risk (78.7%) dominates while High risk "
        "is only 5.1% (102 patients out of 2,000). This 15.4:1 imbalance ratio between Medium and High "
        "means a naive model could achieve 78.7% accuracy by always predicting 'Medium'. "
        "This is why we need SMOTE or class-weighting to ensure the model learns to identify "
        "the critical High-risk minority class."
    )
    
    st.markdown("---")
    st.subheader("2️⃣ Cancer Type Distribution")
    
    cancer_counts = df['Cancer_Type'].value_counts().reset_index()
    cancer_counts.columns = ['Cancer_Type', 'Count']
    cancer_counts['Pct'] = (cancer_counts['Count'] / len(df) * 100).round(1)
    
    fig = px.bar(
        cancer_counts, x='Cancer_Type', y='Count',
        color='Cancer_Type',
        color_discrete_sequence=px.colors.qualitative.Set2,
        text=cancer_counts.apply(lambda r: f"{r['Count']} ({r['Pct']}%)", axis=1)
    )
    fig.update_layout(height=400, showlegend=False)
    fig.update_traces(textposition='outside', textfont_size=13)
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "<b>Lung cancer</b> is the most common type in the dataset (527 patients, 26.4%), "
        "followed by <b>Breast</b> (460, 23.0%), <b>Colon</b> (418, 20.9%), "
        "<b>Prostate</b> (305, 15.3%), and <b>Skin</b> (290, 14.5%). "
        "This distribution roughly mirrors real-world cancer prevalence patterns where "
        "lung and breast cancers are among the most frequently diagnosed worldwide."
    )
    
    st.markdown("---")
    st.subheader("3️⃣ Numerical Feature Distributions (Histograms)")
    
    num_features = ['Age', 'BMI', 'Smoking', 'Alcohol_Use', 'Obesity', 'Air_Pollution',
                    'Diet_Red_Meat', 'Diet_Salted_Processed', 'Fruit_Veg_Intake', 
                    'Physical_Activity', 'Occupational_Hazards', 'Calcium_Intake']
    
    fig = make_subplots(rows=3, cols=4, subplot_titles=num_features,
                        vertical_spacing=0.08, horizontal_spacing=0.06)
    for i, col_name in enumerate(num_features):
        row, col = divmod(i, 4)
        fig.add_trace(
            go.Histogram(x=df[col_name], name=col_name, marker_color='#40916C',
                         opacity=0.85, nbinsx=20),
            row=row+1, col=col+1
        )
    fig.update_layout(height=700, showlegend=False, title_text="Distribution of All 12 Numerical Features")
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "<b>Age</b> follows a roughly normal distribution centered around 63 years (range 25-90), "
        "with most patients between 50-75. <b>BMI</b> is also normally distributed around 26.2 "
        "(range 15-41.4). The <b>lifestyle/environmental factors</b> (Smoking, Alcohol, etc.) are "
        "on 0-10 index scales and show varied distributions — Smoking and Alcohol_Use are "
        "somewhat uniformly distributed, while Air_Pollution and Occupational_Hazards "
        "show slight right skew indicating more patients have moderate-to-high exposure."
    )

# ═══════════════════════════════════════════════════════════════════
# TAB 2: GENDER & CANCER TYPE
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("4️⃣ Cancer Types by Gender — Which Cancers Affect Each Gender?")
    
    gender_cancer = df.groupby(['Cancer_Type', 'Gender_Label']).size().reset_index(name='Count')
    fig = px.bar(
        gender_cancer, x='Cancer_Type', y='Count', color='Gender_Label',
        barmode='group', color_discrete_map=GENDER_COLORS,
        text='Count'
    )
    fig.update_layout(height=450, xaxis_title="Cancer Type", yaxis_title="Patient Count")
    fig.update_traces(textposition='outside', textfont_size=12)
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Female Cancer Distribution")
        female_df = df[df['Gender']==0]['Cancer_Type'].value_counts().reset_index()
        female_df.columns = ['Cancer_Type', 'Count']
        st.dataframe(female_df, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("#### Male Cancer Distribution")
        male_df = df[df['Gender']==1]['Cancer_Type'].value_counts().reset_index()
        male_df.columns = ['Cancer_Type', 'Count']
        st.dataframe(male_df, hide_index=True, use_container_width=True)
    
    insight_box(
        "<b>Females</b> are most affected by <b>Breast cancer (455 patients)</b>, which accounts for "
        "44.5% of all female cases. Other cancers they suffer from include Lung (238), Colon (197), "
        "and Skin (132). <b>Males</b> are predominantly affected by <b>Prostate cancer (305 patients)</b>, "
        "making up 31.2% of male cases, followed by Lung (289), Colon (221), and Skin (158). "
        "Notably, <b>5 males appear in Breast cancer</b> — this is rare but medically realistic, "
        "as male breast cancer accounts for ~1% of all breast cancer cases globally. "
        "<b>Prostate cancer is exclusive to males</b> (Gender=1), which is biologically correct."
    )
    
    st.markdown("---")
    st.subheader("5️⃣ Young Patients Analysis (Age < 30)")
    
    young = df[df['Age'] < 30]
    st.markdown(f"**Found {len(young)} patients under age 30:**")
    st.dataframe(young[['Patient_ID', 'Age', 'Gender_Label', 'Cancer_Type', 'Risk_Level',
                         'Smoking', 'Air_Pollution', 'BMI']].reset_index(drop=True),
                 use_container_width=True, hide_index=True)
    
    insight_box(
        f"Only <b>3 patients ({3/len(df)*100:.2f}%)</b> in the entire dataset are under 30 years old. "
        "All three are <b>females aged 25-29</b> suffering from <b>Lung cancer</b> with <b>Medium risk</b> level. "
        "This is a significant finding — lung cancer in young females could indicate "
        "environmental exposure (air pollution) or genetic predisposition rather than "
        "long-term smoking. The extremely low count of young patients reflects the real-world "
        "trend where cancer predominantly affects older adults (most patients are 50-80+)."
    )
    
    st.markdown("---")
    st.subheader("6️⃣ Gender Distribution Across Risk Levels")
    
    gender_risk = df.groupby(['Risk_Level', 'Gender_Label']).size().reset_index(name='Count')
    fig = px.bar(
        gender_risk, x='Risk_Level', y='Count', color='Gender_Label',
        barmode='group', color_discrete_map=GENDER_COLORS, text='Count'
    )
    fig.update_layout(height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "Gender distribution is <b>fairly balanced across all risk levels</b>. "
        "In High risk: 50 females vs 52 males. In Medium: 799 females vs 775 males. "
        "In Low: 173 females vs 151 males. This suggests <b>gender alone is not a strong "
        "predictor</b> of cancer risk level — lifestyle and environmental factors play a bigger role."
    )

# ═══════════════════════════════════════════════════════════════════
# TAB 3: RISK FACTORS
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("7️⃣ Top Factors Contributing to High Risk — Mean Difference Analysis")
    st.markdown("*Comparing average values of each feature between High-risk and Low-risk patients*")
    
    features_to_compare = [c for c in df.columns if c not in 
                           ['Patient_ID', 'Cancer_Type', 'Risk_Level', 'Gender_Label']]
    high_mean = df[df['Risk_Level'] == 'High'][features_to_compare].mean()
    low_mean = df[df['Risk_Level'] == 'Low'][features_to_compare].mean()
    diff = (high_mean - low_mean).sort_values(ascending=True)
    
    colors = ['#D00000' if v < 0 else '#2D6A4F' for v in diff.values]
    fig = go.Figure(go.Bar(
        x=diff.values, y=diff.index, orientation='h',
        marker_color=colors,
        text=[f"{v:+.2f}" for v in diff.values],
        textposition='outside'
    ))
    fig.update_layout(
        height=650, title="Mean Feature Difference (High Risk − Low Risk)",
        xaxis_title="Difference (positive = higher in High-risk)",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "<b>Strongest positive contributors to High risk (harmful factors):</b><br>"
        "• <b>Air Pollution</b> (+5.59): The STRONGEST differentiator — High-risk patients have "
        "mean Air_Pollution of 8.54 vs 2.95 for Low-risk (almost 3x higher!)<br>"
        "• <b>Smoking</b> (+4.67): High-risk mean 7.52 vs Low-risk 2.85 — smoking nearly triples risk<br>"
        "• <b>Alcohol Use</b> (+4.54): Similar pattern — 7.52 vs 2.98<br>"
        "• <b>Diet Salted/Processed</b> (+4.15): Processed food intake is 7.04 vs 2.89<br>"
        "• <b>Occupational Hazards</b> (+3.63): Workplace exposure at 6.99 vs 3.36<br>"
        "• <b>Diet Red Meat</b> (+3.39): Red meat consumption at 7.36 vs 3.97<br>"
        "• <b>Obesity</b> (+2.45): Obesity index at 7.27 vs 4.82<br><br>"
        "<b>Protective factors (negative = lower in High-risk):</b><br>"
        "• <b>Fruit & Vegetable Intake</b> (-1.76): High-risk patients eat LESS fruits/veggies (3.91 vs 5.67)<br>"
        "• <b>Physical Activity</b> (+1.25): Surprisingly slightly higher in High-risk — "
        "possibly because already-sick patients may exercise more as treatment advice"
    )
    
    st.markdown("---")
    st.subheader("8️⃣ Risk Factor Comparison — Grouped Bar Chart")
    
    risk_means = df.groupby('Risk_Level')[['Smoking', 'Alcohol_Use', 'Air_Pollution',
        'Diet_Salted_Processed', 'Occupational_Hazards', 'Diet_Red_Meat',
        'Obesity', 'Fruit_Veg_Intake']].mean().round(2)
    risk_means_long = risk_means.reset_index().melt(id_vars='Risk_Level', var_name='Factor', value_name='Mean')
    
    fig = px.bar(
        risk_means_long, x='Factor', y='Mean', color='Risk_Level',
        barmode='group', color_discrete_map=RISK_COLORS,
        text=risk_means_long['Mean'].round(1)
    )
    fig.update_layout(height=500, xaxis_tickangle=-30)
    fig.update_traces(textposition='outside', textfont_size=10)
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "This grouped comparison makes the pattern crystal clear: <b>every harmful factor "
        "increases monotonically from Low → Medium → High risk</b>. For example, "
        "Smoking goes from 2.85 (Low) → 5.48 (Medium) → 7.52 (High). "
        "Meanwhile, the protective factor <b>Fruit_Veg_Intake decreases</b>: "
        "5.67 (Low) → 4.84 (Medium) → 3.91 (High). This consistent gradient confirms "
        "these features are genuine predictors of cancer risk."
    )
    
    st.markdown("---")
    st.subheader("9️⃣ BMI vs Risk Level — Does BMI Matter?")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.box(
            df, x='Risk_Level', y='BMI', color='Risk_Level',
            color_discrete_map=RISK_COLORS, points='outliers'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### BMI Statistics by Risk Level")
        bmi_stats = df.groupby('Risk_Level')['BMI'].agg(['mean', 'median', 'std']).round(2)
        bmi_stats.columns = ['Mean', 'Median', 'Std Dev']
        st.dataframe(bmi_stats, use_container_width=True)
    
    warning_box(
        "<b>BMI does NOT strongly differentiate between risk levels.</b> "
        "Mean BMI is nearly identical across groups: High (26.23), Medium (26.26), Low (25.81). "
        "The distributions overlap almost completely. The slight difference for Low-risk (0.4 units lower) "
        "is statistically negligible given the standard deviation of ~3.95. "
        "This means <b>BMI alone is not a reliable indicator of cancer risk level</b> in this dataset — "
        "lifestyle and environmental exposures are far more important."
    )
    
    st.markdown("---")
    st.subheader("🔟 Obesity Index vs Risk Level (0-10 Scale)")
    
    fig = px.violin(
        df, x='Risk_Level', y='Obesity', color='Risk_Level',
        color_discrete_map=RISK_COLORS, box=True, points='all'
    )
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    high_obesity_pct = len(df[(df['Risk_Level']=='High') & (df['Obesity']>=7)]) / len(df[df['Risk_Level']=='High']) * 100
    low_obesity_pct = len(df[(df['Risk_Level']=='Low') & (df['Obesity']>=7)]) / len(df[df['Risk_Level']=='Low']) * 100
    
    insight_box(
        f"While BMI doesn't differentiate risk, the <b>Obesity index (0-10 scale) does show a clear pattern</b>. "
        f"<b>{high_obesity_pct:.1f}%</b> of High-risk patients have obesity index ≥ 7, compared to only "
        f"<b>{low_obesity_pct:.1f}%</b> of Low-risk patients — nearly <b>double</b> the rate. "
        "The median Obesity for High-risk is 8 vs 5 for Low-risk. This distinction between BMI "
        "(a raw measurement) and Obesity (a severity index) suggests the model benefits more "
        "from the processed indicator than the raw BMI value."
    )

# ═══════════════════════════════════════════════════════════════════
# TAB 4: CORRELATIONS
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("1️⃣1️⃣ Feature Correlation Heatmap")
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)
    
    fig = px.imshow(
        corr, text_auto='.2f', color_continuous_scale='RdYlGn_r',
        aspect='auto', zmin=-1, zmax=1
    )
    fig.update_layout(height=700, width=900, title="Pearson Correlation Matrix — All Numerical Features")
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "<b>Key correlation findings:</b><br>"
        "• <b>Overall_Risk_Score</b> is strongly correlated with Smoking (0.71), Alcohol_Use (0.64), "
        "Air_Pollution (0.61), Diet_Salted_Processed (0.57) — confirming it's a composite of these features "
        "and thus causes <b>data leakage</b> if used as input<br>"
        "• <b>Smoking ↔ Alcohol_Use</b> (0.50): Moderate positive correlation — these risk behaviors "
        "tend to co-occur<br>"
        "• <b>Diet_Red_Meat ↔ Diet_Salted_Processed</b> (0.51): Unhealthy diet patterns cluster together<br>"
        "• <b>Fruit_Veg_Intake ↔ Physical_Activity</b> (0.52): Healthy behaviors also cluster together<br>"
        "• <b>Most features have low inter-correlation</b> (<0.3), meaning they provide relatively "
        "independent information — good for model training<br>"
        "• <b>BMI, Age, Gender</b> all have very low correlations with other features (<0.1)"
    )
    
    st.markdown("---")
    st.subheader("1️⃣2️⃣ Interactive Feature Scatter Plot")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        x_feat = st.selectbox("X-axis", ['Smoking', 'Air_Pollution', 'Alcohol_Use', 'Obesity',
            'Diet_Red_Meat', 'Age', 'BMI', 'Fruit_Veg_Intake', 'Physical_Activity'], index=0)
    with c2:
        y_feat = st.selectbox("Y-axis", ['Air_Pollution', 'Smoking', 'Alcohol_Use', 'Obesity',
            'Diet_Red_Meat', 'Age', 'BMI', 'Fruit_Veg_Intake', 'Physical_Activity'], index=0)
    with c3:
        size_feat = st.selectbox("Bubble size", ['Obesity', 'Smoking', 'BMI', 'Age', 'Alcohol_Use'], index=0)
    
    fig = px.scatter(
        df, x=x_feat, y=y_feat, color='Risk_Level', size=size_feat,
        color_discrete_map=RISK_COLORS, opacity=0.6,
        hover_data=['Age', 'BMI', 'Cancer_Type', 'Gender_Label'],
        title=f"{x_feat} vs {y_feat} (sized by {size_feat})"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "Use the interactive scatter plot to explore relationships between any two features. "
        "The <b>red dots (High-risk)</b> tend to cluster in the top-right area for harmful factors — "
        "high Smoking AND high Air_Pollution together create the highest risk. "
        "The <b>green dots (Low-risk)</b> concentrate in the bottom-left, confirming that "
        "patients with low exposure across multiple factors have the lowest risk."
    )

# ═══════════════════════════════════════════════════════════════════
# TAB 5: AGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("1️⃣3️⃣ Age Distribution by Risk Level")
    
    fig = px.histogram(
        df, x='Age', color='Risk_Level', barmode='overlay', opacity=0.7,
        color_discrete_map=RISK_COLORS, nbins=30,
        marginal='box'
    )
    fig.update_layout(height=500, title="Age Distribution Across Risk Levels (with box plot)")
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "<b>Age does NOT significantly differentiate risk levels.</b> "
        "Mean age is nearly identical: High (63.75), Low (63.69), Medium (63.13). "
        "All three distributions overlap heavily, peaking around 60-70 years. "
        "This means a 70-year-old and a 50-year-old have similar cancer risk profiles — "
        "what matters more is their <b>lifestyle and exposure levels</b>, not age alone."
    )
    
    st.markdown("---")
    st.subheader("1️⃣4️⃣ Age Distribution by Cancer Type")
    
    fig = px.box(
        df, x='Cancer_Type', y='Age', color='Cancer_Type',
        color_discrete_sequence=px.colors.qualitative.Set2,
        points='outliers'
    )
    fig.update_layout(height=450, showlegend=False, title="Age Ranges for Each Cancer Type")
    st.plotly_chart(fig, use_container_width=True)
    
    age_by_cancer = df.groupby('Cancer_Type')['Age'].agg(['mean', 'min', 'max', 'median']).round(1)
    st.dataframe(age_by_cancer, use_container_width=True)
    
    insight_box(
        "All cancer types share similar age distributions (mean ~63, median ~64), "
        "which makes sense since this dataset doesn't differentiate by age-specific cancer incidence. "
        "The youngest patients (age 25) appear in Lung cancer (3 female patients). "
        "Prostate cancer spans ages 32-90, with the widest IQR. "
        "Breast cancer has a slight left skew (more patients in 50-65 range)."
    )
    
    st.markdown("---")
    st.subheader("1️⃣5️⃣ Risk Level Distribution by Age Group")
    
    bins = [20, 40, 50, 60, 70, 80, 95]
    labels = ['25-39', '40-49', '50-59', '60-69', '70-79', '80+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    age_risk = pd.crosstab(df['Age_Group'], df['Risk_Level'], normalize='index').round(3) * 100
    age_risk = age_risk.reset_index().melt(id_vars='Age_Group', var_name='Risk_Level', value_name='Percentage')
    
    fig = px.bar(
        age_risk, x='Age_Group', y='Percentage', color='Risk_Level',
        barmode='stack', color_discrete_map=RISK_COLORS,
        text=age_risk['Percentage'].round(1),
        title="Percentage of Each Risk Level Within Age Groups"
    )
    fig.update_layout(height=450, xaxis_title="Age Group", yaxis_title="Percentage (%)")
    fig.update_traces(textposition='inside', textfont_size=10)
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "Surprisingly, the <b>25-39 age group has 0% High-risk patients</b> (only 40 patients in this group). "
        "The <b>40-49 group</b> has 5.2% High-risk, <b>60-69</b> has 5.8% — the highest proportion. "
        "However, these differences are small and not statistically significant given the small High-risk "
        "sample size. The key takeaway: <b>risk level is driven more by lifestyle factors than by age alone</b>."
    )
    
    st.markdown("---")
    st.subheader("1️⃣6️⃣ Average Age by Risk Level × Gender")
    
    age_gender_risk = df.groupby(['Risk_Level', 'Gender_Label'])['Age'].mean().reset_index()
    fig = px.bar(
        age_gender_risk, x='Risk_Level', y='Age', color='Gender_Label',
        barmode='group', color_discrete_map=GENDER_COLORS,
        text=age_gender_risk['Age'].round(1),
        title="Average Patient Age by Risk Level and Gender"
    )
    fig.update_layout(height=400, yaxis_range=[55, 68])
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "Average age is remarkably consistent across all combinations of risk level and gender: "
        "ranging from ~62 to ~65 years. There's no meaningful age-gender interaction for risk prediction. "
        "This reinforces that <b>cancer risk prediction in this dataset depends primarily on "
        "environmental/behavioral factors</b>, not demographics."
    )

# ═══════════════════════════════════════════════════════════════════
# TAB 6: FEATURE DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("1️⃣7️⃣ Interactive Feature Explorer — Violin + Box Plots")
    
    feature_choice = st.selectbox(
        "Select a feature to explore in detail:",
        ['Smoking', 'Air_Pollution', 'Alcohol_Use', 'Diet_Salted_Processed',
         'Occupational_Hazards', 'Diet_Red_Meat', 'Obesity', 'Fruit_Veg_Intake',
         'Physical_Activity', 'Calcium_Intake', 'BMI', 'Age',
         'Physical_Activity_Level', 'Overall_Risk_Score']
    )
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.violin(
            df, x='Risk_Level', y=feature_choice, color='Risk_Level',
            color_discrete_map=RISK_COLORS, box=True, points='all',
            title=f"{feature_choice} — Violin Plot by Risk Level"
        )
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            df, x=feature_choice, color='Risk_Level', barmode='overlay',
            color_discrete_map=RISK_COLORS, opacity=0.7, nbins=20,
            title=f"{feature_choice} — Overlapping Histograms"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # Stats table
    stats = df.groupby('Risk_Level')[feature_choice].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    stats.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
    st.markdown(f"#### {feature_choice} — Statistics by Risk Level")
    st.dataframe(stats, use_container_width=True)
    
    # Dynamic insight
    h_mean = df[df['Risk_Level']=='High'][feature_choice].mean()
    l_mean = df[df['Risk_Level']=='Low'][feature_choice].mean()
    diff_val = h_mean - l_mean
    if abs(diff_val) > 1:
        if diff_val > 0:
            insight_box(f"<b>{feature_choice}</b> is significantly <b>higher</b> in High-risk patients "
                       f"(mean {h_mean:.2f}) compared to Low-risk ({l_mean:.2f}), a difference of "
                       f"<b>+{diff_val:.2f}</b>. This indicates it's a <b>risk-increasing factor</b>.")
        else:
            insight_box(f"<b>{feature_choice}</b> is significantly <b>lower</b> in High-risk patients "
                       f"(mean {h_mean:.2f}) compared to Low-risk ({l_mean:.2f}), a difference of "
                       f"<b>{diff_val:.2f}</b>. This indicates it's a <b>protective factor</b>.")
    else:
        warning_box(f"<b>{feature_choice}</b> shows <b>minimal difference</b> between High-risk "
                   f"(mean {h_mean:.2f}) and Low-risk ({l_mean:.2f}) — difference of only {diff_val:+.2f}. "
                   "This feature is <b>not a strong predictor</b> of risk level on its own.")
    
    st.markdown("---")
    st.subheader("1️⃣8️⃣ Smoking vs Alcohol Use — Dual Risk Behaviors")
    
    fig = px.scatter(
        df, x='Smoking', y='Alcohol_Use', color='Risk_Level',
        size='Obesity', opacity=0.6, color_discrete_map=RISK_COLORS,
        hover_data=['Age', 'BMI', 'Cancer_Type', 'Air_Pollution'],
        title="Smoking vs Alcohol Use (bubble size = Obesity level)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "<b>Dual risk behavior pattern:</b> Patients with BOTH high Smoking (7+) AND high Alcohol_Use (7+) "
        "are overwhelmingly classified as High-risk (red dots in top-right corner). "
        "Conversely, patients with low levels of both (<3 on each) are predominantly Low-risk (green dots in bottom-left). "
        "This confirms that <b>risk factors are synergistic</b> — the combination of multiple unhealthy behaviors "
        "amplifies cancer risk far more than any single factor alone."
    )
    
    st.markdown("---")
    st.subheader("1️⃣9️⃣ Physical Activity vs Air Pollution — Lifestyle vs Environment")
    
    fig = px.density_heatmap(
        df, x='Physical_Activity', y='Air_Pollution', facet_col='Risk_Level',
        color_continuous_scale='Greens', nbinsx=10, nbinsy=10,
        title="Physical Activity vs Air Pollution (by Risk Level)"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "For <b>Low-risk</b> patients, the density concentrates at low Air_Pollution (1-4) across "
        "various Physical_Activity levels. For <b>High-risk</b> patients, the density shifts to "
        "high Air_Pollution (7-10) regardless of physical activity. This suggests that "
        "<b>environmental exposure (Air Pollution) may override the protective effect of physical activity</b> — "
        "even active patients in heavily polluted areas face elevated cancer risk."
    )
    
    st.markdown("---")
    st.subheader("2️⃣0️⃣ Genetic/Medical Flags — Family History, BRCA, H.Pylori")
    
    genetic_features = ['Family_History', 'BRCA_Mutation', 'H_Pylori_Infection']
    genetic_data = []
    for feat in genetic_features:
        for risk in ['Low', 'Medium', 'High']:
            subset = df[df['Risk_Level'] == risk]
            positive = subset[feat].sum()
            total = len(subset)
            genetic_data.append({
                'Feature': feat, 'Risk_Level': risk,
                'Positive_Count': positive, 'Total': total,
                'Percentage': round(positive / total * 100, 1)
            })
    genetic_df = pd.DataFrame(genetic_data)
    
    fig = px.bar(
        genetic_df, x='Feature', y='Percentage', color='Risk_Level',
        barmode='group', color_discrete_map=RISK_COLORS,
        text='Percentage', title="Percentage of Patients with Genetic/Medical Flags by Risk Level"
    )
    fig.update_layout(height=400, yaxis_title="% of Patients")
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "<b>Genetic/Medical flags show weak differentiation:</b><br>"
        "• <b>Family History:</b> 20.6% (High) vs 13.9% (Low) — slight increase but not dramatic<br>"
        "• <b>BRCA Mutation:</b> Only 4.9% (High) vs 3.7% (Low) — very rare across all groups<br>"
        "• <b>H. Pylori Infection:</b> 25.5% (High) vs 17.6% (Low) — the largest gap among genetic flags<br><br>"
        "These binary flags are much <b>weaker predictors</b> compared to the continuous lifestyle/environmental "
        "features. The model relies more heavily on Smoking, Air_Pollution, and Diet patterns."
    )
    
    st.markdown("---")
    st.subheader("2️⃣1️⃣ Cancer Type vs High-Risk Percentage")
    
    cancer_high_risk = []
    for ct in df['Cancer_Type'].unique():
        sub = df[df['Cancer_Type'] == ct]
        high_count = len(sub[sub['Risk_Level'] == 'High'])
        cancer_high_risk.append({
            'Cancer_Type': ct, 'Total': len(sub), 'High_Risk': high_count,
            'High_Risk_Pct': round(high_count / len(sub) * 100, 1)
        })
    chr_df = pd.DataFrame(cancer_high_risk).sort_values('High_Risk_Pct', ascending=True)
    
    fig = px.bar(
        chr_df, x='High_Risk_Pct', y='Cancer_Type', orientation='h',
        color='High_Risk_Pct', color_continuous_scale=['#2D6A4F', '#E85D04', '#D00000'],
        text=chr_df.apply(lambda r: f"{r['High_Risk_Pct']}% ({r['High_Risk']}/{r['Total']})", axis=1),
        title="Which Cancer Types Have the Highest Percentage of High-Risk Patients?"
    )
    fig.update_layout(height=350, coloraxis_showscale=False,
                      xaxis_title="% of High-Risk Patients")
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    insight_box(
        "<b>Lung cancer has the highest proportion of High-risk patients</b> at 8.9% (47 out of 527), "
        "followed by Colon at 6.9% (29/418). Prostate cancer has the <b>lowest</b> High-risk percentage "
        "at only 1.0% (3/305). Breast cancer is at 2.8% (13/460). This aligns with medical knowledge — "
        "lung cancer is often associated with smoking and air pollution (the two strongest risk factors "
        "in our analysis), while prostate cancer has a more age-related progression."
    )

# ═══════════════════════════════════════════════════════════════════
# TAB 7: STATISTICAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("2️⃣2️⃣ Complete Statistical Summary by Risk Level")
    
    numeric_cols = ['Age', 'BMI', 'Smoking', 'Alcohol_Use', 'Obesity', 'Family_History',
                    'Diet_Red_Meat', 'Diet_Salted_Processed', 'Fruit_Veg_Intake',
                    'Physical_Activity', 'Air_Pollution', 'Occupational_Hazards',
                    'BRCA_Mutation', 'H_Pylori_Infection', 'Calcium_Intake',
                    'Physical_Activity_Level']
    
    summary = df.groupby('Risk_Level')[numeric_cols].mean().round(2).T
    summary = summary[['Low', 'Medium', 'High']]
    summary['High−Low Diff'] = (summary['High'] - summary['Low']).round(2)
    summary['Strong Predictor?'] = summary['High−Low Diff'].apply(
        lambda x: '✅ Yes (Positive)' if x > 1.5 else ('✅ Yes (Protective)' if x < -1.5 else '❌ Weak')
    )
    
    st.dataframe(summary, use_container_width=True, height=600)
    
    insight_box(
        "<b>Summary of predictive power for each feature:</b><br>"
        "• <b>Strong positive risk factors (diff > 1.5):</b> Air_Pollution, Smoking, Alcohol_Use, "
        "Diet_Salted_Processed, Occupational_Hazards, Diet_Red_Meat, Obesity<br>"
        "• <b>Protective factors (diff < -1.5):</b> Fruit_Veg_Intake<br>"
        "• <b>Weak predictors (diff close to 0):</b> Age, BMI, Gender, BRCA_Mutation, "
        "Family_History, Physical_Activity_Level, Calcium_Intake, H_Pylori_Infection<br><br>"
        "The model's feature importance aligns with this analysis — "
        "<b>7 lifestyle/environmental features carry most of the predictive weight</b>, while "
        "demographic and genetic features contribute minimally."
    )
    
    st.markdown("---")
    st.subheader("2️⃣3️⃣ Protective Factors Analysis — Fruit/Veg Intake")
    
    fvi_data = []
    for risk in ['Low', 'Medium', 'High']:
        subset = df[df['Risk_Level'] == risk]
        high_fvi = len(subset[subset['Fruit_Veg_Intake'] >= 7])
        fvi_data.append({
            'Risk_Level': risk,
            'High_Fruit_Veg (≥7)': f"{high_fvi}/{len(subset)} ({high_fvi/len(subset)*100:.1f}%)",
            'Low_Fruit_Veg (<3)': f"{len(subset[subset['Fruit_Veg_Intake']<3])}/{len(subset)} ({len(subset[subset['Fruit_Veg_Intake']<3])/len(subset)*100:.1f}%)"
        })
    st.dataframe(pd.DataFrame(fvi_data), hide_index=True, use_container_width=True)
    
    insight_box(
        "<b>Fruit & Vegetable intake has a clear protective pattern:</b> "
        "40.1% of Low-risk patients have high intake (≥7) compared to only 24.5% of High-risk patients. "
        "Conversely, patients with very low intake (<3) are more concentrated in the High-risk group. "
        "This is consistent with medical research showing that fruits and vegetables contain "
        "antioxidants and phytochemicals that help prevent cellular damage and reduce cancer risk."
    )
    
    st.markdown("---")
    st.subheader("📋 Complete EDA Summary — Key Takeaways")
    st.markdown("""
    After analyzing 2,000 patients across 21 features, here are the definitive findings:
    
    **1. Class Imbalance is severe** — Medium (78.7%) >> Low (16.2%) >> High (5.1%). Requires SMOTE or class-weighting.
    
    **2. Top 7 risk-increasing factors** (in order of impact): Air Pollution, Smoking, Alcohol Use, 
    Diet Salted/Processed, Occupational Hazards, Diet Red Meat, Obesity.
    
    **3. Fruit & Vegetable Intake is the strongest protective factor** — higher intake correlates with lower risk.
    
    **4. BMI is NOT a strong predictor** — nearly identical distributions across risk levels.
    
    **5. Age and Gender are weak predictors** — risk is driven by lifestyle, not demographics.
    
    **6. Genetic flags (Family History, BRCA, H.Pylori) are weak** — binary flags with small percentage differences.
    
    **7. Lung cancer has the highest High-risk rate** (8.9%), Prostate the lowest (1.0%).
    
    **8. Risk factors are synergistic** — patients with multiple high-risk behaviors face dramatically elevated risk.
    
    **9. Only 3 patients under 30** — all female, Lung cancer, Medium risk.
    
    **10. Overall_Risk_Score causes data leakage** — it's derived from the target and must be excluded from features.
    """)
