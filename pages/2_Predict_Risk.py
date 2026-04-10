"""
Page 2: Cancer Risk Prediction
================================
Single patient prediction with manual input + Batch CSV prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Predict Risk", page_icon="🔮", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'final_xgb_class_weighted.pkl'))
    le = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))
    features = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))
    return model, le, features

model, le, FEATURE_NAMES = load_artifacts()

def insight_box(text):
    st.markdown(
        f'<div style="background:#F0FFF4; border-left:5px solid #2D6A4F; '
        f'padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0; '
        f'font-size:0.95rem; color:#1B4332;"><b>🔍 Insight:</b> {text}</div>',
        unsafe_allow_html=True
    )

def presentation_note(text):
    st.markdown(
        f'<div style="background:#FFF9E6; border-left:5px solid #E6A700; '
        f'padding:12px 16px; margin:10px 0; border-radius:0 8px 8px 0; '
        f'font-size:0.95rem; color:#6B5200;"><b>🗣️ Presentation Note:</b> {text}</div>',
        unsafe_allow_html=True
    )

st.markdown("# 🔮 Cancer Risk Level Prediction")
st.markdown("*Enter patient features to predict their cancer risk level (Low / Medium / High)*")
st.markdown("---")

insight_box(
    "This page demonstrates both clinical-style single-patient screening and operational "
    "batch triage. The model outputs probabilities for all classes, so decisions can be "
    "explained with confidence scores rather than only a hard label."
)

def preprocess(df_input):
    missing = [c for c in FEATURE_NAMES if c not in df_input.columns]
    if missing:
        st.warning(f"Missing columns filled with 0: {missing}")
        for c in missing:
            df_input[c] = 0
    df_input = df_input[FEATURE_NAMES].copy()
    df_input = df_input.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_input

# ─── Mode Selection ───────────────────────────────────────────────
mode = st.radio("**Prediction Mode:**", ["🧑‍⚕️ Single Patient (Manual)", "📁 Batch CSV Upload"],
                horizontal=True)

# ═══════════════════════════════════════════════════════════════════
# SINGLE PATIENT PREDICTION
# ═══════════════════════════════════════════════════════════════════
if mode == "🧑‍⚕️ Single Patient (Manual)":
    st.subheader("Enter Patient Details")
    presentation_note(
        "Explain that values are intentionally grouped by domain "
        "(demographic, lifestyle, environmental, and genetic) to mirror real clinical intake forms."
    )
    
    # Organized input in columns
    st.markdown("#### Demographics")
    d1, d2, d3 = st.columns(3)
    with d1:
        age = st.slider("Age (years)", 20, 95, 60, help="Patient's current age")
    with d2:
        gender = st.selectbox("Gender", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
    with d3:
        bmi = st.slider("BMI", 15.0, 45.0, 26.0, 0.1, help="Body Mass Index")
    
    st.markdown("#### Lifestyle & Environmental Factors (0-10 scale)")
    st.caption("0 = No exposure/activity, 10 = Maximum exposure/activity")
    
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        smoking = st.slider("🚬 Smoking", 0, 10, 5, help="Smoking intensity/frequency")
        diet_red = st.slider("🥩 Diet Red Meat", 0, 10, 5, help="Red meat consumption level")
    with l2:
        alcohol = st.slider("🍺 Alcohol Use", 0, 10, 5, help="Alcohol consumption level")
        diet_salt = st.slider("🧂 Diet Salted/Processed", 0, 10, 5, help="Processed food intake")
    with l3:
        obesity = st.slider("⚖️ Obesity", 0, 10, 5, help="Obesity severity level")
        fruit_veg = st.slider("🥬 Fruit & Veg Intake", 0, 10, 5, help="Fruit and vegetable consumption")
    with l4:
        air_pol = st.slider("🏭 Air Pollution", 0, 10, 5, help="Environmental air pollution exposure")
        occ_haz = st.slider("⚠️ Occupational Hazards", 0, 10, 5, help="Workplace exposure to harmful substances")
    
    l5, l6, l7, l8 = st.columns(4)
    with l5:
        phys_act = st.slider("🏃 Physical Activity", 0, 10, 5, help="Frequency/intensity of exercise")
    with l6:
        phys_lvl = st.slider("📊 Physical Activity Level", 0, 10, 5, help="Activity rating")
    with l7:
        calcium = st.slider("🦴 Calcium Intake", 0, 10, 5, help="Calcium in diet")
    with l8:
        pass  # spacer
    
    st.markdown("#### Genetic / Medical Flags")
    g1, g2, g3 = st.columns(3)
    with g1:
        family_hist = st.selectbox("👨‍👩‍👧 Family History", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    with g2:
        brca = st.selectbox("🧬 BRCA Mutation", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    with g3:
        h_pylori = st.selectbox("🦠 H. Pylori Infection", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    
    st.markdown("---")
    
    if st.button("🔮 **Predict Risk Level**", use_container_width=True, type="primary"):
        input_data = {
            'Age': age, 'Gender': gender[1], 'BMI': bmi,
            'Smoking': smoking, 'Alcohol_Use': alcohol, 'Obesity': obesity,
            'Family_History': family_hist[1], 'Diet_Red_Meat': diet_red,
            'Diet_Salted_Processed': diet_salt, 'Fruit_Veg_Intake': fruit_veg,
            'Physical_Activity': phys_act, 'Air_Pollution': air_pol,
            'Occupational_Hazards': occ_haz, 'BRCA_Mutation': brca[1],
            'H_Pylori_Infection': h_pylori[1], 'Calcium_Intake': calcium,
            'Physical_Activity_Level': phys_lvl
        }
        
        X = preprocess(pd.DataFrame([input_data]))
        pred_enc = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if pred_label == 'High':
                st.error(f"### ⚠️ Predicted Risk Level: **{pred_label}**")
                st.markdown("**Recommendation:** Immediate clinical follow-up recommended. "
                           "Multiple risk factors are elevated.")
            elif pred_label == 'Medium':
                st.warning(f"### ⚡ Predicted Risk Level: **{pred_label}**")
                st.markdown("**Recommendation:** Regular monitoring advised. Consider lifestyle modifications.")
            else:
                st.success(f"### ✅ Predicted Risk Level: **{pred_label}**")
                st.markdown("**Recommendation:** Continue healthy lifestyle. Regular screening recommended.")
        
        # Probability chart
        prob_df = pd.DataFrame({
            'Risk Level': le.classes_,
            'Probability': probs
        }).sort_values('Probability', ascending=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                prob_df, x='Probability', y='Risk Level', orientation='h',
                color='Risk Level',
                color_discrete_map={'Low': '#2D6A4F', 'Medium': '#E85D04', 'High': '#D00000'},
                text=prob_df['Probability'].apply(lambda x: f"{x:.1%}")
            )
            fig.update_layout(height=250, showlegend=False, title="Class Probabilities")
            fig.update_traces(textposition='outside')
            fig.update_xaxes(range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("#### Probability Breakdown")
            for _, row in prob_df.iterrows():
                cls = row['Risk Level']
                prob = row['Probability']
                bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                st.code(f"{cls:>8}: {bar} {prob:.1%}")

        insight_box(
            "Probability outputs are critical for medical interpretation. For example, a High class "
            "probability close to Medium indicates borderline risk and supports recommending follow-up "
            "tests instead of immediate binary decisions."
        )
        
        # Risk factor analysis for this patient
        st.markdown("---")
        st.subheader("🔍 Patient Risk Factor Profile")
        
        risk_factors = {
            'Smoking': smoking, 'Air_Pollution': air_pol, 'Alcohol_Use': alcohol,
            'Diet_Salted_Processed': diet_salt, 'Occupational_Hazards': occ_haz,
            'Diet_Red_Meat': diet_red, 'Obesity': obesity
        }
        protective_factors = {'Fruit_Veg_Intake': fruit_veg, 'Physical_Activity': phys_act}
        
        elevated = [k for k, v in risk_factors.items() if v >= 7]
        low_protective = [k for k, v in protective_factors.items() if v <= 3]
        
        if elevated:
            st.markdown(f"**⚠️ Elevated Risk Factors (≥7/10):** {', '.join(elevated)}")
        else:
            st.markdown("**✅ No critically elevated risk factors**")
        
        if low_protective:
            st.markdown(f"**⚠️ Low Protective Factors (≤3/10):** {', '.join(low_protective)}")
        else:
            st.markdown("**✅ Protective factors are adequate**")
        
        # Radar chart
        categories = list(risk_factors.keys()) + list(protective_factors.keys())
        values = [risk_factors[k] for k in risk_factors] + [protective_factors[k] for k in protective_factors]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(213, 0, 0, 0.2)' if pred_label == 'High' else 
                      'rgba(232, 93, 4, 0.2)' if pred_label == 'Medium' else
                      'rgba(45, 106, 79, 0.2)',
            line_color='#D00000' if pred_label == 'High' else '#E85D04' if pred_label == 'Medium' else '#2D6A4F'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            height=400, title="Patient Risk Factor Radar"
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# BATCH CSV PREDICTION
# ═══════════════════════════════════════════════════════════════════
else:
    st.subheader("Upload CSV for Batch Prediction")
    st.markdown(f"**Required columns:** `{', '.join(FEATURE_NAMES)}`")
    presentation_note(
        "This batch mode is useful for hospitals or screening camps where many records are scored "
        "at once. It supports prioritization by predicted risk and probability columns."
    )
    
    with st.expander("📋 Download Sample CSV Template"):
        sample = pd.DataFrame({feat: [0] for feat in FEATURE_NAMES})
        st.download_button(
            "⬇️ Download Template CSV",
            sample.to_csv(index=False),
            "cancer_risk_template.csv",
            "text/csv"
        )
    
    uploaded = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded:
        input_df = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded:** {len(input_df)} rows × {len(input_df.columns)} columns")
        
        with st.expander("Preview uploaded data"):
            st.dataframe(input_df.head(), use_container_width=True)
        
        if st.button("🔮 **Run Batch Prediction**", use_container_width=True, type="primary"):
            X = preprocess(input_df)
            preds_enc = model.predict(X)
            probs = model.predict_proba(X)
            preds = le.inverse_transform(preds_enc)
            
            result = X.copy()
            result['Predicted_Risk_Level'] = preds
            for i, cls in enumerate(le.classes_):
                result[f'Prob_{cls}'] = probs[:, i].round(4)
            
            st.success(f"✅ Predictions complete for {len(result)} patients!")
            
            # Summary
            pred_counts = pd.Series(preds).value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Low Risk", pred_counts.get('Low', 0))
            c2.metric("Medium Risk", pred_counts.get('Medium', 0))
            c3.metric("High Risk", pred_counts.get('High', 0))
            
            # Distribution chart
            fig = px.pie(
                values=pred_counts.values, names=pred_counts.index,
                color=pred_counts.index,
                color_discrete_map={'Low': '#2D6A4F', 'Medium': '#E85D04', 'High': '#D00000'},
                title="Prediction Distribution"
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            insight_box(
                "Batch summary helps quickly estimate care workload. A higher share of predicted High-risk "
                "patients means immediate resource planning for diagnostics and specialist review."
            )
            
            st.dataframe(result, use_container_width=True)
            
            st.download_button(
                "⬇️ Download Predictions CSV",
                result.to_csv(index=False),
                "cancer_risk_predictions.csv",
                "text/csv"
            )
