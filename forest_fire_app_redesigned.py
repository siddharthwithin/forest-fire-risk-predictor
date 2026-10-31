
import streamlit as st
import pandas as pd
import joblib
import math

st.set_page_config(
    page_title="Forest Fire Risk Monitor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Model ----------
MODEL_PATH = "forest_fire_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 5%, rgba(38, 122, 78, 0.22), transparent 28%),
        radial-gradient(circle at 85% 12%, rgba(255, 102, 72, 0.12), transparent 25%),
        linear-gradient(135deg, #06130f 0%, #0a2119 45%, #07120f 100%);
    color: #eef8f1;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 34px 36px;
    border: 1px solid rgba(151, 235, 180, 0.18);
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(20, 70, 47, .82), rgba(7, 25, 19, .92)),
        radial-gradient(circle at 90% 30%, rgba(255, 92, 65, .18), transparent 30%);
    box-shadow: 0 20px 60px rgba(0,0,0,.28);
    margin-bottom: 24px;
}

.eyebrow {
    color: #8ee6ad;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(38px, 5vw, 64px);
    line-height: 1;
    margin: 10px 0 12px 0;
    font-weight: 800;
    letter-spacing: -2px;
}

.hero p {
    color: #b9d3c2;
    font-size: 16px;
    max-width: 720px;
    line-height: 1.65;
    margin-bottom: 18px;
}

.badge {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(143, 230, 173, .11);
    border: 1px solid rgba(143, 230, 173, .22);
    color: #b9f4ca;
    font-size: 12px;
    font-weight: 700;
}

.section-title {
    color: #8ee6ad;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin: 8px 0 10px;
}

.card {
    background: rgba(12, 36, 27, .74);
    border: 1px solid rgba(157, 231, 180, .14);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 15px 45px rgba(0,0,0,.20);
    height: 100%;
}

.card h2 {
    margin: 0 0 4px;
    font-size: 23px;
}

.muted {
    color: #9db8a7;
    font-size: 13px;
    line-height: 1.5;
}

.metric-card {
    background: linear-gradient(145deg, rgba(19,54,40,.9), rgba(9,28,21,.9));
    border: 1px solid rgba(154, 226, 177, .13);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    min-height: 112px;
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    margin-top: 5px;
}

.metric-label {
    color: #93ad9d;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.risk-box {
    text-align: center;
    padding: 18px;
    border-radius: 20px;
    margin-top: 15px;
}

.risk-high {
    background: rgba(255, 91, 70, .12);
    border: 1px solid rgba(255, 112, 92, .35);
}

.risk-medium {
    background: rgba(255, 190, 70, .11);
    border: 1px solid rgba(255, 202, 91, .32);
}

.risk-low {
    background: rgba(86, 220, 137, .10);
    border: 1px solid rgba(102, 232, 148, .28);
}

.risk-label {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
}

.gauge-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 16px 0 8px;
}

.gauge {
    width: 230px;
    height: 230px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    background: conic-gradient(
        #ff765e var(--progress),
        rgba(118, 159, 133, .17) var(--progress)
    );
    box-shadow: 0 0 45px rgba(255, 106, 84, .12);
}

.gauge::before {
    content: "";
    position: absolute;
    width: 184px;
    height: 184px;
    border-radius: 50%;
    background: #0b2119;
    box-shadow: inset 0 0 30px rgba(0,0,0,.25);
}

.gauge-content {
    position: relative;
    z-index: 2;
    text-align: center;
}

.gauge-number {
    font-size: 39px;
    font-weight: 800;
    line-height: 1;
}

.gauge-caption {
    margin-top: 7px;
    color: #9db8a7;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.info-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(157, 205, 174, .10);
    font-size: 13px;
}

.info-row:last-child {
    border-bottom: 0;
}

.info-key {
    color: #8fa99a;
}

.info-value {
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #718a7b;
    font-size: 11px;
    padding: 24px 0 4px;
}

div.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(160, 242, 185, .35);
    background: linear-gradient(135deg, #8fe8ae, #54c87d);
    color: #062016;
    font-weight: 800;
    padding: 12px 20px;
    box-shadow: 0 8px 24px rgba(70, 210, 123, .18);
}

div.stButton > button:hover {
    border-color: #b8ffd0;
    transform: translateY(-1px);
}

.stSlider > div > div > div > div {
    background: #8fe8ae !important;
}

label {
    color: #d9eade !important;
}

[data-testid="stMetricValue"] {
    color: #ecfff2;
}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">Environmental Intelligence • Machine Learning Dashboard</div>
    <h1>🌲 Forest Fire Risk Monitor</h1>
    <p>
        Estimate forest-fire risk from environmental conditions using a trained
        Random Forest machine-learning model.
    </p>
    <span class="badge">🔥 Random Forest • Extended Algerian Forest Fires Dataset</span>
</div>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
st.markdown('<div class="section-title">01 / Environmental Inputs</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="card">
        <h2>🌦️ Weather conditions</h2>
        <div class="muted">Adjust the current environmental conditions.</div>
    """, unsafe_allow_html=True)

    temperature = st.slider("Temperature (°C)", 0, 50, 35)
    ws = st.slider("Wind speed (km/h)", 0, 50, 15)
    rain = st.slider("Rainfall (mm)", 0.0, 20.0, 0.0, step=0.1)
    rh = st.slider("Relative humidity (%)", 0, 100, 30)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="card">
        <h2>🔥 Fire-weather indices</h2>
        <div class="muted">Indices supplied to the trained model.</div>
    """, unsafe_allow_html=True)

    ffmc = st.slider("FFMC", 0.0, 100.0, 90.0, step=0.1)
    dmc = st.slider("DMC", 0.0, 100.0, 20.0, step=0.1)
    dc = st.slider("DC", 0.0, 200.0, 100.0, step=0.1)
    isi = st.slider("ISI", 0.0, 20.0, 10.0, step=0.1)
    bui = st.slider("BUI", 0.0, 100.0, 25.0, step=0.1)
    fwi = st.slider("FWI", 0.0, 50.0, 12.0, step=0.1)

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
calculate = st.button("🔥  Calculate Fire Risk", use_container_width=True)

# ---------- Prediction ----------
if calculate:
    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Ws": ws,
        "Rain": rain,
        "RH": rh,
        "FFMC": ffmc,
        "DMC": dmc,
        "DC": dc,
        "ISI": isi,
        "BUI": bui,
        "FWI": fwi,
    }])

    probability = float(model.predict_proba(input_data)[0][1])
    pct = probability * 100

    if probability >= 0.70:
        risk = "HIGH RISK"
        decision = "FIRE"
        risk_class = "risk-high"
    elif probability >= 0.30:
        risk = "MODERATE RISK"
        decision = "NOT FIRE"
        risk_class = "risk-medium"
    else:
        risk = "LOW RISK"
        decision = "NOT FIRE"
        risk_class = "risk-low"

    st.markdown('<div class="section-title">02 / Prediction Output</div>', unsafe_allow_html=True)

    out_left, out_right = st.columns([1.05, 1], gap="large")

    with out_left:
        progress = max(0, min(100, pct))
        st.markdown(f"""
        <div class="card">
            <h2>🎯 Fire-risk assessment</h2>
            <div class="muted">Model-estimated likelihood</div>

            <div class="gauge-wrap">
                <div class="gauge" style="--progress:{progress}%">
                    <div class="gauge-content">
                        <div class="gauge-number">{pct:.2f}%</div>
                        <div class="gauge-caption">Fire likelihood</div>
                    </div>
                </div>
            </div>

            <div class="risk-box {risk_class}">
                <div class="risk-label">{risk}</div>
                <div style="margin-top:7px;font-size:22px;font-weight:800;">{decision}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with out_right:
        st.markdown("""
        <div class="card">
            <h2>📊 Conditions evaluated</h2>
            <div class="muted">Values supplied to the model for this prediction.</div>
        """, unsafe_allow_html=True)

        metrics = [
            ("🌡️", f"{temperature}°C", "Temperature"),
            ("💧", f"{rh}%", "Humidity"),
            ("💨", f"{ws} km/h", "Wind speed"),
            ("🌧️", f"{rain:.1f} mm", "Rainfall"),
        ]

        cols = st.columns(2)
        for i, (icon, value, label) in enumerate(metrics):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:20px">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-row">
            <span class="info-key">Model decision</span>
            <span class="info-value">{decision}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Decision threshold</span>
            <span class="info-value">50%</span>
        </div>
        <div class="info-row">
            <span class="info-key">Algorithm</span>
            <span class="info-value">Random Forest</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:38px; margin-top:22px;">
        <div style="font-size:42px;">🌲</div>
        <h2>Ready for assessment</h2>
        <div class="muted">
            Adjust the environmental values above and calculate the fire-risk estimate.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Model info ----------
st.write("")
st.markdown('<div class="section-title">03 / Model Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

for col, value, label in [
    (c1, "89.3%", "Test accuracy"),
    (c2, "1,215", "Dataset records"),
    (c3, "10", "Input features"),
    (c4, "Random Forest", "Algorithm"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Forest Fire Risk Monitor • Machine-learning demonstration •
    Predictions are estimates based on the trained dataset and should not be treated as an official fire warning system.
</div>
""", unsafe_allow_html=True)
