from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# App configuration and model loading
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "forest_fire_model.pkl"

st.set_page_config(
    page_title="Forest Fire Risk Monitor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading the forest-fire model…")
def load_model(model_path: str):
    return joblib.load(model_path)


# -----------------------------------------------------------------------------
# Visual styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --forest-bg: #071411;
        --forest-panel: #0d1e19;
        --forest-panel-2: #10251f;
        --forest-line: rgba(191, 220, 202, 0.13);
        --forest-text: #edf6ef;
        --forest-muted: #9aafa2;
        --forest-green: #73d6a0;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(43, 111, 76, 0.22), transparent 34%),
            radial-gradient(ellipse at 92% 12%, rgba(37, 91, 70, 0.16), transparent 30%),
            #071411;
        color: var(--forest-text);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    #MainMenu, footer { visibility: hidden; }

    .block-container {
        max-width: 1380px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--forest-text) !important;
        letter-spacing: -0.035em;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--forest-text);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.5rem 0 1.7rem 0;
        margin-bottom: 1.15rem;
        border-bottom: 1px solid var(--forest-line);
    }

    .hero-kicker {
        color: #8cddb0;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.19em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        color: #f1f8f2;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.1rem, 5vw, 3.65rem);
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -0.065em;
        margin: 0;
    }

    .hero-title span { color: #83d9a7; }

    .hero-subtitle {
        color: #a6b9ac;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 670px;
        margin-top: 0.85rem;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        color: #bfe8ce;
        background: rgba(94, 174, 124, 0.10);
        border: 1px solid rgba(128, 210, 157, 0.20);
        border-radius: 999px;
        padding: 0.42rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(17, 39, 31, 0.96), rgba(10, 27, 22, 0.96));
        border: 1px solid var(--forest-line);
        border-radius: 20px;
    }

    .section-kicker {
        color: #83d9a7;
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .section-title {
        color: #eef7f0;
        font-family: 'Manrope', sans-serif;
        font-weight: 750;
        font-size: 1.35rem;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .section-copy {
        color: #9fb2a5;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-bottom: 1.05rem;
    }

    [data-testid="stSlider"] label {
        color: #dceae0 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
        color: #8fa798 !important;
        font-size: 0.76rem !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background-color: #8ee2ad !important;
        border: 2px solid #0d1e19 !important;
        box-shadow: 0 0 0 3px rgba(115, 214, 160, 0.12) !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: #5dbb83 !important;
    }

    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 3.15rem;
        border: 1px solid rgba(151, 231, 178, 0.36);
        border-radius: 12px;
        color: #082015;
        background: linear-gradient(100deg, #8ce2ad, #62c88e);
        font-family: 'Manrope', sans-serif;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        transition: transform 180ms ease, filter 180ms ease, box-shadow 180ms ease;
        box-shadow: 0 8px 25px rgba(66, 177, 111, 0.16);
    }

    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        color: #06170f;
        filter: brightness(1.06);
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(66, 177, 111, 0.23);
    }

    .result-empty {
        min-height: 435px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2rem 1rem;
    }

    .empty-icon {
        width: 76px;
        height: 76px;
        display: grid;
        place-items: center;
        border-radius: 24px;
        font-size: 2.1rem;
        color: #9fe3b7;
        background: rgba(113, 205, 148, 0.10);
        border: 1px solid rgba(133, 219, 163, 0.18);
        margin-bottom: 1.2rem;
    }

    .empty-title {
        color: #eaf5ed;
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: 0.45rem;
    }

    .empty-copy {
        color: #9db2a4;
        font-size: 0.88rem;
        max-width: 280px;
        line-height: 1.6;
    }

    .risk-panel { padding: 0.25rem 0.1rem 0.1rem; }

    .risk-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
        margin-bottom: 1.15rem;
    }

    .risk-label {
        color: #9db3a5;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.16em;
    }

    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
        color: var(--risk);
        background: color-mix(in srgb, var(--risk) 12%, transparent);
        border: 1px solid color-mix(in srgb, var(--risk) 35%, transparent);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        white-space: nowrap;
    }

    .risk-dot {
        width: 7px;
        height: 7px;
        background: var(--risk);
        border-radius: 50%;
        box-shadow: 0 0 12px var(--risk);
    }

    .gauge-area {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.45rem 0 1.2rem;
    }

    .gauge {
        width: min(260px, 72vw);
        aspect-ratio: 1 / 1;
        display: grid;
        place-items: center;
        border-radius: 50%;
        background: conic-gradient(var(--risk) 0 var(--prob), rgba(162, 190, 171, 0.13) var(--prob) 100%);
        position: relative;
        animation: gauge-pop 700ms cubic-bezier(.2,.8,.2,1) both;
    }

    .gauge::before {
        content: '';
        position: absolute;
        inset: 12px;
        border-radius: 50%;
        background: radial-gradient(circle at 50% 35%, #18352a, #0b1b15 72%);
        border: 1px solid rgba(195, 225, 203, 0.10);
    }

    .gauge-content {
        position: relative;
        z-index: 1;
        text-align: center;
    }

    .gauge-number {
        color: #f3faf4;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.55rem, 5vw, 3.55rem);
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.07em;
        font-variant-numeric: tabular-nums;
    }

    .gauge-number span {
        color: var(--risk);
        font-size: 1.35rem;
        margin-left: 0.1rem;
    }

    .gauge-caption {
        color: #9eb3a5;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-top: 0.65rem;
    }

    .prediction-state {
        text-align: center;
        color: #f0f7f1;
        font-family: 'Manrope', sans-serif;
        font-size: 1.14rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0.1rem 0 0.3rem;
    }

    .prediction-note {
        text-align: center;
        color: #9db2a4;
        font-size: 0.82rem;
        line-height: 1.55;
        margin: 0 auto 1.25rem;
        max-width: 390px;
    }

    .result-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.7rem;
        margin-top: 1.1rem;
    }

    .stat-tile {
        padding: 0.9rem 1rem;
        background: rgba(4, 16, 11, 0.40);
        border: 1px solid rgba(190, 220, 199, 0.10);
        border-radius: 13px;
    }

    .stat-label {
        color: #8fa698;
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .stat-value {
        color: #eaf5ed;
        font-family: 'Manrope', sans-serif;
        font-size: 1rem;
        font-weight: 800;
    }

    .input-summary {
        margin-top: 1.15rem;
        padding-top: 1.05rem;
        border-top: 1px solid var(--forest-line);
    }

    .summary-heading {
        color: #9eb3a5;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    .summary-pills { display: flex; flex-wrap: wrap; gap: 0.45rem; }

    .summary-pill {
        color: #d4e7d9;
        background: rgba(126, 190, 146, 0.09);
        border: 1px solid rgba(147, 206, 164, 0.14);
        border-radius: 8px;
        padding: 0.38rem 0.56rem;
        font-size: 0.72rem;
    }

    .disclaimer {
        color: #81978a;
        font-size: 0.73rem;
        line-height: 1.6;
        margin-top: 1.15rem;
        padding: 0.85rem 1rem;
        background: rgba(8, 22, 16, 0.42);
        border: 1px solid rgba(190, 220, 199, 0.09);
        border-radius: 12px;
    }

    .footer-line {
        color: #71887a;
        text-align: center;
        font-size: 0.72rem;
        margin-top: 1.7rem;
    }

    @keyframes gauge-pop {
        from { opacity: 0.35; transform: scale(0.92) rotate(-12deg); }
        to { opacity: 1; transform: scale(1) rotate(0deg); }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }

    @media (max-width: 700px) {
        .block-container { padding-top: 1.1rem; }
        .hero { padding-top: 0.7rem; }
        .result-empty { min-height: 280px; }
        .gauge { width: min(230px, 68vw); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Environmental intelligence · Model dashboard</div>
        <h1 class="hero-title">Forest Fire <span>Risk Monitor</span></h1>
        <div class="hero-subtitle">
            Explore how weather conditions and fire-weather indices influence the
            model's estimate of forest-fire risk.
        </div>
        <div class="hero-chip">🌿 &nbsp; Random Forest · Algerian forest-fire dataset</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load model; show a clear message if the file is not in the right location
# -----------------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Put **forest_fire_model.pkl** in the same folder as **app.py**, then run the app again."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info(
        "Install the project dependencies from requirements.txt. The model was saved with scikit-learn 1.6.1."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Input form and result panel
# -----------------------------------------------------------------------------
input_column, result_column = st.columns([1.08, 0.92], gap="large")

with input_column:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">01 / Input parameters</div>
            <div class="section-title">Environmental conditions</div>
            <div class="section-copy">Adjust the sliders to describe the conditions you want to evaluate.</div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("fire_risk_form"):
            raw_col, index_col = st.columns(2, gap="medium")

            with raw_col:
                st.markdown("**Weather data**")
                temperature = st.slider("Temperature (°C)", 0, 50, 35, 1)
                ws = st.slider("Wind speed (km/h)", 0, 60, 15, 1)
                rain = st.slider("Rainfall (mm)", 0.0, 20.0, 0.0, 0.1)
                rh = st.slider("Relative humidity (%)", 0, 100, 30, 1)

            with index_col:
                st.markdown("**Fire-weather indices**")
                ffmc = st.slider("FFMC", 0.0, 100.0, 90.0, 0.1, help="Fine Fuel Moisture Code")
                dmc = st.slider("DMC", 0.0, 100.0, 20.0, 0.1, help="Duff Moisture Code")
                dc = st.slider("DC", 0.0, 500.0, 100.0, 0.1, help="Drought Code")
                isi = st.slider("ISI", 0.0, 50.0, 10.0, 0.1, help="Initial Spread Index")
                bui = st.slider("BUI", 0.0, 100.0, 25.0, 0.1, help="Build Up Index")
                fwi = st.slider("FWI", 0.0, 50.0, 12.0, 0.1, help="Fire Weather Index")

            submitted = st.form_submit_button("🔥  Calculate fire risk", use_container_width=True)

        st.markdown(
            "<div class='disclaimer'>Use values in the same units and definitions as the training dataset. The slider limits are interface limits, not official safety thresholds.</div>",
            unsafe_allow_html=True,
        )

with result_column:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">02 / Prediction output</div>
            <div class="section-title">Fire-risk assessment</div>
            """,
            unsafe_allow_html=True,
        )

        if submitted:
            input_data = pd.DataFrame(
                [{
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
                }]
            )

            try:
                class_values = list(model.classes_)
                fire_class_index = class_values.index(1)
                probability = float(model.predict_proba(input_data)[0][fire_class_index])
                predicted_class = int(model.predict(input_data)[0])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

            probability_percent = probability * 100

            if probability_percent < 20:
                risk_level = "LOW"
                risk_color = "#65d69a"
                risk_message = "The model estimates a low fire-risk pattern for these inputs."
            elif probability_percent < 50:
                risk_level = "MODERATE"
                risk_color = "#f0bd62"
                risk_message = "The model estimates a moderate fire-risk pattern. Keep monitoring conditions."
            else:
                risk_level = "HIGH"
                risk_color = "#ff785f"
                risk_message = "The model estimates a high fire-risk pattern for these inputs."

            classification = "FIRE" if predicted_class == 1 else "NOT FIRE"

            st.markdown(
                f"""
                <div class="risk-panel">
                    <div class="risk-topline">
                        <div class="risk-label">MODEL-ESTIMATED LIKELIHOOD</div>
                        <div class="risk-badge" style="--risk:{risk_color};">
                            <span class="risk-dot"></span>{risk_level} RISK
                        </div>
                    </div>
                    <div class="gauge-area">
                        <div class="gauge" style="--prob:{probability_percent:.2f}%; --risk:{risk_color};">
                            <div class="gauge-content">
                                <div class="gauge-number">{probability_percent:.2f}<span>%</span></div>
                                <div class="gauge-caption">FIRE LIKELIHOOD</div>
                            </div>
                        </div>
                    </div>
                    <div class="prediction-state">{classification}</div>
                    <div class="prediction-note">{risk_message}</div>
                    <div class="result-stats">
                        <div class="stat-tile">
                            <div class="stat-label">Model decision</div>
                            <div class="stat-value">{classification}</div>
                        </div>
                        <div class="stat-tile">
                            <div class="stat-label">Decision threshold</div>
                            <div class="stat-value">50%</div>
                        </div>
                    </div>
                    <div class="input-summary">
                        <div class="summary-heading">Conditions evaluated</div>
                        <div class="summary-pills">
                            <span class="summary-pill">🌡️ {temperature}°C</span>
                            <span class="summary-pill">💧 RH {rh}%</span>
                            <span class="summary-pill">💨 Wind {ws} km/h</span>
                            <span class="summary-pill">🌧️ Rain {rain:.1f} mm</span>
                            <span class="summary-pill">FWI {fwi:.1f}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-empty">
                    <div class="empty-icon">🌲</div>
                    <div class="empty-title">Your results will appear here</div>
                    <div class="empty-copy">Set the environmental values, then calculate the fire-risk estimate to see the animated gauge and model prediction.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.markdown(
    """
    <div class="disclaimer">
        <strong>Important:</strong> This is an academic machine-learning demonstration trained on the
        Algerian forest-fire dataset. The displayed percentage is the model's estimate, not a verified
        real-world probability or an official fire warning. Do not use it for emergency decisions.
    </div>
    <div class="footer-line">FOREST FIRE RISK MONITOR · MACHINE LEARNING PROJECT</div>
    """,
    unsafe_allow_html=True,
)
