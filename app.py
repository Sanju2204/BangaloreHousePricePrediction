import streamlit as st
import pandas as pd
import pickle
import numpy as np
import json

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model/house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    feature_names = list(model.feature_names_in_)
    locations = sorted([
        col.replace("location_", "")
        for col in feature_names
        if col.startswith("location_")
    ])
    return model, feature_names, locations

model, feature_names, locations = load_model()

# ── Prediction Logic ─────────────────────────────────────────────────────────
def predict_price(location, sqft, bath, bhk):
    col_name = f"location_{location}"
    if col_name not in feature_names:
        col_name = "location_other"
    row = {f: 0 for f in feature_names}
    row["total_sqft"] = sqft
    row["bath"] = bath
    row["bhk"] = bhk
    row[col_name] = 1
    df = pd.DataFrame([row])
    return round(model.predict(df)[0], 2)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&family=Satoshi:wght@300;400;500;700&display=swap');

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #060612 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,57,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,210,180,0.12) 0%, transparent 55%),
        #060612 !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

.block-container {
    max-width: 760px !important;
    padding: 2.5rem 1.5rem 4rem !important;
}

/* ── Typography ── */
h1, h2, h3, h4, p, label, span, div {
    font-family: 'Satoshi', sans-serif !important;
    color: #e2e8f0;
}

/* ── Hero Section ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 0 2.2rem;
    position: relative;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(99,57,255,0.15);
    border: 1px solid rgba(99,57,255,0.35);
    border-radius: 30px;
    padding: 5px 16px;
    font-size: 11.5px;
    font-weight: 500;
    color: #a78bfa;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-badge .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #a78bfa;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .5; transform: scale(.75); }
}
.hero-title {
    font-family: 'Clash Display', sans-serif !important;
    font-size: clamp(2rem, 6vw, 3.2rem) !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    color: #f8fafc !important;
    margin-bottom: .5rem !important;
}
.hero-title .accent {
    background: linear-gradient(90deg, #818cf8, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 15px !important;
    color: #64748b !important;
    margin-bottom: 0 !important;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem 2rem;
    backdrop-filter: blur(8px);
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,57,255,0.4), transparent);
}
.card-title {
    font-family: 'Clash Display', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: .12em;
    margin-bottom: 1.2rem !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-title::before {
    content: '';
    width: 3px; height: 14px;
    background: linear-gradient(180deg, #818cf8, #06b6d4);
    border-radius: 2px;
    display: inline-block;
}

/* ── Streamlit Inputs override ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Satoshi', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: rgba(129,140,248,0.5) !important;
    background: rgba(129,140,248,0.07) !important;
}
[data-testid="stSelectbox"] svg { color: #818cf8 !important; }

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #6139ff, #06b6d4) !important;
}
[data-testid="stSlider"] .st-by { color: #818cf8 !important; }

/* Labels */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
    font-family: 'Satoshi', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: .08em;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #6139ff 0%, #4f46e5 60%, #0891b2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Clash Display', sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    letter-spacing: .04em;
    cursor: pointer;
    transition: transform .15s, box-shadow .2s !important;
    box-shadow: 0 4px 20px rgba(99,57,255,0.35) !important;
    margin-top: .4rem !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,57,255,0.5) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Result Box ── */
.result-wrap {
    background: linear-gradient(135deg, rgba(16,185,129,.08), rgba(6,182,212,.06));
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 2rem 2rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeUp .4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(16,185,129,.12) 0%, transparent 70%);
    border-radius: 50%;
}
.result-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: #10b981;
    font-weight: 600;
    margin-bottom: .6rem;
}
.result-price {
    font-family: 'Clash Display', sans-serif;
    font-size: clamp(2.4rem, 8vw, 4rem);
    font-weight: 700;
    color: #f8fafc;
    line-height: 1;
    margin-bottom: .5rem;
}
.result-price sup { font-size: 40%; vertical-align: top; padding-top: .4em; color: #10b981; }
.result-price sub { font-size: 30%; vertical-align: bottom; color: #64748b; }
.result-note { font-size: 13px; color: #64748b; }

/* ── Stats Row ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 1.4rem;
}
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'Clash Display', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #818cf8;
    margin-bottom: .25rem;
}
.stat-lbl {
    font-size: 10.5px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: .07em;
}

/* ── Warning ── */
.warn-box {
    background: rgba(245,158,11,.08);
    border: 1px solid rgba(245,158,11,.25);
    border-radius: 12px;
    padding: .8rem 1.1rem;
    font-size: 13px;
    color: #fbbf24;
    margin-bottom: 1rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 2.5rem;
    font-size: 12px;
    color: #334155;
}
.footer span { color: #4f46e5; }

/* hide streamlit branding */
#MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge"><span class="dot"></span> ML Powered · Linear Regression</div>
  <h1 class="hero-title">Bangalore <span class="accent">House Price</span> Predictor</h1>
  <p class="hero-sub">Estimated price using real market data · 127 features · 200+ localities</p>
</div>
""", unsafe_allow_html=True)

# ── Form Card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Property Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("📍 Location", options=locations, index=locations.index("Whitefield") if "Whitefield" in locations else 0)
    sqft = st.number_input("📐 Total Area (sq ft)", min_value=300.0, max_value=10000.0, value=1200.0, step=50.0)
with col2:
    bhk = st.number_input("🛏 BHK", min_value=1, max_value=10, value=2, step=1)
    bath = st.number_input("🚿 Bathrooms", min_value=1, max_value=8, value=2, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Validation Warning ────────────────────────────────────────────────────────
if bath > bhk + 2:
    st.markdown(f"""
    <div class="warn-box">
      ⚠ &nbsp; <strong>{bath} bathrooms</strong> for a <strong>{bhk} BHK</strong> seems unusual. Please double-check.
    </div>
    """, unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("⚡  Predict Price", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Calculating..."):
        price = predict_price(location, sqft, bath, bhk)
        price_cr  = price / 100
        psf       = int((price * 100_000) / sqft)

    st.markdown(f"""
    <div class="result-wrap">
      <div class="result-label">✦ Estimated Market Price</div>
      <div class="result-price">
        <sup>₹</sup>{price:.1f}<sub>&nbsp;Lakhs</sub>
      </div>
      <div class="result-note">{location} · {int(sqft):,} sq ft · {bhk} BHK · {bath} Bath</div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-val">₹{psf:,}</div>
          <div class="stat-lbl">Per Sq Ft</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">{price_cr:.2f} Cr</div>
          <div class="stat-lbl">In Crores</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">{bhk}B / {bath}B</div>
          <div class="stat-lbl">Config</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built with <span>♥</span> · Bangalore Real Estate ML Model · For estimation purposes only
</div>
""", unsafe_allow_html=True)