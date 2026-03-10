# app.py — California Housing Price Prediction
# Streamlit app with modern UI design

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── Page Config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="California Housing Price Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --cream:   #F5F0E8;
    --sand:    #E8DFD0;
    --terracotta: #C2714F;
    --rust:    #A0522D;
    --forest:  #2C4A3E;
    --moss:    #4A6741;
    --ink:     #1A1A2E;
    --mist:    #8B9BAE;
    --gold:    #D4A853;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1100px;
}

/* ── Hero Section ── */
.hero {
    background: linear-gradient(135deg, var(--forest) 0%, #1a3a30 50%, var(--ink) 100%);
    border-radius: 24px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(212,168,83,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 10%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(194,113,79,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(212,168,83,0.2);
    color: var(--gold);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
    border: 1px solid rgba(212,168,83,0.3);
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #FFFFFF;
    line-height: 1.15;
    margin: 0 0 1rem 0;
}
.hero h1 em {
    font-style: italic;
    color: var(--gold);
}
.hero p {
    color: rgba(255,255,255,0.65);
    font-size: 1rem;
    line-height: 1.7;
    max-width: 560px;
    margin: 0;
    font-weight: 300;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}
.hero-stat {
    display: flex;
    flex-direction: column;
}
.hero-stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #fff;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* ── Section Heading ── */
.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--ink);
    margin: 0 0 0.3rem 0;
}
.section-sub {
    font-size: 0.85rem;
    color: var(--mist);
    margin: 0 0 1.5rem 0;
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border: 1px solid #EAE4DA;
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1rem;
}
.card-sm {
    background: var(--cream);
    border: 1px solid #E0D8CA;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
}

/* ── Labels for inputs ── */
label, .stNumberInput label, .stSelectbox label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #4A4A5A !important;
    letter-spacing: 0.02em !important;
}

/* ── Input fields ── */
.stNumberInput input, .stSelectbox select,
[data-baseweb="input"] input {
    border-radius: 10px !important;
    border-color: #DDD6C8 !important;
    background: #FDFAF6 !important;
    color: #1A1A2E !important;
    -webkit-text-fill-color: #1A1A2E !important;
    font-size: 0.95rem !important;
}
.stNumberInput input:focus {
    border-color: var(--forest) !important;
    box-shadow: 0 0 0 3px rgba(44,74,62,0.1) !important;
}
[data-baseweb="base-input"], [data-testid="stNumberInput"] input {
    background: #FDFAF6 !important;
    color: #1A1A2E !important;
    -webkit-text-fill-color: #1A1A2E !important;
}

/* ── Primary Button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--forest) 0%, var(--moss) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(44,74,62,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(44,74,62,0.35) !important;
}
.stButton > button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}

/* ── Result Card ── */
.result-card {
    background: linear-gradient(135deg, var(--forest) 0%, #1e3d33 100%);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '🏡';
    position: absolute;
    font-size: 8rem;
    opacity: 0.06;
    top: -10px; right: -10px;
    line-height: 1;
}
.result-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.55);
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: #FFFFFF;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-range {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
}

/* ── Metric Boxes ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.2rem 0;
}
.metric-box {
    flex: 1;
    background: var(--cream);
    border: 1px solid #E0D8CA;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--ink);
}
.metric-lbl {
    font-size: 0.72rem;
    color: var(--mist);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Comparison Bar ── */
.compare-bar-wrap {
    background: #EAE4DA;
    border-radius: 100px;
    height: 8px;
    margin: 0.4rem 0 1rem 0;
    overflow: hidden;
}
.compare-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--forest), var(--moss));
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Warning ── */
.stAlert {
    border-radius: 12px !important;
}

/* ── Divider ── */
hr {
    border-color: #EAE4DA !important;
    margin: 2rem 0 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    color: var(--mist) !important;
    background: var(--cream) !important;
    border-radius: 10px !important;
}

/* ── Ocean badge ── */
.ocean-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(44,74,62,0.08);
    color: var(--forest);
    border: 1px solid rgba(44,74,62,0.15);
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    margin-top: 0.5rem;
}

/* ── Model info pills ── */
.pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.pill {
    background: var(--cream);
    border: 1px solid #DDD6C8;
    border-radius: 100px;
    padding: 0.3rem 0.8rem;
    font-size: 0.78rem;
    color: #5A5A6A;
    font-weight: 500;
}
.pill-green {
    background: rgba(44,74,62,0.08);
    border-color: rgba(44,74,62,0.2);
    color: var(--forest);
}

/* ── Feature importance bar ── */
.feat-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.6rem; }
.feat-name { font-size: 0.78rem; color: #5A5A6A; width: 160px; flex-shrink: 0; }
.feat-bar-bg { flex: 1; background: #EAE4DA; border-radius: 4px; height: 6px; }
.feat-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--forest), var(--moss)); }
.feat-pct { font-size: 0.75rem; color: var(--mist); width: 36px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ─── Train Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
    df = pd.read_csv(url)

    df['rooms_per_household']      = df['total_rooms'] / df['households']
    df['bedrooms_ratio']           = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    numeric_features = [
        'longitude','latitude','housing_median_age',
        'total_rooms','total_bedrooms','population','households',
        'median_income','rooms_per_household','bedrooms_ratio','population_per_household'
    ]
    categorical_features = ['ocean_proximity']

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre_num = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    pre_cat = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', pre_num, numeric_features),
                                       ('cat', pre_cat, categorical_features)])
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, max_depth=15,
                                      min_samples_split=5, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    rf = pipeline.named_steps['reg']
    ohe_cats = pipeline.named_steps['pre'].transformers_[1][1].named_steps['ohe'].categories_[0]
    feat_names = numeric_features + [f'ocean_{c}' for c in ohe_cats]
    importance = sorted(
        [{'feature': f, 'importance': float(i)} for f, i in zip(feat_names, rf.feature_importances_)],
        key=lambda x: x['importance'], reverse=True
    )

    return pipeline, {
        'rmse': round(rmse, 0), 'mae': round(mae, 0), 'r2': round(r2, 4),
        'n_train': len(X_train), 'y_mean': round(float(y.mean()), 0),
        'importance': importance
    }

with st.spinner("Loading model…"):
    pipeline, stats = train_model()

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-tag">🏡 ML Price Estimator</div>
    <h1>California Housing<br><em>Price Prediction</em></h1>
    <p>ระบบประเมินราคาบ้านในแคลิฟอร์เนียด้วย Machine Learning<br>
       Train จากข้อมูลสำมะโนประชากร 20,640 ย่าน</p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-val">R² {stats['r2']:.3f}</span>
            <span class="hero-stat-lbl">Model Accuracy</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-val">${stats['rmse']:,.0f}</span>
            <span class="hero-stat-lbl">RMSE</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-val">{stats['n_train']:,}</span>
            <span class="hero-stat-lbl">Training Samples</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-val">Random<br>Forest</span>
            <span class="hero-stat-lbl">Algorithm</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # ── Location ──
    st.markdown('<p class="section-heading">📍 Location</p><p class="section-sub">พิกัดตำแหน่งของย่านที่อยู่อาศัย</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        longitude = st.number_input("Longitude", min_value=-124.5, max_value=-114.0,
                                     value=-119.0, step=0.01, format="%.2f",
                                     help="แคลิฟอร์เนีย: -124.5 ถึง -114.0")
    with c2:
        latitude = st.number_input("Latitude", min_value=32.5, max_value=42.0,
                                    value=36.0, step=0.01, format="%.2f",
                                    help="แคลิฟอร์เนีย: 32.5 ถึง 42.0")

    ocean_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
    ocean_desc = {
        '<1H OCEAN': '🌊  ห่างจากทะเล < 1 ชั่วโมง',
        'INLAND':    '🏔️  อยู่ในแผ่นดิน',
        'NEAR OCEAN':'🏖️  ใกล้มหาสมุทรแปซิฟิก',
        'NEAR BAY':  '⛵  ใกล้อ่าว San Francisco',
        'ISLAND':    '🏝️  บนเกาะ'
    }
    ocean_proximity = st.selectbox("Ocean Proximity", options=ocean_options,
                                    help="ระยะห่างจากทะเล")
    st.markdown(f'<div class="ocean-badge">{ocean_desc[ocean_proximity]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Housing Info ──
    st.markdown('<p class="section-heading">🏘️ Housing Info</p><p class="section-sub">ข้อมูลเกี่ยวกับบ้านและย่านที่อยู่อาศัย</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        housing_median_age = st.number_input("Median House Age (ปี)", min_value=1, max_value=52, value=20, step=1,
                                              help="อายุมัธยฐานของบ้านในย่าน")
        total_rooms = st.number_input("Total Rooms", min_value=1, max_value=40000, value=2000, step=100,
                                       help="จำนวนห้องรวมทั้งหมดในย่าน")
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=7000, value=400, step=10,
                                          help="จำนวนห้องนอนรวมทั้งหมดในย่าน")
    with c4:
        population = st.number_input("Population (คน)", min_value=1, max_value=40000, value=1200, step=100,
                                      help="จำนวนประชากรในย่าน")
        households = st.number_input("Households (ครัวเรือน)", min_value=1, max_value=7000, value=350, step=10,
                                      help="จำนวนครัวเรือนในย่าน")
        median_income = st.number_input("Median Income (หมื่น $)", min_value=0.5, max_value=15.0,
                                         value=4.0, step=0.1, format="%.1f",
                                         help="5.0 = รายได้ $50,000/ปี | avg = 3.87")

    # ── Validation ──
    warnings_list = []
    if total_bedrooms > total_rooms:
        warnings_list.append("⚠️ Total Bedrooms มากกว่า Total Rooms — กรุณาตรวจสอบ")
    if population < households:
        warnings_list.append("⚠️ Population น้อยกว่า Households — กรุณาตรวจสอบ")
    for w in warnings_list:
        st.warning(w)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ──
    predict_btn = st.button("Estimate Price →", type="primary",
                             use_container_width=True,
                             disabled=len(warnings_list) > 0)

# ─── Right Column: Model Info ──────────────────────────────────────────────────
with right_col:
    st.markdown('<p class="section-heading">📊 Model Performance</p><p class="section-sub">ประสิทธิภาพของโมเดล</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <div class="pill-row">
            <span class="pill pill-green">✓ RandomForestRegressor</span>
            <span class="pill">100 Trees</span>
            <span class="pill">Depth 15</span>
        </div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{stats['r2']:.3f}</div>
                <div class="metric-lbl">R² Score</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">${stats['mae']:,.0f}</div>
                <div class="metric-lbl">MAE</div>
            </div>
        </div>
        <p style="font-size:0.78rem;color:#8B9BAE;margin:0;">
        โมเดลอธิบายความแปรปรวนของราคาบ้านได้ <strong>{stats['r2']*100:.1f}%</strong> 
        ข้อผิดพลาดเฉลี่ย ±${stats['mae']:,.0f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br><p class="section-heading">🎯 Feature Importance</p><p class="section-sub">Features ที่มีอิทธิพลต่อราคาบ้านมากที่สุด</p>', unsafe_allow_html=True)

    feat_html = '<div class="card">'
    top_imp = stats['importance'][:7]
    max_imp = top_imp[0]['importance']
    feat_labels = {
        'median_income': 'Median Income',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'housing_median_age': 'House Age',
        'rooms_per_household': 'Rooms/HH',
        'bedrooms_ratio': 'Bedroom Ratio',
        'population_per_household': 'Pop/HH',
        'total_rooms': 'Total Rooms',
        'households': 'Households',
        'population': 'Population',
        'total_bedrooms': 'Total Bedrooms',
    }

    # รวม ocean_* categories ทั้งหมดให้เป็น "Ocean Proximity" ก้อนเดียว
    merged = {}
    for item in stats['importance']:
        fname = item['feature']
        if fname.startswith('ocean_'):
            merged['Ocean Proximity'] = merged.get('Ocean Proximity', 0) + item['importance']
        else:
            label = feat_labels.get(fname, fname)
            merged[label] = merged.get(label, 0) + item['importance']
    merged_sorted = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:7]
    max_imp = merged_sorted[0][1]

    for fname, imp in merged_sorted:
        pct = imp / max_imp * 100
        feat_html += f"""
        <div class="feat-row">
            <span class="feat-name">{fname}</span>
            <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{pct:.0f}%"></div></div>
            <span class="feat-pct">{imp*100:.1f}%</span>
        </div>"""
    feat_html += '</div>'
    st.markdown(feat_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="card-sm" style="margin-top:1rem;">
        <p style="font-size:0.78rem;color:#8B9BAE;margin:0;line-height:1.6;">
        ⚠️ <strong>Disclaimer</strong><br>
        ผลลัพธ์นี้เป็นการประมาณจาก ML เท่านั้น<br>
        ข้อมูลมาจากสำมะโนประชากรแคลิฟอร์เนีย ปี 1990<br>
        ไม่ควรใช้แทนการประเมินราคาจากผู้เชี่ยวชาญ
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Prediction Result ─────────────────────────────────────────────────────────
if predict_btn:
    rph  = total_rooms / households
    br   = total_bedrooms / total_rooms
    pph  = population / households

    input_df = pd.DataFrame([{
        'longitude': longitude, 'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms, 'total_bedrooms': total_bedrooms,
        'population': population, 'households': households,
        'median_income': median_income, 'ocean_proximity': ocean_proximity,
        'rooms_per_household': rph, 'bedrooms_ratio': br,
        'population_per_household': pph
    }])

    with st.spinner("Estimating price…"):
        price = float(pipeline.predict(input_df)[0])
        price = max(10000, min(price, 600000))

    mae_val = stats['mae']
    low_p   = max(0, price - mae_val)
    high_p  = price + mae_val
    avg_p   = stats['y_mean']
    diff    = price - avg_p
    pct_bar = min(price / 500000, 1.0)

    st.markdown("---")
    st.markdown('<p class="section-heading">💰 Estimated Price</p>', unsafe_allow_html=True)

    r1, r2_col, r3 = st.columns([1, 2, 1])
    with r2_col:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Median House Value</div>
            <div class="result-price">${price:,.0f}</div>
            <div class="result-range">Range: ${low_p:,.0f} — ${high_p:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Low Estimate", f"${low_p:,.0f}")
    m2.metric("Prediction",   f"${price:,.0f}")
    m3.metric("High Estimate",f"${high_p:,.0f}")
    st.caption(f"ช่วงราคาอ้างอิงจาก MAE ±${mae_val:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison vs average
    if diff > 0:
        st.success(f"📈 สูงกว่าค่าเฉลี่ย dataset **${avg_p:,.0f}** อยู่ **{(diff/avg_p)*100:.1f}%** (+${diff:,.0f})")
    else:
        st.info(f"📉 ต่ำกว่าค่าเฉลี่ย dataset **${avg_p:,.0f}** อยู่ **{abs(diff/avg_p)*100:.1f}%** (${diff:,.0f})")

    # Price bar
    st.markdown(f"""
    <div style="margin:0.5rem 0 1.5rem 0;">
        <div style="font-size:0.78rem;color:#8B9BAE;margin-bottom:0.4rem;">
            Price Level vs. Max ($500,000)
        </div>
        <div class="compare-bar-wrap">
            <div class="compare-bar-fill" style="width:{pct_bar*100:.1f}%"></div>
        </div>
        <div style="font-size:0.75rem;color:#8B9BAE;text-align:right;">${price:,.0f} / $500,000</div>
    </div>
    """, unsafe_allow_html=True)

    # Derived features summary
    with st.expander("📋 View Full Input Summary"):
        summary_df = pd.DataFrame({
            'Feature': [
                'Longitude','Latitude','Ocean Proximity','House Age (yr)',
                'Total Rooms','Total Bedrooms','Population','Households',
                'Median Income (×$10k)',
                '— Rooms/Household (derived)','— Bedroom Ratio (derived)','— Pop/Household (derived)'
            ],
            'Value': [
                longitude, latitude, ocean_proximity, housing_median_age,
                f"{total_rooms:,}", f"{total_bedrooms:,}",
                f"{population:,}", f"{households:,}",
                f"{median_income:.1f}",
                f"{rph:.2f}", f"{br:.3f}", f"{pph:.2f}"
            ]
        }).set_index('Feature')
        st.dataframe(summary_df, use_container_width=True)
