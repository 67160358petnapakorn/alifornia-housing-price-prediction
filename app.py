# app.py — Streamlit application สำหรับทำนายราคาบ้านในแคลิฟอร์เนีย

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

# ===== การตั้งค่าหน้าเว็บ =====
# st.set_page_config ต้องเป็น Streamlit command แรกเสมอ
st.set_page_config(
    page_title="ระบบทำนายราคาบ้านแคลิฟอร์เนีย",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===== Train โมเดลตอนเริ่ม app =====
# ใช้ @st.cache_resource เพื่อ train ครั้งเดียว ไม่ train ซ้ำทุกครั้งที่ผู้ใช้ interact
# วิธีนี้ไม่ต้องเก็บไฟล์ .pkl ใน GitHub (ซึ่งมักใหญ่เกิน 25MB)
@st.cache_resource
def train_model():
    """โหลดข้อมูลและ train pipeline — ทำครั้งเดียวตอนเริ่ม app"""

    url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
    df = pd.read_csv(url)

    # Feature Engineering (เหมือนกับใน notebook)
    df['rooms_per_household']     = df['total_rooms'] / df['households']
    df['bedrooms_ratio']          = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    numeric_features = [
        'longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms', 'population', 'households',
        'median_income', 'rooms_per_household', 'bedrooms_ratio', 'population_per_household'
    ]
    categorical_features = ['ocean_proximity']

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # สร้าง Pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    # คำนวณ metrics
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    # Feature importance
    rf = pipeline.named_steps['regressor']
    ohe_cats = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].categories_[0]
    all_feat_names = numeric_features + [f'ocean_{c}' for c in ohe_cats]
    importance_list = sorted(
        [{'feature': f, 'importance': float(i)} for f, i in zip(all_feat_names, rf.feature_importances_)],
        key=lambda x: x['importance'], reverse=True
    )

    metadata = {
        'model_type': 'RandomForestRegressor',
        'metrics': {'rmse': round(rmse, 2), 'mae': round(mae, 2), 'r2': round(r2, 4)},
        'model_comparison': [
            {'model': 'Linear Regression (Baseline)', 'test_rmse': 69000, 'r2': 0.638},
            {'model': 'Random Forest',                'test_rmse': round(rmse, 2), 'r2': round(r2, 4)},
            {'model': 'Gradient Boosting',            'test_rmse': 58000, 'r2': 0.752},
        ],
        'feature_importance': importance_list,
        'data_stats': {
            'training_samples': len(X_train),
            'target_mean': round(float(y.mean()), 2),
        }
    }

    return pipeline, metadata

with st.spinner("⏳ กำลังโหลดและ train โมเดล (ครั้งแรกอาจใช้เวลา 1-2 นาที)..."):
    pipeline, metadata = train_model()

# ===== Sidebar =====
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดลนี้")
    st.write(f"**ประเภทโมเดล:** {metadata['model_type']}")
    st.write(f"**R² Score:** {metadata['metrics']['r2']:.4f}")
    st.write(f"**RMSE:** ${metadata['metrics']['rmse']:,.0f}")
    st.write(f"**MAE:** ${metadata['metrics']['mae']:,.0f}")
    st.write(f"**ข้อมูล train:** {metadata['data_stats']['training_samples']:,} ย่าน")

    st.divider()

    st.subheader("📊 Feature Importance (Top 5)")
    importance_data = metadata['feature_importance'][:5]
    for item in importance_data:
        feat = item['feature'].replace('ocean_', 'ทำเล: ')
        imp  = item['importance']
        st.write(f"**{feat}**")
        st.progress(float(imp) / float(importance_data[0]['importance']))

    st.divider()

    st.subheader("🏆 เปรียบเทียบโมเดล")
    for m in metadata['model_comparison']:
        st.write(f"**{m['model'].replace(' (Baseline)', '')}**")
        st.caption(f"RMSE: ${m['test_rmse']:,.0f} | R²: {m['r2']:.3f}")

    st.divider()

    st.subheader("⚠️ ข้อควรระวัง")
    st.warning(
        "ผลลัพธ์นี้เป็นการประมาณจาก AI เท่านั้น "
        "ข้อมูลมาจากสำมะโนประชากรปี 1990 "
        "กรุณาใช้ประกอบการตัดสินใจเท่านั้น"
    )

# ===== Header =====
st.title("🏠 ระบบประเมินราคาบ้านในแคลิฟอร์เนีย")
st.markdown("""
กรอกข้อมูลย่านที่อยู่อาศัยด้านล่าง ระบบจะประเมินราคาบ้านมัธยฐาน
โดยใช้โมเดล Machine Learning ที่ train จากข้อมูลสำมะโนประชากรแคลิฟอร์เนีย 20,640 ย่าน
""")

st.divider()

# ===== Input =====
st.subheader("📋 กรอกข้อมูลย่านที่อยู่อาศัย")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input(
        "ลองจิจูด (Longitude)",
        min_value=-124.5, max_value=-114.0,
        value=-119.0, step=0.01, format="%.2f",
        help="ช่วงแคลิฟอร์เนีย: -124.5 ถึง -114.0 (ยิ่งลบมาก = ยิ่งตะวันตก)"
    )
    latitude = st.number_input(
        "ละติจูด (Latitude)",
        min_value=32.5, max_value=42.0,
        value=36.0, step=0.01, format="%.2f",
        help="ช่วงแคลิฟอร์เนีย: 32.5 ถึง 42.0 (ยิ่งมาก = ยิ่งเหนือ)"
    )
    housing_median_age = st.number_input(
        "อายุมัธยฐานของบ้านในย่าน (ปี)",
        min_value=1, max_value=52,
        value=20, step=1,
        help="อายุเฉลี่ยของบ้านในย่านนั้น"
    )
    total_rooms = st.number_input(
        "จำนวนห้องรวมทั้งหมดในย่าน",
        min_value=1, max_value=40000,
        value=2000, step=100,
        help="รวมทุกห้องของทุกหลังในย่าน"
    )
    total_bedrooms = st.number_input(
        "จำนวนห้องนอนรวมทั้งหมดในย่าน",
        min_value=1, max_value=7000,
        value=400, step=10,
        help="รวมห้องนอนของทุกหลังในย่าน"
    )

with col2:
    population = st.number_input(
        "จำนวนประชากรในย่าน (คน)",
        min_value=1, max_value=40000,
        value=1200, step=100,
        help="จำนวนคนที่อาศัยอยู่ในย่านนั้น"
    )
    households = st.number_input(
        "จำนวนครัวเรือนในย่าน",
        min_value=1, max_value=7000,
        value=350, step=10,
        help="จำนวนบ้าน/หน่วยที่อยู่อาศัยในย่าน"
    )
    median_income = st.number_input(
        "รายได้มัธยฐานของครัวเรือน (หน่วย: หมื่นดอลลาร์)",
        min_value=0.5, max_value=15.0,
        value=4.0, step=0.1, format="%.1f",
        help="เช่น 5.0 = รายได้ $50,000/ปี | ค่าเฉลี่ยทั้ง dataset = 3.87"
    )
    ocean_proximity = st.selectbox(
        "ความใกล้ชิดกับทะเล",
        options=['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'],
        help="ระยะห่างจากมหาสมุทร"
    )
    ocean_desc = {
        '<1H OCEAN': '🌊 ห่างจากทะเลน้อยกว่า 1 ชั่วโมง',
        'INLAND':    '🏔️ อยู่ในแผ่นดิน ห่างไกลจากทะเล',
        'NEAR OCEAN':'🏖️ ใกล้มหาสมุทรแปซิฟิก',
        'NEAR BAY':  '⛵ ใกล้อ่าวซานฟรานซิสโก',
        'ISLAND':    '🏝️ อยู่บนเกาะ'
    }
    st.caption(ocean_desc[ocean_proximity])

st.divider()

# ===== Input Validation =====
warnings_list = []
if total_bedrooms > total_rooms:
    warnings_list.append("⚠️ จำนวนห้องนอนมากกว่าจำนวนห้องรวม — กรุณาตรวจสอบอีกครั้ง")
if households > total_bedrooms:
    warnings_list.append("⚠️ จำนวนครัวเรือนมากกว่าจำนวนห้องนอน — ดูผิดปกติ")
if population < households:
    warnings_list.append("⚠️ จำนวนประชากรน้อยกว่าจำนวนครัวเรือน — กรุณาตรวจสอบ")

for w in warnings_list:
    st.warning(w)

# ===== ปุ่มทำนาย =====
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button(
        "🔍 ประเมินราคาบ้าน",
        use_container_width=True,
        type="primary",
        disabled=len(warnings_list) > 0
    )

# ===== แสดงผล =====
if predict_button:

    rooms_per_household      = total_rooms / households
    bedrooms_ratio           = total_bedrooms / total_rooms
    population_per_household = population / households

    input_data = pd.DataFrame([{
        'longitude':               longitude,
        'latitude':                latitude,
        'housing_median_age':      housing_median_age,
        'total_rooms':             total_rooms,
        'total_bedrooms':          total_bedrooms,
        'population':              population,
        'households':              households,
        'median_income':           median_income,
        'ocean_proximity':         ocean_proximity,
        'rooms_per_household':     rooms_per_household,
        'bedrooms_ratio':          bedrooms_ratio,
        'population_per_household': population_per_household
    }])

    with st.spinner("กำลังประเมินราคา..."):
        predicted_price = pipeline.predict(input_data)[0]
        predicted_price = max(10000, min(predicted_price, 600000))

    st.subheader("📊 ผลการประเมินราคา")
    st.success(f"### 💰 ราคาบ้านมัธยฐานที่ประเมินได้\n# ${predicted_price:,.0f}")

    mae  = metadata['metrics']['mae']
    low  = max(0, predicted_price - mae)
    high = predicted_price + mae

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("ราคาต่ำสุด (ประมาณ)", f"${low:,.0f}")
    col_b.metric("ราคาที่ประเมิน",       f"${predicted_price:,.0f}")
    col_c.metric("ราคาสูงสุด (ประมาณ)", f"${high:,.0f}")
    st.caption(f"ช่วงราคาอ้างอิงจาก MAE ของโมเดล = ±${mae:,.0f}")

    avg_price = metadata['data_stats']['target_mean']
    diff      = predicted_price - avg_price
    diff_pct  = (diff / avg_price) * 100

    st.divider()
    st.write("**📈 เปรียบเทียบกับค่าเฉลี่ยทั้ง Dataset:**")
    if diff > 0:
        st.info(f"ราคาที่ประเมินสูงกว่าค่าเฉลี่ย **${avg_price:,.0f}** อยู่ **{diff_pct:.1f}%** (${diff:+,.0f})")
    else:
        st.info(f"ราคาที่ประเมินต่ำกว่าค่าเฉลี่ย **${avg_price:,.0f}** อยู่ **{abs(diff_pct):.1f}%** (${diff:+,.0f})")

    st.write("**ระดับราคาเทียบกับราคาสูงสุด ($500,000):**")
    st.progress(
        min(float(predicted_price) / 500000, 1.0),
        text=f"${predicted_price:,.0f} จาก $500,000"
    )

    with st.expander("📋 ดูข้อมูลที่กรอกและ Derived Features"):
        summary = {
            "ลองจิจูด":                       longitude,
            "ละติจูด":                        latitude,
            "อายุบ้านมัธยฐาน (ปี)":           housing_median_age,
            "จำนวนห้องรวม":                   f"{total_rooms:,}",
            "จำนวนห้องนอนรวม":                f"{total_bedrooms:,}",
            "จำนวนประชากร":                   f"{population:,}",
            "จำนวนครัวเรือน":                 f"{households:,}",
            "รายได้มัธยฐาน (หมื่น $)":        median_income,
            "ความใกล้ชิดทะเล":                ocean_proximity,
            "— ห้องต่อครัวเรือน (คำนวณ)":    f"{rooms_per_household:.2f}",
            "— สัดส่วนห้องนอน (คำนวณ)":      f"{bedrooms_ratio:.3f}",
            "— คนต่อครัวเรือน (คำนวณ)":      f"{population_per_household:.2f}"
        }
        st.dataframe(
            pd.DataFrame.from_dict(summary, orient="index", columns=["ค่า"]),
            use_container_width=True
        )
