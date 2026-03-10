# California-housing-price-prediction
# 🏡 California Housing Price Prediction

ระบบประเมินราคาบ้านในแคลิฟอร์เนียด้วย Machine Learning  
Built with **Python**, **scikit-learn**, and **Streamlit**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alifornia-housing-price-prediction-6huveizsk4gusa5qjhdcp5.streamlit.app/)

---

## 📌 ภาพรวมของโปรเจค

โปรเจคนี้สร้างระบบทำนายราคาบ้านมัธยฐานในย่านต่างๆ ของรัฐแคลิฟอร์เนีย สหรัฐอเมริกา  
โดยใช้ข้อมูลจากสำมะโนประชากรปี 1990 จำนวน **20,640 ย่าน**

### ทำไมปัญหานี้ถึงน่าสนใจ?
ราคาอสังหาริมทรัพย์ขึ้นอยู่กับปัจจัยหลายอย่างพร้อมกัน ทั้งทำเลที่ตั้ง รายได้ของคนในย่าน และความใกล้ชิดกับทะเล ความสัมพันธ์เหล่านี้ซับซ้อนและไม่เป็นเส้นตรง ซึ่ง ML สามารถเรียนรู้ได้ดีกว่าสูตรคำนวณแบบเดิม

---

## 🗂️ โครงสร้างไฟล์

```
alifornia-housing-price-prediction/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── Housing_Price_Prediction.ipynb  # Colab notebook (EDA + Model training)
└── README.md
```

---

## 📊 Dataset

| รายละเอียด | ค่า |
|-----------|-----|
| ที่มา | California Census 1990 |
| จำนวนแถว | 20,640 ย่าน |
| จำนวน Features | 9 (+ 3 derived) |
| Target | `median_house_value` (USD) |

### Features ที่ใช้

| Feature | ความหมาย |
|---------|-----------|
| `longitude` / `latitude` | พิกัดที่ตั้ง |
| `housing_median_age` | อายุมัธยฐานของบ้านในย่าน |
| `total_rooms` / `total_bedrooms` | จำนวนห้องรวมในย่าน |
| `population` / `households` | ประชากรและครัวเรือน |
| `median_income` | รายได้มัธยฐาน (หน่วย: หมื่น USD) |
| `ocean_proximity` | ความใกล้ชิดกับทะเล |
| `rooms_per_household` ⭐ | ห้องเฉลี่ยต่อครัวเรือน (derived) |
| `bedrooms_ratio` ⭐ | สัดส่วนห้องนอน (derived) |
| `population_per_household` ⭐ | คนต่อครัวเรือน (derived) |

---

## 🤖 Model Development

### โมเดลที่เปรียบเทียบ

| Model | Test RMSE | R² |
|-------|-----------|-----|
| Linear Regression (Baseline) | ~$69,000 | 0.638 |
| **Random Forest** ✅ | **~$50,000** | **0.806** |
| Gradient Boosting | ~$58,000 | 0.752 |

### Pipeline
```
Input → SimpleImputer (median) → StandardScaler
      → OneHotEncoder (ocean_proximity)
      → RandomForestRegressor
      → Predicted Price
```

### Hyperparameter Tuning
ใช้ `RandomizedSearchCV` กับ `KFold (k=3)` เพื่อหา best parameters:
- `n_estimators`: 100
- `max_depth`: 15  
- `min_samples_split`: 5

---

## 🚀 วิธีรันบนเครื่อง

```bash
# 1. Clone repo
git clone https://github.com/your-username/alifornia-housing-price-prediction.git
cd alifornia-housing-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run app
streamlit run app.py
```

---

## 🌐 Live Demo

👉 [เปิด App](https://alifornia-housing-price-prediction-6huveizsk4gusa5qjhdcp5.streamlit.app/)

---

## 📓 Notebook

Colab notebook ครอบคลุม:
- ✅ EDA พร้อม 5 กราฟ (distribution, correlation heatmap, scatter plots, geographic map)
- ✅ Feature Engineering
- ✅ เปรียบเทียบ 3 โมเดล
- ✅ Hyperparameter Tuning ด้วย RandomizedSearchCV
- ✅ Residual Analysis & Feature Importance

---

## 👩‍💻 ผู้จัดทำ

| รายละเอียด | ข้อมูล |
|-----------|--------|
| ชื่อ | เพชรนภากร พลนิกร |
| รหัสนิสิต | 67160358 |
| วิชา | Data Science |
| อาจารย์ | ผศ.ดร.สุภาวดี ศรีคำดี|
