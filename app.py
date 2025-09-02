import streamlit as st
import pickle
import numpy as np

# ------------------------
# تحميل الموديل و LabelEncoder
# ------------------------
with open("model.pkl", "rb") as f:
    model, le = pickle.load(f)   # لازم يكون متخزن الاتنين (model + le)

st.title("💄 Makeup Rating Predictor")
st.write("أدخل البيانات عشان نعمل توقع للتقييم المتوقع للمنتج 👇")

# ------------------------
# تحديد أشهر 5 براندز
# ------------------------
top_brands = ["Maybelline", "L'Oreal", "MAC", "Sephora", "Clinique"]

# ------------------------
# إدخال البيانات من المستخدم
# ------------------------
price = st.number_input("💰 دخل السعر", min_value=1.0, step=1.0)

# بدل الـ slider خليها number_input عادي
rating = st.number_input("⭐ دخل التقييم الفردي", min_value=1.0, max_value=5.0, step=0.1)

brand_name = st.selectbox("🏷️ اختار البراند", top_brands)

# ------------------------
# تحويل البراند لرقم بالـ LabelEncoder
# ------------------------
if brand_name in le.classes_:
    brand_encoded = le.transform([brand_name])[0]
else:
    st.error("⚠️ البراند مش موجود في الموديل المدرب!")
    brand_encoded = None

# ------------------------
# التوقع
# ------------------------
if st.button("🔮 توقع") and brand_encoded is not None:
    features = np.array([[price, brand_encoded, rating]])
    prediction = model.predict(features)

    # قص التوقع بين 1 و 5
    prediction = max(1.0, min(5.0, prediction[0]))

    st.success(f"📊 التقييم المتوقع للمنتج: {prediction:.2f}")
