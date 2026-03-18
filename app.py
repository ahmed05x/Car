import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================== تحميل ==================
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ================== تصميم ==================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
}
h1 {
    color: #2c3e50;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Car Price Prediction App")
st.write("Predict car price easily using Machine Learning")

# ================== Inputs ==================

brand = st.selectbox("Brand", encoders["Brand"].classes_)
location = st.selectbox("Location", encoders["Location"].classes_)
fuel = st.selectbox("Fuel Type", encoders["Fuel_Type"].classes_)
owner = st.selectbox("Owner Type", encoders["Owner_Type"].classes_)
transmission = st.selectbox("Transmission", encoders["Transmission"].classes_)  

engine = st.number_input("Engine (CC)", 500, 5000, step=100)
power = st.number_input("Power (bhp)", 20, 500)
mileage = st.number_input("Mileage (km/l)", 5.0, 40.0)
car_age = st.slider("Car Age", 0, 20)

# ================== Feature Engineering ==================

power_engine_ratio = power / engine if engine != 0 else 0
mileage_engine_ratio = mileage / engine if engine != 0 else 0

# ================== Encoding ==================

brand_encoded = encoders["Brand"].transform([brand])[0]
location_encoded = encoders["Location"].transform([location])[0]
fuel_encoded = encoders["Fuel_Type"].transform([fuel])[0]
owner_encoded = encoders["Owner_Type"].transform([owner])[0]
transmission_encoded = encoders["Transmission"].transform([transmission])[0]

# ================== DataFrame ==================

input_df = pd.DataFrame({
    "Engine":[engine],
    "Power":[power],
    "Mileage":[mileage],
    "Car_Age":[car_age],
    "Owner_Type":[owner_encoded],
    "Power_Engine_Ratio":[power_engine_ratio],
    "Mileage_Engine_Ratio":[mileage_engine_ratio],
    "Brand":[brand_encoded],
    "Location":[location_encoded],
    "Fuel_Type":[fuel_encoded],
    "Transmission":[transmission_encoded]
})

# ترتيب الأعمدة
input_df = input_df.reindex(columns=columns)

# ================== Prediction ==================

if st.button("Predict Price 💰"):
    rate = 0.56

    with st.spinner("Predicting... 🔍"):
        prediction = model.predict(input_df)[0]

        # لو عامل log transform
        prediction = np.exp(prediction)

    st.success("Prediction Done ✅")

    st.metric(
        label="Estimated Car Price",
        value=f"{(prediction * rate * 100000):,.0f} EGP 💵"
    )
