import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ============================
# Load Files
# ============================

model = joblib.load("flight_delay_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
dropdown = joblib.load("dropdown_values.pkl")

# ============================
# Page Configuration
# ============================

st.set_page_config(
    page_title="AeroPredict",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AeroPredict")
st.subheader("Smart Flight Delay Prediction Using Machine Learning")

st.markdown("---")

# ============================
# Input Section
# ============================

col1, col2 = st.columns(2)

with col1:

    flight_date = st.date_input(
        "Flight Date",
        value=date(2024, 1, 1)
    )

    carrier = st.selectbox(
        "Airline",
        dropdown["airlines"]
    )

    origin = st.selectbox(
        "Origin Airport",
        dropdown["origins"]
    )

    origin_city = st.selectbox(
        "Origin City",
        dropdown["origin_cities"]
    )

    origin_state = st.selectbox(
        "Origin State",
        dropdown["origin_states"]
    )

    dep_time = st.number_input(
        "Scheduled Departure Time (HHMM)",
        0,
        2359,
        900
    )

    cancelled = st.selectbox(
        "Cancelled",
        [0, 1]
    )

with col2:

    destination = st.selectbox(
        "Destination Airport",
        dropdown["destinations"]
    )

    destination_city = st.selectbox(
        "Destination City",
        dropdown["destination_cities"]
    )

    destination_state = st.selectbox(
        "Destination State",
        dropdown["destination_states"]
    )

    arr_time = st.number_input(
        "Scheduled Arrival Time (HHMM)",
        0,
        2359,
        1200
    )

    elapsed = st.number_input(
        "Scheduled Flight Duration (Minutes)",
        20,
        1000,
        180
    )

    distance = st.number_input(
        "Distance (Miles)",
        1,
        6000,
        1000
    )

    diverted = st.selectbox(
        "Diverted",
        [0, 1]
    )

# ============================
# Feature Engineering
# ============================

year = flight_date.year
month = flight_date.month
day = flight_date.day

day_of_week = flight_date.isoweekday()

flight_month = month
flight_day = day
flight_dayofweek = day_of_week
flight_week = flight_date.isocalendar().week

departure_hour = dep_time // 100
arrival_hour = arr_time // 100

if 5 <= departure_hour < 12:
    departure_period = "Morning"
elif 12 <= departure_hour < 17:
    departure_period = "Afternoon"
elif 17 <= departure_hour < 21:
    departure_period = "Evening"
else:
    departure_period = "Night"

is_weekend = 1 if day_of_week in [6, 7] else 0

route = origin + "_" + destination

if distance <= 500:
    distance_category = "Short"
elif distance <= 1500:
    distance_category = "Medium"
else:
    distance_category = "Long"

if elapsed <= 120:
    duration_category = "Short"
elif elapsed <= 240:
    duration_category = "Medium"
else:
    duration_category = "Long"

# ============================
# Prediction
# ============================

if st.button("Predict Flight Delay"):

    input_df = pd.DataFrame({

        "year":[year],
        "month":[month],
        "day_of_month":[day],
        "day_of_week":[day_of_week],
        "op_unique_carrier":[carrier],
        "origin":[origin],
        "origin_city_name":[origin_city],
        "origin_state_nm":[origin_state],
        "dest":[destination],
        "dest_city_name":[destination_city],
        "dest_state_nm":[destination_state],
        "crs_dep_time":[dep_time],
        "crs_arr_time":[arr_time],
        "cancelled":[cancelled],
        "diverted":[diverted],
        "crs_elapsed_time":[elapsed],
        "distance":[distance],
        "flight_month":[flight_month],
        "flight_day":[flight_day],
        "flight_dayofweek":[flight_dayofweek],
        "flight_week":[flight_week],
        "departure_hour":[departure_hour],
        "arrival_hour":[arrival_hour],
        "departure_period":[departure_period],
        "is_weekend":[is_weekend],
        "route":[route],
        "distance_category":[distance_category],
        "duration_category":[duration_category]

    })

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Flight is likely to be Delayed")
    else:
        st.success("✅ Flight is likely to be On Time")

    st.metric(
        "Delay Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.markdown("### Flight Summary")

    st.write(f"**Airline:** {carrier}")
    st.write(f"**Route:** {origin} ➜ {destination}")
    st.write(f"**Departure Time:** {dep_time}")
    st.write(f"**Arrival Time:** {arr_time}")
    st.write(f"**Distance:** {distance} miles")