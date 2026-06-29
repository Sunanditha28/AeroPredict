import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="AeroPredict",
    page_icon="✈️",
    layout="wide"
)

# ============================================
# Load Files
# ============================================

model = joblib.load("flight_delay_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")
dropdown = joblib.load("dropdown_values.pkl")
train_columns = joblib.load("train_columns.pkl")

# ============================================
# Title
# ============================================

st.title("✈️ AeroPredict")
st.markdown("## Smart Flight Delay Prediction Using Machine Learning")

st.markdown("---")

# ============================================
# User Inputs
# ============================================

col1, col2 = st.columns(2)

with col1:

    flight_date = st.date_input(
        "Flight Date",
        value=date.today()
    )

    airline = st.selectbox(
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

    departure_time = st.number_input(
        "Scheduled Departure Time (HHMM)",
        min_value=0,
        max_value=2359,
        value=900
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

    arrival_time = st.number_input(
        "Scheduled Arrival Time (HHMM)",
        min_value=0,
        max_value=2359,
        value=1200
    )

    elapsed_time = st.number_input(
        "Scheduled Flight Duration (Minutes)",
        min_value=30,
        max_value=1000,
        value=180
    )

distance = st.number_input(
    "Distance (Miles)",
    min_value=1,
    max_value=6000,
    value=1000
)

cancelled = st.selectbox(
    "Cancelled",
    [0,1]
)

diverted = st.selectbox(
    "Diverted",
    [0,1]
)

st.markdown("---")

# ============================================
# Feature Engineering
# ============================================

flight_year = flight_date.year
flight_month = flight_date.month
flight_day = flight_date.day
flight_dayofweek = flight_date.weekday()

flight_week = flight_date.isocalendar().week

departure_hour = departure_time // 100
arrival_hour = arrival_time // 100

day_of_week = flight_dayofweek + 1

is_weekend = 1 if day_of_week in [6,7] else 0

route = f"{str(origin).strip()}_{str(destination).strip()}"

# Departure Period

if departure_hour < 12:
    departure_period = "Morning"

elif departure_hour < 17:
    departure_period = "Afternoon"

elif departure_hour < 21:
    departure_period = "Evening"

else:
    departure_period = "Night"

# Distance Category

if distance <= 500:
    distance_category = "Short"

elif distance <= 1500:
    distance_category = "Medium"

else:
    distance_category = "Long"

# Duration Category

if elapsed_time <= 120:
    duration_category = "Short"

elif elapsed_time <= 240:
    duration_category = "Medium"

else:
    duration_category = "Long"

# ============================================
# Prediction Button
# ============================================

if st.button("Predict Flight Delay"):

    input_df = pd.DataFrame({

        "year":[flight_year],

        "month":[flight_month],

        "day_of_month":[flight_day],

        "day_of_week":[day_of_week],

        "op_unique_carrier":[airline],

        "origin":[origin],

        "origin_city_name":[origin_city],

        "origin_state_nm":[origin_state],

        "dest":[destination],

        "dest_city_name":[destination_city],

        "dest_state_nm":[destination_state],

        "crs_dep_time":[departure_time],

        "crs_arr_time":[arrival_time],

        "cancelled":[cancelled],

        "diverted":[diverted],

        "crs_elapsed_time":[elapsed_time],

        "distance":[distance],

        "flight_year":[flight_year],

        "flight_month":[flight_month],

        "flight_day":[flight_day],

        "flight_dayofweek":[flight_dayofweek],

        "flight_week":[flight_week],

        "departure_hour":[departure_hour],

        "arrival_hour":[arrival_hour],

        "is_weekend":[is_weekend],

        "route":[route],

        "departure_period":[departure_period],

        "distance_category":[distance_category],

        "duration_category":[duration_category]

    })

    # ============================================
    # Label Encoding
    # ============================================

    label_columns = [
        "op_unique_carrier",
        "origin",
        "origin_city_name",
        "origin_state_nm",
        "dest",
        "dest_city_name",
        "dest_state_nm",
        "route"
    ]

    st.write("Generated Route:", route)
    st.write("Origin:", origin)
    st.write("Destination:", destination)

    for col in label_columns:

        value = str(input_df.loc[0, col]).strip()
    
        if value not in encoders[col].classes_:
            st.error(f"{value} is not available in training data for '{col}'.")
            st.stop()
    
        input_df[col] = encoders[col].transform([value])

    # ============================================
    # One-Hot Encoding
    # ============================================

    input_df = pd.get_dummies(
        input_df,
        columns=[
            "departure_period",
            "distance_category",
            "duration_category"
        ],
        drop_first=True,
        dtype=int
    )

    # ============================================
    # Match Training Columns
    # ============================================

    input_df = input_df.reindex(
        columns=train_columns,
        fill_value=0
    )

    # ============================================
    # Feature Scaling
    # ============================================

    input_scaled = scaler.transform(input_df)

    # ============================================
    # Prediction
    # ============================================

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Flight is likely to be Delayed")
    else:
        st.success("✅ Flight is likely to be On Time")

    st.metric(
        label="Delay Probability",
        value=f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.markdown("## Flight Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Airline",
            "Origin",
            "Destination",
            "Departure Time",
            "Arrival Time",
            "Distance",
            "Duration"
        ],
        "Value": [
            airline,
            origin,
            destination,
            departure_time,
            arrival_time,
            f"{distance} Miles",
            f"{elapsed_time} Minutes"
        ]
    })

    st.table(summary)
