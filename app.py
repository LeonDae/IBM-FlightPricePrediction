"""
Flight Price Prediction — Streamlit Frontend
Author: Ditsu Kundu
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import sys

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: #f0f7ff;
        border: 2px solid #1a73e8;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .stButton > button {
        background: #1a73e8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
    }
    .stButton > button:hover {
        background: #1558c0;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Constants matching notebook encoding
# ──────────────────────────────────────────────
DAY_MAP   = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
             'Friday':4,'Saturday':5,'Sunday':6}
STOPS_MAP = {'Non-stop':0,'1 Stop':1,'2+ Stops':2}
TIME_MAP  = {'Before 6 AM':0,'6 AM – 12 PM':1,'12 PM – 6 PM':2,'After 6 PM':3}

# Internal values used during training
TIME_MAP_INTERNAL  = {'Before 6 AM':0,'6 AM - 12 PM':1,'12 PM - 6 PM':2,'After 6 PM':3}
STOPS_MAP_INTERNAL = {'Non-stop': 'non-stop', '1 Stop': '1-stop', '2+ Stops': '2+-stop'}

AIRLINES     = sorted(['SpiceJet','Indigo','GO FIRST','Vistara',
                        'Air India','AirAsia','StarAir','AllianceAir','AkasaAir'])
CITIES       = sorted(['Delhi','Mumbai','Bangalore','Kolkata',
                        'Hyderabad','Chennai','Ahmedabad'])
CLASSES      = ['Economy','Premium Economy','Business','First']
DAYS         = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
DEPARTURES   = ['Before 6 AM','6 AM – 12 PM','12 PM – 6 PM','After 6 PM']
ARRIVALS     = ['Before 6 AM','6 AM – 12 PM','12 PM – 6 PM','After 6 PM']
STOPS_LABELS = ['Non-stop','1 Stop','2+ Stops']

# ──────────────────────────────────────────────
# Load model (train if not found)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path    = 'model/rf_model.pkl'
    encoders_path = 'model/label_encoders.pkl'
    columns_path  = 'model/feature_columns.pkl'

    if not (os.path.exists(model_path) and
            os.path.exists(encoders_path) and
            os.path.exists(columns_path)):
        return None, None, None

    model    = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    columns  = joblib.load(columns_path)
    return model, encoders, columns


def train_model_inline():
    """Train & save model directly inside the app (no notebook needed)."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    with st.spinner("🔧 Training model for the first time — please wait ~2 min..."):
        df = pd.read_csv('Cleaned_dataset.csv')
        df_model = df.copy()
        df_model.drop(columns=['Date_of_journey', 'Flight_code'], inplace=True)

        day_map_   = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
                      'Friday':4,'Saturday':5,'Sunday':6}
        stops_map_ = {'non-stop':0,'1-stop':1,'2+-stop':2}
        time_map_  = {'Before 6 AM':0,'6 AM - 12 PM':1,'12 PM - 6 PM':2,'After 6 PM':3}

        df_model['Journey_day'] = df_model['Journey_day'].map(day_map_)
        df_model['Total_stops'] = df_model['Total_stops'].map(stops_map_)
        df_model['Departure']   = df_model['Departure'].map(time_map_)
        df_model['Arrival']     = df_model['Arrival'].map(time_map_)

        label_encoders = {}
        for col in ['Airline', 'Class', 'Source', 'Destination']:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le

        X = df_model.drop(columns=['Fare'])
        y = df_model['Fare']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                      n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)

        os.makedirs('model', exist_ok=True)
        joblib.dump(model,            'model/rf_model.pkl')
        joblib.dump(label_encoders,   'model/label_encoders.pkl')
        joblib.dump(list(X.columns),  'model/feature_columns.pkl')
        st.success("✅ Model trained and saved successfully!")

    st.cache_resource.clear()
    st.rerun()


# ──────────────────────────────────────────────
# Prediction helper
# ──────────────────────────────────────────────
def predict_fare(model, encoders, columns,
                 airline, flight_class, source, destination,
                 departure_label, arrival_label, stops_label,
                 journey_day, duration_hours, days_left):

    dep_internal  = departure_label.replace('–', '-')
    arr_internal  = arrival_label.replace('–', '-')
    stops_raw     = STOPS_MAP_INTERNAL[stops_label]

    dep_enc   = TIME_MAP_INTERNAL[dep_internal]
    arr_enc   = TIME_MAP_INTERNAL[arr_internal]
    stops_enc = STOPS_MAP[stops_label]   # 0 / 1 / 2
    day_enc   = DAY_MAP[journey_day]

    airline_enc = encoders['Airline'].transform([airline])[0]
    class_enc   = encoders['Class'].transform([flight_class])[0]
    source_enc  = encoders['Source'].transform([source])[0]
    dest_enc    = encoders['Destination'].transform([destination])[0]

    row = {
        'Journey_day':       day_enc,
        'Airline':           airline_enc,
        'Class':             class_enc,
        'Source':            source_enc,
        'Departure':         dep_enc,
        'Total_stops':       stops_enc,
        'Arrival':           arr_enc,
        'Destination':       dest_enc,
        'Duration_in_hours': duration_hours,
        'Days_left':         days_left,
    }

    X = pd.DataFrame([row])[columns]
    prediction = model.predict(X)[0]
    return max(0, prediction)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.2rem;">✈️ Flight Price Predictor</h1>
    <p style="margin:0.4rem 0 0 0; opacity:0.9; font-size:1rem;">
        Predict Indian domestic flight fares using Machine Learning (Random Forest)
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
model, encoders, columns = load_model()

if model is None:
    st.warning("⚠️ No trained model found. Click below to train one from the dataset.")
    if st.button("🚀 Train Model Now"):
        train_model_inline()
    st.stop()

# ──────────────────────────────────────────────
# Sidebar — Model info
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/airplane-mode-on.png", width=80)
    st.title("About")
    st.markdown("""
**Model:** Random Forest Regressor  
**Dataset:** 452,088 Indian domestic flights  
**Features:** Airline, Class, Route, Timing, Duration, Days Left  
**Target:** Fare (INR)
    """)
    st.divider()
    st.markdown("**Airlines covered:**")
    for a in AIRLINES:
        st.markdown(f"- {a}")
    st.divider()
    st.markdown("**Routes covered:**")
    st.markdown(", ".join(CITIES))

# ──────────────────────────────────────────────
# Main input form
# ──────────────────────────────────────────────
st.subheader("🔍 Enter Flight Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        airline   = st.selectbox("✈️ Airline", AIRLINES)
        source    = st.selectbox("🛫 Source City", CITIES)
        departure = st.selectbox("🕐 Departure Time", DEPARTURES)

    with col2:
        flight_class = st.selectbox("💺 Travel Class", CLASSES)
        destination  = st.selectbox("🛬 Destination City", CITIES)
        arrival      = st.selectbox("🕐 Arrival Time", ARRIVALS)

    with col3:
        stops        = st.selectbox("🔄 Number of Stops", STOPS_LABELS)
        journey_day  = st.selectbox("📅 Day of Journey", DAYS)
        duration     = st.slider("⏱️ Flight Duration (hours)", 0.5, 20.0, 2.5, 0.25)

    days_left = st.slider("📆 Days Left to Journey", 1, 365, 30)

    submit = st.form_submit_button("💰 Predict Fare")

# ──────────────────────────────────────────────
# Prediction output
# ──────────────────────────────────────────────
if submit:
    if source == destination:
        st.error("⚠️ Source and Destination cannot be the same city.")
    else:
        try:
            predicted_fare = predict_fare(
                model, encoders, columns,
                airline, flight_class, source, destination,
                departure, arrival, stops,
                journey_day, duration, days_left
            )

            st.markdown(f"""
<div class="prediction-card">
    <h2 style="color:#1a73e8; margin:0;">Estimated Fare</h2>
    <h1 style="font-size:3rem; margin:0.5rem 0; color:#0d47a1;">
        ₹{predicted_fare:,.0f}
    </h1>
    <p style="color:#555; margin:0;">
        {airline} &nbsp;|&nbsp; {source} → {destination} &nbsp;|&nbsp;
        {flight_class} &nbsp;|&nbsp; {stops}
    </p>
</div>
""", unsafe_allow_html=True)

            # Quick breakdown metrics
            st.markdown("### 📊 Journey Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Route",    f"{source} → {destination}")
            m2.metric("Duration", f"{duration:.2f} hrs")
            m3.metric("Days Left", str(days_left))
            m4.metric("Stops",    stops)

            # Fare insight
            st.markdown("---")
            if days_left <= 7:
                st.info("💡 **Tip:** Booking very close to the journey date usually increases fares. "
                        "Consider booking earlier for a better price.")
            elif days_left >= 60:
                st.success("✅ **Great timing!** Booking 60+ days in advance typically gets you the best fares.")
            else:
                st.info("💡 **Tip:** Fares are usually lowest when booked 30–60 days in advance.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ──────────────────────────────────────────────
# EDA Section (show saved plots if available)
# ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📈 Exploratory Data Analysis — View Charts"):
    plot_files = {
        "Fare Distribution":         "plots/fare_distribution.png",
        "Fare by Airline":           "plots/fare_by_airline.png",
        "Fare by Class":             "plots/fare_by_class.png",
        "Fare vs Days Left":         "plots/fare_vs_daysleft.png",
        "Fare by Stops":             "plots/fare_by_stops.png",
        "Fare vs Duration":          "plots/fare_vs_duration.png",
        "Route Heatmap":             "plots/route_heatmap.png",
        "Fare by Day of Week":       "plots/fare_by_day.png",
        "Correlation Matrix":        "plots/correlation_matrix.png",
        "Model Comparison":          "plots/model_comparison.png",
        "Actual vs Predicted":       "plots/actual_vs_predicted.png",
        "Residuals Distribution":    "plots/residuals.png",
        "Feature Importance":        "plots/feature_importance.png",
    }

    available = {k: v for k, v in plot_files.items() if os.path.exists(v)}

    if not available:
        st.info("📌 Run the Jupyter notebook `DitsuKundu_FlightPricePrediction.ipynb` "
                "first to generate EDA plots, then refresh this page.")
    else:
        for title, path in available.items():
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)

# ──────────────────────────────────────────────
# Dataset preview
# ──────────────────────────────────────────────
with st.expander("📂 Dataset Preview (first 100 rows)"):
    try:
        df_preview = pd.read_csv('Cleaned_dataset.csv', nrows=100)
        st.dataframe(df_preview, use_container_width=True)
        st.caption("Showing first 100 of 452,088 records.")
    except Exception as e:
        st.error(f"Could not load dataset: {e}")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "✈️ Flight Price Prediction &nbsp;|&nbsp; Author: <b>Ditsu Kundu</b> &nbsp;|&nbsp; "
    "Model: Random Forest &nbsp;|&nbsp; Dataset: 452K Indian domestic flights"
    "</div>",
    unsafe_allow_html=True
)
