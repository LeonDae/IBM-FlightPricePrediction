# ✈️ Flight Price Prediction

**Author:** Ditsu Kundu  
**Language:** Python 3.10+  
**Frontend:** Streamlit  
**Model:** Random Forest Regressor (R² = 0.942)

Predict Indian domestic flight fares using machine learning — trained on 452,088 real flight records covering 9 airlines and 7 major cities.

---

## 📁 Project Structure

```
DitsuBob/
│
├── DitsuKundu_FlightPricePrediction.ipynb   # Main notebook (EDA + training + evaluation)
├── app.py                                    # Streamlit web frontend
├── Cleaned_dataset.csv                       # Dataset (452,088 records)
├── requirements.txt                          # Python dependencies
│
├── model/
│   ├── rf_model.pkl                          # Trained Random Forest model
│   ├── label_encoders.pkl                    # Label encoders for categorical features
│   └── feature_columns.pkl                   # Ordered feature column names
│
└── plots/                                    # EDA & evaluation charts (auto-generated)
    ├── fare_distribution.png
    ├── fare_by_airline.png
    ├── fare_by_class.png
    ├── fare_by_stops.png
    ├── fare_by_day.png
    ├── fare_vs_daysleft.png
    ├── fare_vs_duration.png
    ├── route_heatmap.png
    ├── correlation_matrix.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    ├── residuals.png
    └── feature_importance.png
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `Cleaned_dataset.csv` | Link- 'https://docs.google.com/spreadsheets/d/1ZLKxfGmJV8WIVC_SVTnW1jdmV3ue9Cl0Sw9RB0HQ3H8/edit?usp=sharing'

| Records | 452,088 |
| Features | 13 |
| Target | `Fare` (INR) |

### Columns

| Column | Type | Description |
|---|---|---|
| `Date_of_journey` | Date | Journey date (YYYY-MM-DD) |
| `Journey_day` | Categorical | Day of the week (Monday–Sunday) |
| `Airline` | Categorical | Airline name |
| `Flight_code` | String | Flight identifier (dropped during training) |
| `Class` | Categorical | Travel class |
| `Source` | Categorical | Departure city |
| `Departure` | Categorical | Departure time slot |
| `Total_stops` | Categorical | Number of stops |
| `Arrival` | Categorical | Arrival time slot |
| `Destination` | Categorical | Arrival city |
| `Duration_in_hours` | Numeric | Total flight duration |
| `Days_left` | Numeric | Days between booking and journey |
| `Fare` | Numeric | **Target — ticket price (INR)** |

### Categorical Values

| Feature | Values |
|---|---|
| **Airline** | Air India, AirAsia, AkasaAir, AllianceAir, GO FIRST, Indigo, SpiceJet, StarAir, Vistara |
| **Class** | Economy, Premium Economy, Business, First |
| **Source / Destination** | Ahmedabad, Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai |
| **Total_stops** | non-stop, 1-stop, 2+-stop |
| **Departure / Arrival** | Before 6 AM, 6 AM – 12 PM, 12 PM – 6 PM, After 6 PM |
| **Journey_day** | Monday – Sunday |

---

## 🚀 Quick Start

### 1. Clone / open the project

```bash
cd DitsuBob
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Recommended) Run the notebook for full EDA + model training

```bash
jupyter notebook DitsuKundu_FlightPricePrediction.ipynb
```

Run all cells top-to-bottom. This will:
- Perform Exploratory Data Analysis and save 13 charts to `plots/`
- Train and compare 5 models
- Save the best model (`Random Forest`) to `model/`

### 4. Launch the Streamlit web app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

> **Note:** If `model/rf_model.pkl` does not exist (notebook not run yet), the app will offer to train the model automatically on first launch.

---

## 🧠 Machine Learning Pipeline

### Feature Engineering

| Raw Feature | Encoding |
|---|---|
| `Journey_day` | Ordinal (Monday=0 … Sunday=6) |
| `Total_stops` | Ordinal (non-stop=0, 1-stop=1, 2+-stop=2) |
| `Departure` / `Arrival` | Ordinal (Before 6 AM=0 … After 6 PM=3) |
| `Airline`, `Class`, `Source`, `Destination` | Label Encoding |
| `Date_of_journey`, `Flight_code` | Dropped |

### Models Compared

| Model | MAE (INR) | RMSE (INR) | R² |
|---|---|---|---|
| **Random Forest** ⭐ | **2,702** | **4,893** | **0.9420** |
| Decision Tree | 3,578 | 6,097 | 0.9099 |
| Gradient Boosting | — | — | — |
| Ridge Regression | 13,252 | 15,698 | 0.4028 |
| Linear Regression | 13,252 | 15,698 | 0.4028 |

### Best Model — Random Forest

```
n_estimators : 200
max_depth    : 15
n_jobs       : -1  (all CPU cores)
random_state : 42
test_size    : 20%
```

**Results on held-out test set (90,418 samples):**

| Metric | Score |
|---|---|
| R² | **0.9420** |
| MAE | **₹2,702** |
| RMSE | **₹4,893** |

### Top Features by Importance

1. `Class` — travel class has the strongest effect on fare
2. `Days_left` — earlier booking → lower fare
3. `Duration_in_hours` — longer flights cost more
4. `Airline` — carrier significantly affects price
5. `Total_stops` — more stops can increase price
6. `Source` / `Destination` — route matters
7. `Departure` / `Arrival` / `Journey_day` — time-based effects

---

## 🖥️ Streamlit App Features

- **Fare Prediction Form** — select airline, route, class, timing, stops, duration and days left
- **Live Prediction** — displays estimated fare in INR with journey summary
- **Smart Booking Tips** — contextual advice based on days left to journey
- **EDA Charts Viewer** — browse all 13 generated plots inside the app
- **Dataset Preview** — scrollable table of the first 100 records
- **Auto-train fallback** — trains the model on first run if `.pkl` files are missing

---

## 📦 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.32.0
joblib>=1.3.0
notebook>=7.0.0
ipykernel>=6.0.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📓 Notebook Sections

| # | Section |
|---|---|
| 1 | Import Libraries |
| 2 | Load & Inspect Data |
| 3 | Exploratory Data Analysis (7 charts) |
| 4 | Feature Engineering & Preprocessing |
| 5 | Model Training (5 models) |
| 6 | Model Evaluation (Actual vs Predicted, Residuals) |
| 7 | Feature Importance |
| 8 | Save Model & Encoders |

---

## 📝 License

This project is for academic and educational purposes.  
Dataset used for non-commercial learning only.

---

<div align="center">
  <sub>✈️ Flight Price Prediction &nbsp;|&nbsp; Ditsu Kundu &nbsp;|&nbsp; Python · scikit-learn · Streamlit</sub>
</div>
