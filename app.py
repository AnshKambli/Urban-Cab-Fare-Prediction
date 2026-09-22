"""
Urban Cab Fare — Real-Time Fare & Surge Predictor
===================================================
A Streamlit app that trains the regression (Final_Fare) and classification
(High_Surge) models from the EDA notebook and serves live predictions.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Place `UrbanCabFare.csv` in the same folder as this file, or upload it
from the sidebar when the app starts.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Urban Cab Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "UrbanCabFare.csv")


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as the EDA notebook — keep these in sync."""
    df = df.copy()
    if "Driver_Name" in df.columns:
        df["Driver_Name"] = df["Driver_Name"].fillna("Unknown")

    df["Ride_Date"] = pd.to_datetime(df["Ride_Date"], format="%d-%m-%Y", errors="coerce")
    df["Is_Weekend"] = df["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)

    def time_bucket(h):
        if 5 <= h < 12:
            return "Morning"
        elif 12 <= h < 17:
            return "Afternoon"
        elif 17 <= h < 21:
            return "Evening"
        else:
            return "Night"

    df["Time_of_Day"] = df["Hour_of_Day"].apply(time_bucket)
    df["High_Surge"] = (df["Surge_Multiplier"] >= 1.5).astype(int)
    return df


FEATURE_COLS = [
    "City", "Hour_of_Day", "Day_of_Week", "Time_of_Day", "Is_Weekend",
    "Distance_km", "Traffic_Level", "Type_of_vehicle",
    "No_of_active_drivers", "Ride_Requests", "Demand_Supply_Ratio", "Trip_Duration",
]
CATEGORICAL_COLS = ["City", "Day_of_Week", "Time_of_Day", "Traffic_Level", "Type_of_vehicle"]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]


def make_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_COLS),
    ])


# ---------------------------------------------------------------------------
# Model training (cached — runs once per uploaded dataset)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models on your data...")
def train_models(df: pd.DataFrame):
    df = engineer_features(df)
    X = df[FEATURE_COLS]

    # --- Regression: Final_Fare ---
    y_reg = df["Final_Fare"]
    Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2, random_state=RANDOM_STATE)

    lr_pipe = Pipeline([("prep", make_preprocessor()), ("model", LinearRegression())])
    lr_pipe.fit(Xtr, ytr)
    lr_pred = lr_pipe.predict(Xte)

    rf_reg_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE)),
    ])
    rf_reg_pipe.fit(Xtr, ytr)
    rf_reg_pred = rf_reg_pipe.predict(Xte)

    reg_metrics = pd.DataFrame([
        {"Model": "Linear Regression",
         "MAE": mean_absolute_error(yte, lr_pred),
         "RMSE": np.sqrt(mean_squared_error(yte, lr_pred)),
         "R2": r2_score(yte, lr_pred)},
        {"Model": "Random Forest Regressor",
         "MAE": mean_absolute_error(yte, rf_reg_pred),
         "RMSE": np.sqrt(mean_squared_error(yte, rf_reg_pred)),
         "R2": r2_score(yte, rf_reg_pred)},
    ])

    # --- Classification: High_Surge ---
    y_clf = df["High_Surge"]
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )

    logreg_pipe = Pipeline([("prep", make_preprocessor()),
                             ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    logreg_pipe.fit(Xtr_c, ytr_c)
    logreg_pred = logreg_pipe.predict(Xte_c)
    logreg_proba = logreg_pipe.predict_proba(Xte_c)[:, 1]

    rf_clf_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)),
    ])
    rf_clf_pipe.fit(Xtr_c, ytr_c)
    rf_clf_pred = rf_clf_pipe.predict(Xte_c)
    rf_clf_proba = rf_clf_pipe.predict_proba(Xte_c)[:, 1]

    clf_metrics = pd.DataFrame([
        {"Model": "Logistic Regression",
         "Accuracy": accuracy_score(yte_c, logreg_pred),
         "Precision": precision_score(yte_c, logreg_pred),
         "Recall": recall_score(yte_c, logreg_pred),
         "F1": f1_score(yte_c, logreg_pred),
         "ROC-AUC": roc_auc_score(yte_c, logreg_proba)},
        {"Model": "Random Forest Classifier",
         "Accuracy": accuracy_score(yte_c, rf_clf_pred),
         "Precision": precision_score(yte_c, rf_clf_pred),
         "Recall": recall_score(yte_c, rf_clf_pred),
         "F1": f1_score(yte_c, rf_clf_pred),
         "ROC-AUC": roc_auc_score(yte_c, rf_clf_proba)},
    ])

    return {
        "df": df,
        "fare_model": rf_reg_pipe,          # best regressor -> used for live predictions
        "fare_model_linear": lr_pipe,
        "surge_model": rf_clf_pipe,          # best classifier -> used for live predictions
        "surge_model_logistic": logreg_pipe,
        "reg_metrics": reg_metrics,
        "clf_metrics": clf_metrics,
    }


# ---------------------------------------------------------------------------
# Sidebar — data source + navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🚕 Urban Cab Fare")
st.sidebar.caption("Fare & surge prediction, powered by your trip data")

uploaded = st.sidebar.file_uploader("Upload UrbanCabFare.csv", type=["csv"])
data_source = uploaded if uploaded is not None else (DEFAULT_PATH if os.path.exists(DEFAULT_PATH) else None)

if data_source is None:
    st.sidebar.warning("No dataset found. Upload UrbanCabFare.csv to get started.")
    st.title("🚕 Urban Cab Fare Predictor")
    st.info(
        "👋 Upload **UrbanCabFare.csv** from the sidebar to train the models and "
        "unlock live fare & surge predictions."
    )
    st.stop()

raw_df = load_data(data_source)
bundle = train_models(raw_df)
df = bundle["df"]

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "💰 Fare Predictor", "⚡ Surge Predictor", "📊 Data Insights", "🧪 Model Performance"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Trained on **{len(df):,}** historical rides")

CITIES = sorted(df["City"].unique())
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TRAFFIC_LEVELS = ["Low", "Medium", "High", "Jam"]
VEHICLES = sorted(df["Type_of_vehicle"].unique())


def build_single_row(city, hour, day, distance, traffic, vehicle,
                      active_drivers, ride_requests, dsr, duration):
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
    if 5 <= hour < 12:
        tod = "Morning"
    elif 12 <= hour < 17:
        tod = "Afternoon"
    elif 17 <= hour < 21:
        tod = "Evening"
    else:
        tod = "Night"

    return pd.DataFrame([{
        "City": city, "Hour_of_Day": hour, "Day_of_Week": day, "Time_of_Day": tod,
        "Is_Weekend": is_weekend, "Distance_km": distance, "Traffic_Level": traffic,
        "Type_of_vehicle": vehicle, "No_of_active_drivers": active_drivers,
        "Ride_Requests": ride_requests, "Demand_Supply_Ratio": dsr, "Trip_Duration": duration,
    }])


# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("🚕 Urban Cab Fare Predictor")
    st.markdown("Live fare estimation and surge-risk prediction, trained on historical ride data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical rides", f"{len(df):,}")
    c2.metric("Cities covered", df["City"].nunique())
    c3.metric("Avg. Final Fare", f"₹{df['Final_Fare'].mean():.0f}")
    c4.metric("High-surge rides", f"{df['High_Surge'].mean()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fare model — held-out test performance")
        st.dataframe(bundle["reg_metrics"].set_index("Model").round(3), use_container_width=True)
        st.caption("Random Forest Regressor is used for live predictions (lower error, higher R²).")
    with col2:
        st.subheader("Surge model — held-out test performance")
        st.dataframe(bundle["clf_metrics"].set_index("Model").round(3), use_container_width=True)
        st.caption("Random Forest Classifier is used for live predictions.")

    st.info(
        "⚠️ **Heads-up:** surge classification scores are very high on this dataset because "
        "`Surge_Multiplier` follows a near-fixed rule based on `Demand_Supply_Ratio` here. "
        "On real production data, expect noisier, lower — though still useful — scores. "
        "Retrain periodically as live data comes in.",
        icon="⚠️",
    )

    st.markdown("Use the sidebar to get a **fare estimate**, check **surge risk**, or explore the data.")

# ---------------------------------------------------------------------------
# PAGE: Fare Predictor
# ---------------------------------------------------------------------------
elif page == "💰 Fare Predictor":
    st.title("💰 Fare Predictor")
    st.caption("Estimate the final fare for a ride before it's requested.")

    with st.form("fare_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("City", CITIES)
            vehicle = st.selectbox("Vehicle type", VEHICLES)
            day = st.selectbox("Day of week", DAYS)
        with c2:
            hour = st.slider("Hour of day", 0, 23, 9)
            distance = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
            duration = st.number_input("Trip duration (min)", min_value=1, max_value=240, value=25)
        with c3:
            traffic = st.selectbox("Traffic level", TRAFFIC_LEVELS)
            active_drivers = st.slider("Active drivers nearby", 1, 150, 50)
            ride_requests = st.slider("Ride requests nearby", 1, 300, 100)

        dsr = st.slider(
            "Demand / Supply ratio", 0.1, 40.0, round(ride_requests / max(active_drivers, 1), 2), step=0.1,
            help="Ride requests ÷ active drivers in the area right now.",
        )

        submitted = st.form_submit_button("Predict Fare", use_container_width=True, type="primary")

    if submitted:
        row = build_single_row(city, hour, day, distance, traffic, vehicle,
                                active_drivers, ride_requests, dsr, duration)
        pred_rf = bundle["fare_model"].predict(row)[0]
        pred_lr = bundle["fare_model_linear"].predict(row)[0]

        st.markdown("### Estimated Fare")
        c1, c2 = st.columns(2)
        c1.metric("Random Forest estimate (recommended)", f"₹{pred_rf:,.0f}")
        c2.metric("Linear Regression estimate", f"₹{pred_lr:,.0f}")

        fare_per_km = pred_rf / distance if distance else 0
        st.caption(f"≈ ₹{fare_per_km:.1f} per km at this distance and demand level.")

# ---------------------------------------------------------------------------
# PAGE: Surge Predictor
# ---------------------------------------------------------------------------
elif page == "⚡ Surge Predictor":
    st.title("⚡ Surge Risk Predictor")
    st.caption("Predict whether a ride is likely to hit high surge pricing (≥ 1.5×) right now.")

    with st.form("surge_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("City", CITIES, key="s_city")
            vehicle = st.selectbox("Vehicle type", VEHICLES, key="s_vehicle")
            day = st.selectbox("Day of week", DAYS, key="s_day")
        with c2:
            hour = st.slider("Hour of day", 0, 23, 18, key="s_hour")
            distance = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=10.0, key="s_dist")
            duration = st.number_input("Trip duration (min)", min_value=1, max_value=240, value=25, key="s_dur")
        with c3:
            traffic = st.selectbox("Traffic level", TRAFFIC_LEVELS, key="s_traffic")
            active_drivers = st.slider("Active drivers nearby", 1, 150, 30, key="s_drivers")
            ride_requests = st.slider("Ride requests nearby", 1, 300, 150, key="s_requests")

        dsr = st.slider(
            "Demand / Supply ratio", 0.1, 40.0, round(ride_requests / max(active_drivers, 1), 2), step=0.1,
            key="s_dsr", help="Ride requests ÷ active drivers in the area right now.",
        )

        submitted = st.form_submit_button("Predict Surge Risk", use_container_width=True, type="primary")

    if submitted:
        row = build_single_row(city, hour, day, distance, traffic, vehicle,
                                active_drivers, ride_requests, dsr, duration)
        proba = bundle["surge_model"].predict_proba(row)[0, 1]
        pred = bundle["surge_model"].predict(row)[0]

        st.markdown("### Surge Risk")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("High-surge probability", f"{proba*100:.1f}%")
            st.metric("Prediction", "🔴 High Surge" if pred == 1 else "🟢 Normal Pricing")
        with c2:
            st.progress(min(max(proba, 0.0), 1.0))
            if pred == 1:
                st.warning("Demand is outpacing supply — expect surge pricing ≥ 1.5×. Good time for drivers to reposition here.")
            else:
                st.success("Demand and supply look balanced — normal pricing expected.")

# ---------------------------------------------------------------------------
# PAGE: Data Insights
# ---------------------------------------------------------------------------
elif page == "📊 Data Insights":
    st.title("📊 Data Insights")
    st.caption("Key patterns from the historical ride data used to train these models.")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Final_Fare"], kde=True, bins=30, color="teal", ax=ax)
        ax.set_title("Final Fare Distribution")
        st.pyplot(fig, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="City", y="Final_Fare", ax=ax2, palette="Set2")
        ax2.set_title("Final Fare by City")
        ax2.tick_params(axis="x", rotation=30)
        st.pyplot(fig2, use_container_width=True)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        hourly = df.groupby("Hour_of_Day")["Final_Fare"].mean()
        sns.lineplot(x=hourly.index, y=hourly.values, marker="o", ax=ax3, color="darkgreen")
        ax3.set_title("Average Fare by Hour of Day")
        st.pyplot(fig3, use_container_width=True)

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Traffic_Level", y="Surge_Multiplier",
                    order=TRAFFIC_LEVELS, ax=ax4, palette="rocket")
        ax4.set_title("Surge Multiplier by Traffic Level")
        st.pyplot(fig4, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr_cols = ["Hour_of_Day", "Distance_km", "No_of_active_drivers", "Ride_Requests",
                 "Demand_Supply_Ratio", "Trip_Duration", "Base_Fare", "Surge_Multiplier", "Final_Fare"]
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax5)
    st.pyplot(fig5, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Model Performance
# ---------------------------------------------------------------------------
elif page == "🧪 Model Performance":
    st.title("🧪 Model Performance")

    st.subheader("Regression — Predicting Final_Fare")
    st.dataframe(bundle["reg_metrics"].set_index("Model").round(3), use_container_width=True)

    st.subheader("Classification — Predicting High_Surge")
    st.dataframe(bundle["clf_metrics"].set_index("Model").round(3), use_container_width=True)

    st.subheader("Feature Importance — Fare Model (Random Forest)")
    prep = bundle["fare_model"].named_steps["prep"]
    feat_names = NUMERIC_COLS + list(prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS))
    importances = bundle["fare_model"].named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=fi, x="importance", y="feature", palette="mako", ax=ax)
    ax.set_title("Top Features — Fare Prediction")
    st.pyplot(fig, use_container_width=True)

    st.caption(
        "Models retrain automatically whenever a new CSV is uploaded, using an 80/20 "
        "train-test split with a fixed random seed for reproducibility."
    )
