import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import math
import os

from datetime import date, time, datetime
from sklearn.base import clone

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Travel Time Prediction",
    page_icon="🚕",
    layout="wide"
)

# =====================================================
# 2. DASHBOARD DESIGN
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #172b46);
}
.block-container {
    padding-top: 2rem;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #087ea4, #3154a5);
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
}
.hero p {
    color: #eaf6ff;
    font-size: 17px;
}
.result-box {
    background: #183b4e;
    border: 1px solid #38bdf8;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
}
.result-box h2, .result-box h1, .result-box p {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. FILE PATHS
# =====================================================
MODEL_PATH = "travel_time_lstm_attention.keras"
FEATURE_SCALER_PATH = "feature_scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"
DATASET_PATH = os.path.join("dataset", "train.csv")

FEATURES = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "hour",
    "day_of_week",
    "month",
    "store_and_fwd_flag",
    "distance_km"
]

TARGET = "trip_duration"

# =====================================================
# 4. NYC DEMO LOCATIONS
# Coordinates are (longitude, latitude)
# =====================================================
LOCATIONS = {
    "Columbia University (College)": (-73.9626, 40.8075),
    "Grand Central Terminal (Railway Station)": (-73.9772, 40.7527),
    "Times Square": (-73.9855, 40.7580),
    "Penn Station": (-73.9936, 40.7506),
    "Empire State Building": (-73.9857, 40.7484),
    "Central Park": (-73.9654, 40.7829),
    "New York University": (-73.9965, 40.7295),
    "Brooklyn Bridge": (-73.9969, 40.7061),
    "Wall Street": (-74.0090, 40.7075),
    "JFK Airport": (-73.7781, 40.6413)
}

# =====================================================
# 5. DISTANCE CALCULATION
# =====================================================
def calculate_distance_km(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    return earth_radius * 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

# =====================================================
# 6. PREPARE DATASET FOR SCALER REPAIR
# =====================================================
@st.cache_data
def prepare_training_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "vendor_id",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
        "store_and_fwd_flag",
        "trip_duration"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="coerce"
    )

    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month

    df["store_and_fwd_flag"] = (
        df["store_and_fwd_flag"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"N": 0, "Y": 1})
    )

    df["distance_km"] = df.apply(
        lambda row: calculate_distance_km(
            row["pickup_latitude"],
            row["pickup_longitude"],
            row["dropoff_latitude"],
            row["dropoff_longitude"]
        ),
        axis=1
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=FEATURES + [TARGET]
    )

    df = df[
        (df["trip_duration"] > 0)
        & (df["passenger_count"] > 0)
    ]

    return df

# =====================================================
# 7. LOAD MODEL AND SCALERS
# =====================================================
@st.cache_resource
def load_model_and_scalers():
    for path in [
        MODEL_PATH,
        FEATURE_SCALER_PATH,
        TARGET_SCALER_PATH
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    feature_scaler = joblib.load(
        FEATURE_SCALER_PATH
    )

    target_scaler = joblib.load(
        TARGET_SCALER_PATH
    )

    model_shape = model.input_shape

    if isinstance(model_shape, list):
        if len(model_shape) != 1:
            raise ValueError(
                "This dashboard expects a model with one input."
            )
        model_shape = model_shape[0]

    if len(model_shape) != 3:
        raise ValueError(
            f"Expected a sequence model with 3 input dimensions. "
            f"Actual input shape: {model_shape}"
        )

    sequence_length = model_shape[1]
    expected_features = model_shape[2]

    if sequence_length is None or expected_features is None:
        raise ValueError(
            "The model input dimensions are not fully defined."
        )

    if expected_features != len(FEATURES):
        raise ValueError(
            f"The saved model expects {expected_features} features, "
            f"but this dashboard prepares {len(FEATURES)}. "
            "The model must be retrained using the intended "
            "11-feature training code."
        )

    # Repair the feature scaler if its feature count/order differs.
    scaler_names = getattr(
        feature_scaler,
        "feature_names_in_",
        None
    )

    feature_scaler_matches = (
        getattr(feature_scaler, "n_features_in_", None)
        == len(FEATURES)
    )

    if scaler_names is not None:
        feature_scaler_matches = (
            feature_scaler_matches
            and list(scaler_names) == FEATURES
        )

    if not feature_scaler_matches:
        training_df = prepare_training_data(
            DATASET_PATH
        )

        replacement_scaler = clone(feature_scaler)

        replacement_scaler.fit(
            training_df[FEATURES]
        )

        feature_scaler = replacement_scaler

        joblib.dump(
            feature_scaler,
            FEATURE_SCALER_PATH
        )

    # The target scaler must accept one target column.
    if getattr(target_scaler, "n_features_in_", None) != 1:
        training_df = prepare_training_data(
            DATASET_PATH
        )

        replacement_target_scaler = clone(
            target_scaler
        )

        replacement_target_scaler.fit(
            training_df[[TARGET]]
        )

        target_scaler = replacement_target_scaler

        joblib.dump(
            target_scaler,
            TARGET_SCALER_PATH
        )

    return (
        model,
        feature_scaler,
        target_scaler,
        int(sequence_length)
    )

# =====================================================
# 8. DASHBOARD HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🚕 Travel Time Prediction System</h1>
    <p>
        An academic travel-duration estimation dashboard
        using LSTM and Attention.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Dataset: NYC taxi trips. Supported locations are a "
    "predefined New York City demo list, not live location search."
)

# =====================================================
# 9. USER INPUTS
# =====================================================
st.subheader("📍 Plan Your Journey")

col1, col2 = st.columns(2)

with col1:
    pickup = st.selectbox(
        "Pickup Location",
        list(LOCATIONS.keys()),
        index=0
    )

    passengers = st.selectbox(
        "Number of Passengers",
        [1, 2, 3, 4, 5, 6],
        index=1
    )

    travel_date = st.date_input(
        "Travel Date",
        value=date(2016, 3, 15)
    )

with col2:
    dropoff = st.selectbox(
        "Drop-off Location",
        list(LOCATIONS.keys()),
        index=1
    )

    vendor_id = st.selectbox(
        "Taxi Provider",
        [1, 2],
        index=0,
        help="The dataset uses numeric vendor IDs."
    )

    travel_time = st.time_input(
        "Pickup Time",
        value=time(8, 30)
    )

st.info(
    "The app calculates straight-line distance between "
    "the selected locations. This is not road distance."
)

# =====================================================
# 10. PREDICT TRAVEL TIME
# =====================================================
if st.button(
    "🚀 Predict Travel Time",
    type="primary",
    use_container_width=True
):
    try:
        with st.spinner(
            "Loading model and calculating estimate..."
        ):
            (
                model,
                feature_scaler,
                target_scaler,
                sequence_length
            ) = load_model_and_scalers()

            pickup_lon, pickup_lat = LOCATIONS[pickup]
            dropoff_lon, dropoff_lat = LOCATIONS[dropoff]

            selected_datetime = datetime.combine(
                travel_date,
                travel_time
            )

            hour = selected_datetime.hour
            day_of_week = selected_datetime.weekday()
            month = selected_datetime.month

            # NYC taxi dataset uses Y/N.
            # This demo sets the flag to N (0).
            store_and_fwd_flag = 0

            distance_km = calculate_distance_km(
                pickup_lat,
                pickup_lon,
                dropoff_lat,
                dropoff_lon
            )

            input_df = pd.DataFrame(
                [[
                    vendor_id,
                    passengers,
                    pickup_lon,
                    pickup_lat,
                    dropoff_lon,
                    dropoff_lat,
                    hour,
                    day_of_week,
                    month,
                    store_and_fwd_flag,
                    distance_km
                ]],
                columns=FEATURES
            )

            if feature_scaler.n_features_in_ != len(FEATURES):
                raise ValueError(
                    "The feature scaler still does not match "
                    "the 11 dashboard features."
                )

            scaled_input = feature_scaler.transform(
                input_df
            )

            # Prototype only: repeat this one trip to satisfy
            # the model's required sequence length.
            sequence = np.repeat(
                scaled_input.reshape(
                    1, 1, len(FEATURES)
                ),
                repeats=sequence_length,
                axis=1
            )

            scaled_prediction = model.predict(
                sequence,
                verbose=0
            )

            scaled_prediction = np.asarray(
                scaled_prediction
            ).reshape(-1, 1)

            if scaled_prediction.shape[0] != 1:
                raise ValueError(
                    "The model returned an unexpected number "
                    "of predictions."
                )

            predicted_seconds = float(
                target_scaler.inverse_transform(
                    scaled_prediction
                )[0, 0]
            )

            if not np.isfinite(predicted_seconds):
                raise ValueError(
                    "The model returned an invalid numeric result."
                )

            predicted_seconds = max(
                0.0,
                predicted_seconds
            )

            predicted_minutes = predicted_seconds / 60

        # =================================================
        # 11. DISPLAY RESULTS
        # =================================================
        st.divider()
        st.subheader("📊 Your Trip Estimate")

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "📏 Straight-line Distance",
            f"{distance_km:.2f} km"
        )

        r2.metric(
            "⏱️ Predicted Duration",
            f"{predicted_minutes:.1f} minutes"
        )

        r3.metric(
            "👥 Passengers",
            str(passengers)
        )

        st.markdown(
            f"""
            <div class="result-box">
                <h2>Estimated Model Output</h2>
                <h1>{predicted_minutes:.1f} minutes</h1>
                <p>{pickup} → {dropoff}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"Model output: approximately "
            f"{predicted_seconds:.0f} seconds."
        )

        st.warning(
            "Research prototype: the current app repeats the "
            "same trip features to create the LSTM sequence. "
            "It does not use 10 actual historical trips. "
            "The result is experimental and has not been "
            "validated for real-world travel. Rebuilding a "
            "scaler cannot repair a model trained with a "
            "different feature definition."
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {type(error).__name__}: {error}"
        )

        st.info(
            "Check that app.py, the trained model, both scaler "
            "files, and dataset/train.csv are in the expected "
            "project locations."
        )

# =====================================================
# 12. FOOTER
# =====================================================
st.divider()

st.caption(
    "Academic project prototype | LSTM + Attention | "
    "NYC taxi trip-duration dataset"
)
