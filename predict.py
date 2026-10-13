import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

# ============================================================
# TRAVEL TIME PREDICTION USING LSTM + ATTENTION
# ============================================================

print("\n" + "=" * 55)
print("       TRAVEL TIME PREDICTION SYSTEM")
print("              LSTM + ATTENTION")
print("=" * 55)

# ------------------------------------------------------------
# Load trained model and scalers
# ------------------------------------------------------------

print("\nLoading trained model...")

try:
    model = tf.keras.models.load_model(
        "travel_time_lstm_attention.keras"
    )

    feature_scaler = joblib.load(
        "feature_scaler.pkl"
    )

    target_scaler = joblib.load(
        "target_scaler.pkl"
    )

    print("Model loaded successfully!")
    print("Scalers loaded successfully!")

except Exception as error:
    print("\nERROR while loading model/files:")
    print(type(error).__name__, ":", error)
    input("\nPress Enter to exit...")
    exit()


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------

def predict_travel_time():

    print("\n" + "-" * 55)
    print("              ENTER TRIP DETAILS")
    print("-" * 55)

    try:

        # User inputs
        vendor_id = int(
            input("Vendor ID (1 or 2): ")
        )

        passenger_count = int(
            input("Number of passengers: ")
        )

        pickup_longitude = float(
            input("Pickup longitude: ")
        )

        pickup_latitude = float(
            input("Pickup latitude: ")
        )

        dropoff_longitude = float(
            input("Drop-off longitude: ")
        )

        dropoff_latitude = float(
            input("Drop-off latitude: ")
        )

        pickup_datetime = input(
            "Pickup date and time (YYYY-MM-DD HH:MM): "
        ).strip()

        # ----------------------------------------------------
        # Convert date/time
        # ----------------------------------------------------

        dt = datetime.strptime(
            pickup_datetime,
            "%Y-%m-%d %H:%M"
        )

        # ----------------------------------------------------
        # Create additional features
        # Same features used during model training
        # ----------------------------------------------------

        hour = dt.hour
        day = dt.day
        month = dt.month
        day_of_week = dt.weekday()

        longitude_diff = (
            dropoff_longitude - pickup_longitude
        )

        latitude_diff = (
            dropoff_latitude - pickup_latitude
        )

        # ----------------------------------------------------
        # Create feature array
        # ----------------------------------------------------

        features = np.array([[
            vendor_id,
            passenger_count,
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            hour,
            day,
            month,
            day_of_week,
            longitude_diff,
            latitude_diff
        ]], dtype=float)

        # ----------------------------------------------------
        # Scale features
        # ----------------------------------------------------

        scaled_features = feature_scaler.transform(
            features
        )

        # ----------------------------------------------------
        # LSTM requires 10 time steps
        # Repeat the entered trip information 10 times
        # ----------------------------------------------------

        sequence = np.repeat(
            scaled_features[:, np.newaxis, :],
            10,
            axis=1
        )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction_scaled = model.predict(
            sequence,
            verbose=0
        )

        # Convert prediction back to seconds
        prediction_seconds = target_scaler.inverse_transform(
            prediction_scaled
        )[0][0]

        prediction_seconds = max(
            0,
            float(prediction_seconds)
        )

        # ----------------------------------------------------
        # Convert seconds to minutes and seconds
        # ----------------------------------------------------

        minutes = int(
            prediction_seconds // 60
        )

        seconds = int(
            prediction_seconds % 60
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print("\n")
        print("=" * 55)
        print("             PREDICTION RESULT")
        print("=" * 55)

        print(
            f"Pickup Time       : {dt.strftime('%Y-%m-%d %H:%M')}"
        )

        print(
            f"Passengers        : {passenger_count}"
        )

        print(
            f"Predicted Time    : {minutes} minutes "
            f"{seconds} seconds"
        )

        print(
            f"Predicted Seconds : {prediction_seconds:.2f}"
        )

        print("=" * 55)

        print(
            "\nTravel time prediction completed successfully!"
        )

    except ValueError as error:

        print("\n" + "-" * 55)
        print("INVALID INPUT")
        print("-" * 55)
        print("Please enter the values in the correct format.")
        print("Date format must be:")
        print("YYYY-MM-DD HH:MM")
        print("\nError:", error)

    except Exception as error:

        print("\n" + "-" * 55)
        print("PREDICTION ERROR")
        print("-" * 55)
        print(
            type(error).__name__,
            ":",
            error
        )


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

while True:

    predict_travel_time()

    print("\n")

    again = input(
        "Do you want to predict another trip? (yes/no): "
    ).strip().lower()

    if again not in ["yes", "y"]:

        print("\n" + "=" * 55)
        print(" Thank you for using Travel Time Prediction System!")
        print("=" * 55)

        break