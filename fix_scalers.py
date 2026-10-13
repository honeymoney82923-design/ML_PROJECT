import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

print("Loading dataset...")

df = pd.read_csv("dataset/train.csv")

# Prepare date and time features
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"]
)

df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

# Prepare location difference features
df["longitude_diff"] = (
    df["dropoff_longitude"] - df["pickup_longitude"]
)

df["latitude_diff"] = (
    df["dropoff_latitude"] - df["pickup_latitude"]
)

# Use the same 12 features as the training code
features = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "hour",
    "day",
    "month",
    "day_of_week",
    "longitude_diff",
    "latitude_diff"
]

df = df[features + ["trip_duration"]].dropna()

df = df[
    (df["trip_duration"] > 0)
    & (df["trip_duration"] < 7200)
]

# Fit new scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(df[features].values)
target_scaler.fit(df[["trip_duration"]].values)

# Save the corrected scalers
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

print("\nScaler files updated!")
print("Feature count:", feature_scaler.n_features_in_)
print("Target count:", target_scaler.n_features_in_)
print("\nCompleted.")