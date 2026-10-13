import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.layers import Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "dataset/train.csv"
N_ROWS = 50000
SEQ_LEN = 10


# ============================================================
# LOAD DATA
# ============================================================

print("==============================================")
print("TRAVEL TIME PREDICTION")
print("LSTM + ATTENTION")
print("==============================================")
print()

print("Loading dataset...")

usecols = [
    "vendor_id",
    "pickup_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "trip_duration"
]

df = pd.read_csv(
    DATA_PATH,
    usecols=usecols,
    nrows=N_ROWS
)

print("Rows loaded:", len(df))


# ============================================================
# PREPROCESSING
# ============================================================

print("Preprocessing data...")

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"],
    errors="coerce"
)

df["store_and_fwd_flag"] = df[
    "store_and_fwd_flag"
].map({
    "Y": 1,
    "N": 0
})

df = df.dropna()

df = df[
    (df["trip_duration"] > 0) &
    (df["trip_duration"] < 7200)
]

df = df.sort_values(
    "pickup_datetime"
).reset_index(drop=True)

print("Rows after preprocessing:", len(df))


# ============================================================
# TIME FEATURES
# ============================================================

print("Creating time features...")

df["hour"] = df[
    "pickup_datetime"
].dt.hour

df["day_of_week"] = df[
    "pickup_datetime"
].dt.dayofweek

df["month"] = df[
    "pickup_datetime"
].dt.month


# ============================================================
# DISTANCE FEATURE
# ============================================================

print("Creating distance feature...")

df["distance_km"] = np.sqrt(
    (
        df["pickup_longitude"]
        -
        df["dropoff_longitude"]
    ) ** 2
    +
    (
        df["pickup_latitude"]
        -
        df["dropoff_latitude"]
    ) ** 2
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

features = [
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

target = "trip_duration"

X = df[features].values
y = df[[target]].values

print("Number of features:", len(features))


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("Splitting data...")

split = int(len(df) * 0.8)

X_train_raw = X[:split]
y_train_raw = y[:split]


# ============================================================
# SCALING
# ============================================================

print("Scaling data...")

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(X_train_raw)
target_scaler.fit(y_train_raw)

X_scaled = feature_scaler.transform(X)
y_scaled = target_scaler.transform(y)


# ============================================================
# CREATE SEQUENCES
# ============================================================

print("Creating LSTM sequences...")

X_sequences = []
y_sequences = []

for i in range(SEQ_LEN, len(X_scaled)):

    X_sequences.append(
        X_scaled[i - SEQ_LEN:i]
    )

    y_sequences.append(
        y_scaled[i]
    )

X_sequences = np.array(
    X_sequences,
    dtype=np.float32
)

y_sequences = np.array(
    y_sequences,
    dtype=np.float32
)

print(
    "Sequence shape:",
    X_sequences.shape
)


# ============================================================
# TRAIN TEST SEQUENCES
# ============================================================

train_end = split - SEQ_LEN

X_train = X_sequences[:train_end]
y_train = y_sequences[:train_end]

X_test = X_sequences[train_end:]
y_test = y_sequences[train_end:]

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print()
print("Building LSTM + Attention model...")

inputs = Input(
    shape=(
        SEQ_LEN,
        len(features)
    )
)

x = LSTM(
    32,
    return_sequences=True
)(inputs)

x = Dropout(
    0.2
)(x)

attention = Attention()([
    x,
    x
])

x = GlobalAveragePooling1D()(
    attention
)

x = Dense(
    16,
    activation="relu"
)(x)

x = Dropout(
    0.2
)(x)

outputs = Dense(
    1
)(x)

model = Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print()
print("Model created successfully.")

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("==============================================")
print("STARTING TRAINING")
print("==============================================")
print()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("Making predictions...")

predicted_scaled = model.predict(
    X_test,
    verbose=1
)

predicted = target_scaler.inverse_transform(
    predicted_scaled
).flatten()

actual = target_scaler.inverse_transform(
    y_test
).flatten()


# ============================================================
# PERFORMANCE METRICS
# ============================================================

mae = mean_absolute_error(
    actual,
    predicted
)

mse = mean_squared_error(
    actual,
    predicted
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual,
    predicted
)

print()
print("==============================================")
print("MODEL PERFORMANCE")
print("==============================================")
print(
    "MAE  :",
    round(mae, 2),
    "seconds"
)

print(
    "MSE  :",
    round(mse, 2)
)

print(
    "RMSE :",
    round(rmse, 2),
    "seconds"
)

print(
    "R2   :",
    round(r2, 4)
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("Saving model...")

model.save(
    "travel_time_lstm_attention.keras"
)

print(
    "Saved: travel_time_lstm_attention.keras"
)


# ============================================================
# SAVE SCALERS
# ============================================================

with open(
    "feature_scaler.pkl",
    "wb"
) as f:
    pickle.dump(
        feature_scaler,
        f
    )

with open(
    "target_scaler.pkl",
    "wb"
) as f:
    pickle.dump(
        target_scaler,
        f
    )

print("Scalers saved.")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual_Travel_Time": actual,
    "Predicted_Travel_Time": predicted
})

results.to_csv(
    "prediction_results.csv",
    index=False
)

print(
    "Saved: prediction_results.csv"
)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    actual[:200],
    label="Actual"
)

plt.plot(
    predicted[:200],
    label="Predicted"
)

plt.xlabel(
    "Trip Number"
)

plt.ylabel(
    "Travel Time (seconds)"
)

plt.title(
    "Actual vs Predicted Travel Time"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png"
)

plt.close()

print(
    "Saved: actual_vs_predicted.png"
)


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_loss.png"
)

plt.close()

print(
    "Saved: training_loss.png"
)


# ============================================================
# FINISHED
# ============================================================

print()
print("==============================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==============================================")
print()
print("Files created:")
print("1. travel_time_lstm_attention.keras")
print("2. feature_scaler.pkl")
print("3. target_scaler.pkl")
print("4. prediction_results.csv")
print("5. actual_vs_predicted.png")
print("6. training_loss.png")
print()
print("DONE!")