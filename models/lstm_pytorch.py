import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/saved/lstm_model.pt"   # saved weights
SCALER_PATH = "models/saved/lstm_scaler.npy" # saved scaler params
SEQ_LENGTH  = 7    # days of history used per prediction
HIDDEN_SIZE = 64   # LSTM hidden units (increased from 50)
EPOCHS      = 50   # max training epochs
LR          = 0.001
PATIENCE    = 7    # early-stopping: stop if no improvement for N epochs


# ── 1. Prepare & normalise data ───────────────────────────────────────────────
def prepare_data(df):
    """Aggregate to daily sales and normalise to [0, 1]."""
    sales  = df.groupby("Date")["Quantity_Sold"].sum().sort_index()
    values = sales.values.astype(float).reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    return scaled, scaler


# ── 2. Sliding-window sequences ───────────────────────────────────────────────
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# ── 3. Model definition ───────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── 4. Train with early stopping ─────────────────────────────────────────────
def train_model(df, epochs=EPOCHS, force_retrain=False):
    """
    Train the LSTM.  If a saved model already exists and force_retrain=False,
    the saved model is loaded instead of retraining.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # ── Load existing model if available ──────────────────────────────────────
    if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ Loading saved LSTM model (use force_retrain=True to retrain)")
        model  = LSTMModel()
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        scaler = _load_scaler()
        return model, scaler

    print("🔁 Training LSTM model...")

    scaled, scaler = prepare_data(df)

    # Need at least seq_length + 1 data points
    if len(scaled) <= SEQ_LENGTH:
        raise ValueError(
            f"Not enough data to train LSTM. "
            f"Need > {SEQ_LENGTH} days, got {len(scaled)}."
        )

    X, y = create_sequences(scaled)

    # ── Train / validation split (80 / 20) ───────────────────────────────────
    split  = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(-1)
    X_tr  = to_tensor(X_tr);  y_tr  = to_tensor(y_tr)
    X_val = to_tensor(X_val); y_val = to_tensor(y_val)

    model     = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=False
    )

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(X_tr), y_tr)
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        optimizer.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{epochs}  train={train_loss.item():.5f}  val={val_loss:.5f}")

        # ── Early stopping ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  ⏹  Early stop at epoch {epoch} (best val loss={best_val_loss:.5f})")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # ── Save model & scaler ────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    _save_scaler(scaler)
    print(f"  💾 Model saved → {MODEL_PATH}")

    model.eval()
    return model, scaler


# ── 5. Predict ────────────────────────────────────────────────────────────────
def predict_future(model, scaler, df, seq_length=SEQ_LENGTH):
    """Use the last `seq_length` days to predict the next day, then inverse-scale."""
    scaled, _ = prepare_data(df)

    if len(scaled) < seq_length:
        raise ValueError(f"Need at least {seq_length} days of data to predict.")

    last_seq = scaled[-seq_length:]
    tensor   = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(tensor).item()

    # Inverse-transform back to real units
    prediction = scaler.inverse_transform([[pred_scaled]])[0][0]
    return float(prediction)


# ── 6. Confidence range (simple) ─────────────────────────────────────────────
def prediction_with_range(model, scaler, df, seq_length=SEQ_LENGTH, margin=0.10):
    """
    Returns (prediction, lower_bound, upper_bound).
    Uses a ±10 % margin as a simple confidence range.
    """
    pred  = predict_future(model, scaler, df, seq_length)
    lower = pred * (1 - margin)
    upper = pred * (1 + margin)
    return pred, lower, upper


# ── 7. Public API ─────────────────────────────────────────────────────────────
def run_lstm(df, force_retrain=False):
    """
    Full pipeline.  Returns predicted next-day sales as a float.
    Loads saved model if available; trains only when needed.
    """
    model, scaler = train_model(df, force_retrain=force_retrain)
    prediction    = predict_future(model, scaler, df)
    return prediction


def run_lstm_with_range(df, force_retrain=False):
    """Returns (prediction, lower, upper) for dashboard confidence display."""
    model, scaler = train_model(df, force_retrain=force_retrain)
    return prediction_with_range(model, scaler, df)


# ── 8. Scaler helpers ─────────────────────────────────────────────────────────
def _save_scaler(scaler):
    np.save(SCALER_PATH, [scaler.data_min_, scaler.data_max_])

def _load_scaler():
    arr    = np.load(SCALER_PATH, allow_pickle=True)
    scaler = MinMaxScaler()
    scaler.fit(arr[0].reshape(-1, 1))          # sets internal state
    scaler.data_min_  = arr[0]
    scaler.data_max_  = arr[1]
    scaler.data_range_ = arr[1] - arr[0]
    scaler.scale_      = 1.0 / scaler.data_range_
    scaler.min_        = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = 1
    return scaler


# ── 9. Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.preprocessing import preprocess_data

    df   = preprocess_data()
    pred, lo, hi = run_lstm_with_range(df)

    print(f"\n📈 Next-Day Sales Prediction : {pred:.1f} units")
    print(f"   Confidence range          : {lo:.1f} – {hi:.1f} units")