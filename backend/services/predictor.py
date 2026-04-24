import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from io import BytesIO
from flask import send_file

# ==============================
# CONFIG
# ==============================
SEQUENCE_LEN = 60
INITIAL_CAPITAL = 100000

FEATURES = [
    'returns',        # index 0 — the column the model predicts
    'log_returns',
    'MA10',
    'MA50',
    'trend',
    'volatility_10',
    'volatility_30',
    'RSI',
    'MACD',
    'MACD_Signal',
    'volume_change'
]

# ==============================
# DATA
# ==============================
def load_data(symbol):
    df = yf.Ticker(symbol.upper()).history(period="5y").reset_index()
    return df if not df.empty else None


# ==============================
# FEATURES
# No 'target' column is created here.
# The target IS 'returns' (index 0 in FEATURES), extracted inside
# create_sequences directly from the scaled array.
# ==============================
def create_features(df):
    df = df[['Date', 'Close', 'Volume']].copy()

    df['returns']       = df['Close'].pct_change()
    df['log_returns']   = np.log(df['Close'] / df['Close'].shift(1))
    df['MA10']          = df['Close'].rolling(10).mean()
    df['MA50']          = df['Close'].rolling(50).mean()
    df['trend']         = df['MA10'] - df['MA50']
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_30'] = df['returns'].rolling(30).std()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['volume_change'] = df['Volume'].pct_change()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


# ==============================
# SEQUENCES
# Takes ONE argument: the already-scaled 2-D numpy array.
# y is always column 0 (= 'returns') of the NEXT timestep.
# ==============================
def create_sequences(data):
    X, y = [], []
    for i in range(SEQUENCE_LEN, len(data)):
        X.append(data[i - SEQUENCE_LEN:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ==============================
# MODEL
# ==============================
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ==============================
# SCALER PERSISTENCE
# Fitted ONCE on training data; reused in every downstream call.
# ==============================
def _scaler_path(symbol):
    return f"models/{symbol.upper()}_scaler.pkl"

def save_scaler(symbol, scaler):
    os.makedirs("models", exist_ok=True)
    with open(_scaler_path(symbol), 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(symbol):
    path = _scaler_path(symbol)
    return pickle.load(open(path, 'rb')) if os.path.exists(path) else None


# ==============================
# INVERSE TRANSFORM
# Model output is in scaled [0,1] space; invert to get real return.
# ==============================
def inverse_return(scaler, val):
    dummy       = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = val
    return float(scaler.inverse_transform(dummy)[0, 0])


# ==============================
# SIGNAL
# ==============================
def signal(r):
    if r > 0.002:
        return "BUY 🟩"
    elif r < -0.002:
        return "SELL 🟥"
    return "HOLD ⬜"


# ==============================
# MODEL PATH HELPER
# ==============================
def _model_path(symbol):
    return f"models/{symbol.upper()}_returns.h5"


# ==============================
# MODEL LOAD / TRAIN
# ==============================
def get_model(symbol, X_train, y_train, X_val, y_val):
    path = _model_path(symbol)
    if os.path.exists(path):
        return load_model(path, compile=False)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    os.makedirs("models", exist_ok=True)
    model.save(path)
    return model


# ==============================
# TRAIN + PREDICT  (historical mode)
# ==============================
def train_and_predict(symbol):
    df = load_data(symbol)
    if df is None:
        return {"error": "No data found"}

    df    = create_features(df)
    split = int(len(df) * 0.8)

    train_df, test_df = df[:split], df[split:]

    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURES])
    save_scaler(symbol, scaler)

    X_train, y_train = create_sequences(scaler.transform(train_df[FEATURES]))
    X_test,  y_test  = create_sequences(scaler.transform(test_df[FEATURES]))

    model = get_model(symbol, X_train, y_train, X_test, y_test)

    seq         = scaler.transform(df[FEATURES])[-SEQUENCE_LEN:]
    pred_scaled = float(model.predict(np.array([seq]), verbose=0)[0][0])
    pred_return = inverse_return(scaler, pred_scaled)
    last_price  = float(df['Close'].iloc[-1])

    return {
        "symbol":                   symbol.upper(),
        "last_price":               round(last_price, 2),
        "predicted_next_day_price": round(last_price * (1 + pred_return), 2),
        "predicted_return_pct":     round(pred_return * 100, 3),
        "signal":                   signal(pred_return),
        "mode":                     "historical"
    }


# ==============================
# REALTIME PREDICTION
# ==============================
def realtime_predict(symbol, user_input):
    df = load_data(symbol)
    if df is None:
        return {"error": "No data"}

    scaler = load_scaler(symbol)
    if scaler is None:
        return {"error": "Run train_and_predict first"}

    price   = float(user_input['close'])
    new_row = pd.DataFrame([{
        "Date":   pd.Timestamp.now(),
        "Close":  price,
        "Volume": user_input.get('volume', df['Volume'].iloc[-1])
    }])

    df          = pd.concat([df, new_row], ignore_index=True)
    df          = create_features(df)
    seq         = scaler.transform(df[FEATURES])[-SEQUENCE_LEN:]

    model_file  = _model_path(symbol)
    if not os.path.exists(model_file):
        return {"error": "Model not found. Run train_and_predict first."}

    model       = load_model(model_file, compile=False)
    pred_scaled = float(model.predict(np.array([seq]), verbose=0)[0][0])
    pred_return = inverse_return(scaler, pred_scaled)

    return {
        "symbol":                   symbol.upper(),
        "input_price":              round(price, 2),
        "predicted_next_day_price": round(price * (1 + pred_return), 2),
        "predicted_return_pct":     round(pred_return * 100, 3),
        "signal":                   signal(pred_return),
        "mode":                     "realtime"
    }


# ==============================
# BACKTEST
# ==============================
def backtest_model(symbol):
    df = load_data(symbol)
    if df is None:
        return {"error": "No data"}

    df = create_features(df).tail(1000)

    scaler = load_scaler(symbol)
    if scaler is None:
        return {"error": "Run train_and_predict first"}

    model_file = _model_path(symbol)
    if not os.path.exists(model_file):
        return {"error": "Model not found"}

    X, _ = create_sequences(scaler.transform(df[FEATURES]))

    model = load_model(model_file, compile=False)
    preds = model.predict(X, verbose=0).flatten()

    prices = df['Close'].values[SEQUENCE_LEN:]

    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    holding_days = 0

    equity_curve = []
    trades = 0
    wins = 0

    # 🔥 STRATEGY PARAMETERS
    ENTRY_THRESHOLD = 0.001      # earlier entry
    EXIT_THRESHOLD = -0.001
    STOP_LOSS = -0.02            # -2%
    TAKE_PROFIT = 0.04           # +4%
    MAX_HOLD_DAYS = 10
    COST = 0.002                 # transaction cost

    for pred_scaled, price in zip(preds, prices):
        pred_return = inverse_return(scaler, pred_scaled)

        # ======================
        # ENTRY
        # ======================
        if position == 0 and pred_return > ENTRY_THRESHOLD:
            position = 1
            entry_price = price
            holding_days = 0

        # ======================
        # POSITION MANAGEMENT
        # ======================
        elif position == 1:
            holding_days += 1
            current_return = (price - entry_price) / entry_price

            exit_flag = False

            # exit conditions
            if pred_return < EXIT_THRESHOLD:
                exit_flag = True
            elif current_return < STOP_LOSS:
                exit_flag = True
            elif current_return > TAKE_PROFIT:
                exit_flag = True
            elif holding_days >= MAX_HOLD_DAYS:
                exit_flag = True

            if exit_flag:
                trade_return = current_return - COST
                capital *= (1 + trade_return)

                trades += 1
                if trade_return > 0:
                    wins += 1

                position = 0

        # ======================
        # MARK-TO-MARKET
        # ======================
        if position == 1:
            equity_curve.append(capital * (price / entry_price))
        else:
            equity_curve.append(capital)

    eq = np.array(equity_curve)

    # ======================
    # METRICS
    # ======================
    returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0])

    sharpe = 0
    if np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

    peak = eq[0]
    max_dd = 0
    for v in eq:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    return {
        "symbol": symbol.upper(),
        "final_capital": round(float(eq[-1]), 2),
        "total_return_pct": round((eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round((wins / trades * 100) if trades else 0, 2),
        "total_trades": trades
    }


# ==============================
# ACCURACY GRAPH
# ==============================
def generate_accuracy_graph(symbol):
    df = load_data(symbol)
    if df is None:
        return {"error": "No data"}

    df = create_features(df).tail(500)

    model_file = _model_path(symbol)
    if not os.path.exists(model_file):
        return {"error": "Model not found. Run train_and_predict first."}

    scaler = load_scaler(symbol)
    if scaler is None:
        return {"error": "Run train_and_predict first"}

    scaled        = scaler.transform(df[FEATURES])
    X, y_scaled   = create_sequences(scaled)

    model         = load_model(model_file, compile=False)
    preds_scaled  = model.predict(X, verbose=0).flatten()

    y_real     = [inverse_return(scaler, v) for v in y_scaled]
    preds_real = [inverse_return(scaler, v) for v in preds_scaled]

    plt.figure(figsize=(12, 6))
    plt.plot(y_real,     label='Actual Returns',    linewidth=1)
    plt.plot(preds_real, label='Predicted Returns', linewidth=1, alpha=0.8)
    plt.title(f"{symbol.upper()} — Actual vs Predicted Returns")
    plt.xlabel("Timestep")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png', dpi=120)
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')