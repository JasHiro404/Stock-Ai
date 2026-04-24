from flask import Blueprint, jsonify, request, send_file
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from services.predictor import train_and_predict, generate_accuracy_graph, realtime_predict

api = Blueprint('api', __name__)


# ==============================
# 🔹 HEALTH CHECK
# ==============================
@api.route('/health')
def health():
    return jsonify({"status": "ok", "message": "Stock AI API running 🚀"})


# ==============================
# 🔹 HISTORICAL STOCK DATA
# ==============================
@api.route('/history/<symbol>')
def get_history(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="30d")

        data = hist.reset_index()
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data['Date'] = data['Date'].astype(str)

        return jsonify(data.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# 🔹 STOCK PRICE CHART
# ==============================
@api.route('/plot/<symbol>')
def plot_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="30d")

        data = hist.reset_index()
        data = data[['Date', 'Close']]
        data['Date'] = pd.to_datetime(data['Date'])

        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], color='#00c853')
        plt.title(f"{symbol.upper()} Stock Price (Last 30 Days)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.xticks(rotation=45)
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# 🔮 STANDARD PREDICTION (from CSV/yfinance)
# ==============================
@api.route('/predict/<symbol>')
def predict(symbol):
    try:
        result = train_and_predict(symbol.upper())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# 🔮 REALTIME PREDICTION (manual input)
# POST /api/predict/realtime
# Body: {
#   "symbol": "RELIANCE.NS",
#   "close": 2950.50,
#   "volume": 5000000
# }
# ==============================
@api.route('/predict/realtime', methods=['POST'])
def predict_realtime():
    try:
        body = request.get_json()

        if not body:
            return jsonify({"error": "Request body is empty. Send JSON with symbol and close price."})

        symbol = body.get('symbol', '').upper()
        if not symbol:
            return jsonify({"error": "Missing 'symbol' in request body"})

        close = body.get('close')
        if close is None:
            return jsonify({"error": "Missing 'close' (today's closing price) in request body"})

        user_input = {
            "close": float(close),
            "volume": float(body.get('volume', 0)),
            "open": float(body.get('open', close)),
            "high": float(body.get('high', close)),
            "low": float(body.get('low', close)),
        }

        result = realtime_predict(symbol, user_input)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# 📊 ACCURACY GRAPH
# ==============================
@api.route('/accuracy/<symbol>')
def accuracy_graph(symbol):
    try:
        return generate_accuracy_graph(symbol.upper())
    except Exception as e:
        return jsonify({"error": str(e)})
    
@api.route("/predict/realtime/<symbol>", methods=["POST"])
def realtime(symbol):
    data = request.json
    return realtime_predict(symbol, data)
@api.route('/backtest/<symbol>')
def backtest(symbol):
    from services.predictor import backtest_model
    return backtest_model(symbol)