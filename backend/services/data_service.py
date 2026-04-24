import yfinance as yf

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d")

    if data.empty:
        return {"error": "Invalid stock symbol"}

    latest = data.iloc[-1]

    return {
        "symbol": symbol,
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "close": float(latest["Close"]),
        "volume": int(latest["Volume"])
    }