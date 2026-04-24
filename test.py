import yfinance as yf
import pandas as pd# --- Step 1: Fetch last 120 days from yfinance ---

try:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="120d")

    if hist is None or hist.empty:
        return {"error": f"No data returned from yfinance for {symbol}"}

    if len(hist) < SEQUENCE_LEN:
        return {"error": f"Not enough live data for {symbol}"}

    hist = hist.reset_index()

    # Normalize column names
    hist.columns = [str(col).strip() for col in hist.columns]

    # Rename Date/Datetime column
    if 'Datetime' in hist.columns:
        hist = hist.rename(columns={'Datetime': 'Date'})

    if 'Date' not in hist.columns or 'Close' not in hist.columns:
        return {"error": f"Unexpected columns: {list(hist.columns)}"}

    hist = hist[['Date', 'Close', 'Volume']].copy()

    # ✅ KEY FIX: strip timezone so concat works
    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)

except Exception as e:
    return {"error": f"yfinance fetch failed: {str(e)}"}

# --- Step 2: Append user's today row ---
today_row = pd.DataFrame([{
    'Date': pd.Timestamp.now(),   # now both are tz-naive ✅
    'Close': float(user_input['close']),
    'Volume': float(user_input.get('volume', hist['Volume'].iloc[-1]))
}])

df = pd.concat([hist, today_row], ignore_index=True)
df = df.sort_values('Date').reset_index(drop=True)