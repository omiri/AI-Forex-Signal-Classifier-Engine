# main.py
import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
from sklearn.ensemble import RandomForestClassifier
import joblib

print(" Step 1: Streaming multi-asset historical market data from Live Feeds...")

assets = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X"
}

combined_X = pd.DataFrame()
combined_y = pd.Series(dtype=int)

for name, ticker_symbol in assets.items():
    print(f"   -> Processing historical layers for {name}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="60d", interval="1h")
        
        if df.empty:
            continue
            
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd_df['MACD_12_26_9'] if macd_df is not None and not macd_df.empty else 0
            
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['EMA_Distance'] = df['Close'] - df['EMA_20']
        
        # Pull ATR tracking column for alignment structure
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        df = df.dropna()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        X_asset = df[['RSI', 'MACD', 'EMA_Distance']].iloc[:-1]
        y_asset = df['Target'].iloc[:-1]
        
        combined_X = pd.concat([combined_X, X_asset], ignore_index=True)
        combined_y = pd.concat([combined_y, y_asset], ignore_index=True)
    except Exception as e:
        print(f" Error processing {name}: {e}")

print("\n Step 2: Training unified Random Forest Architecture...")
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(combined_X, combined_y)

joblib.dump(model, 'final_saved_model.joblib')
print("\n Step 3: Multi-Asset Production model successfully exported!")