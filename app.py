# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta_classic as ta
import joblib
from datetime import datetime

st.set_page_config(page_title="AI Live Forex Engine", page_icon="⚡", layout="centered")

st.title("⚡ AI Live Forex Signal & Risk Engine")
st.write("Connected Real-Time Deployment with Dynamic Risk Management (ATR)")

try:
    model = joblib.load('final_saved_model.joblib')
    st.success(" Multi-Asset Serialized Checkpoint State Active!")
except Exception as e:
    st.error(" Pre-trained model file not found. Run 'main.py' first.")

st.subheader("Select Live Forex Pair Directory Feed:")
asset_choice = st.selectbox(
    "Target Currency Pair", 
    ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
)

ticker_mapping = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X"
}
symbol = ticker_mapping[asset_choice]

if st.button("Query Stream Pipeline & Classify Market Signal", use_container_width=True):
    with st.spinner(f"Ingesting real-time tracking metrics for {asset_choice}..."):
        try:
            ticker_data = yf.Ticker(symbol)
            live_df = ticker_data.history(period="5d", interval="1h")
            
            if live_df.empty:
                st.error("Data stream empty. Market may be closed or feed delayed.")
            else:
                # Calculate indicators
                live_df['RSI'] = ta.rsi(live_df['Close'], length=14)
                
                macd_frame = ta.macd(live_df['Close'], fast=12, slow=26, signal=9)
                live_df['MACD'] = macd_frame['MACD_12_26_9'] if macd_frame is not None and not macd_frame.empty else 0
                    
                live_df['EMA_20'] = ta.ema(live_df['Close'], length=20)
                live_df['EMA_Distance'] = live_df['Close'] - live_df['EMA_20']
                
                # Dynamic Volatility calculation using Average True Range (ATR)
                live_df['ATR'] = ta.atr(live_df['High'], live_df['Low'], live_df['Close'], length=14)
                
                latest_row = live_df.dropna().iloc[-1]
                
                current_price = latest_row['Close']
                current_rsi = latest_row['RSI']
                current_macd = latest_row['MACD']
                current_ema_dist = latest_row['EMA_Distance']
                current_atr = latest_row['ATR']
                
                # Display core parameters
                st.markdown("### Real-Time Metric Matrix Extracted:")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                metrics_col1.metric("Live Spot Price", f"{current_price:.4f}")
                metrics_col2.metric("Calculated RSI", f"{current_rsi:.2f}")
                metrics_col3.metric("Market Volatility (ATR)", f"{current_atr:.5f}")
                
                # Inference
                live_input = pd.DataFrame([{
                    'RSI': current_rsi,
                    'MACD': current_macd,
                    'EMA_Distance': current_ema_dist
                }])
                
                prediction = model.predict(live_input)[0]
                probabilities = model.predict_proba(live_input)[0]
                confidence = probabilities[prediction] * 100
                
                st.markdown("---")
                
                # Dynamic Volatility Exit Calculation Parameters (Risk Multipliers)
                risk_buffer = current_atr * 1.5
                reward_target = risk_buffer * 2.0  # 1:2 strict mathematical risk-reward ratio
                
                if prediction == 1:
                    st.markdown(f"### 🏹 AI Live Predicted Direction: <span style='color:#00cc66'>**BULLISH (BUY)**</span>", unsafe_allow_html=True)
                    stop_loss = current_price - risk_buffer
                    take_profit = current_price + reward_target
                else:
                    st.markdown(f"###  AI Live Predicted Direction: <span style='color:#ff3333'>**BEARISH (SELL)**</span>", unsafe_allow_html=True)
                    stop_loss = current_price + risk_buffer
                    take_profit = current_price - reward_target
                    
                st.metric(label="Classification Confidence Level", value=f"{confidence:.1f}%")
                
                # Risk Architecture Matrix Interface Output
                st.markdown("###  Automated Risk Management Architecture")
                st.info("Exit targets are calculated dynamically based on current 14-period Average True Range (ATR) market volatility.")
                
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                risk_col1.metric("Execution Entry Price", f"{current_price:.4f}")
                risk_col2.metric("Suggested STOP LOSS (SL)", f"{stop_loss:.4f}", delta="-Risk Target" if prediction == 1 else "+Risk Target", delta_color="inverse")
                risk_col3.metric("Suggested TAKE PROFIT (TP)", f"{take_profit:.4f}", delta="+Reward Target" if prediction == 1 else "-Reward Target")
                
                st.caption(f"Pipeline executed live at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Local Time.")
                
        except Exception as err:
            st.error(f"Execution Error during automated pipeline ingestion: {err}")