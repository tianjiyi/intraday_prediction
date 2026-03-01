# Project Overview: Kronos QQQ Intraday Prediction

## 🎯 Goal
Predict **30-minute future paths** for **QQQ** (Nasdaq-100 ETF) using the **Kronos** time-series foundation model. The system generates probabilistic forecasts to aid in intraday trading decisions.

## 🔑 Key Features
- **Probabilistic Forecasting**: Generates `N` (default 100) Monte Carlo sampled paths for the next 30 minutes.
- **Metrics**: Calculates:
  - **P(Up)**: Probability that price will be higher in 30 mins.
  - **Expected Return**: Mean return across all sampled paths.
  - **Confidence Bands**: 10th, 50th, and 90th percentile paths.
  - **Mean Reversion**: Probability of touching VWAP or Bollinger Band middle.
- **Data Source**: **Alpaca API** for real-time 1-minute OHLCV data.
- **Model**: **Kronos** (Transformer-based foundation model for time series).

## 📂 Project Structure
- **`cli_kronos_prob_qqq.py`**: Main CLI script. Fetches data, runs Kronos inference, and saves results.
- **`visualize_predictions.py`**: Tool to visualize the generated paths and confidence bands.
- **`config.yaml`**: Configuration for model (size, context), data (lookback), and sampling (temperature, top_p).
- **`README.md`**: Detailed technical documentation, setup guide, and troubleshooting.
- **`Kronos/`**: Directory containing the Kronos model source code.

## 🚀 Quick Start
1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   pip install alpaca-py plotly kaleido pyyaml
   ```
2. **Configure Credentials**:
   Set `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY` environment variables.
3. **Run Prediction**:
   ```bash
   python cli_kronos_prob_qqq.py
   ```
4. **Visualize**:
   ```bash
   python visualize_predictions.py
   ```

## 📊 Current Status
- **Codebase**: MVP implementation is complete (`cli_kronos_prob_qqq.py`).
- **Documentation**: Comprehensive `README.md` is available.
- **Next Steps**: Verify environment setup and run the first prediction test.
