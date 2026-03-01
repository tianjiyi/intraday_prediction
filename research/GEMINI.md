# Project Gemini Context: Intraday Prediction with Kronos

## Overview
This project is a sophisticated financial forecasting system designed to predict intraday movements of the QQQ ETF. It leverages **Kronos**, a time-series foundation model, to generate probabilistic 30-minute forecasts. The system aids in trading decisions by calculating probabilities of price movements (e.g., P(Close > Current)).

## Core Architecture: Kronos
The heart of the system is the **Kronos** model (`Kronos/model/kronos.py`).
- **Type**: Transformer-based foundation model.
- **Mechanism**: Uses a neural tokenizer (`KronosTokenizer`) to convert continuous time-series data into a discrete, hierarchical representation.
- **Inference**: Performs autoregressive prediction on these discrete tokens to generate future price paths.

## Key Modules

### 1. Prediction Engine (`cli_kronos_prob_qqq.py`)
The primary interface for generating on-demand forecasts.
- **Input**: Fetches real-time or historical data (e.g., from Alpaca).
- **Process**: Runs the Kronos model to generate Monte Carlo samples (simulated future paths).
- **Output**: Calculates probabilities and saves results (CSV/JSON).
- **Note**: A potential issue was identified where the number of Monte Carlo samples might be hardcoded, overriding configuration.

### 2. Model Training (`model_training/`)
A pipeline to fine-tune the base Kronos model for specific assets (QQQ).
- **Entry Point**: `train_predictor.py`
- **Config**: `model_training/config.py`
- **Dataset**: Uses a custom 3-year QQQ dataset (`intraday_dataset.py`).
- **Goal**: Specializes the general-purpose Kronos model for high-accuracy intraday forecasting.

### 3. Backtesting Suite (`backtesting/`)
Evaluates model performance on historical data.
- **Entry Point**: `backtesting/run_backtest.py`
- **Components**:
    - `BacktestEngine`: Manages the simulation loop.
    - `metrics.py`: Calculates accuracy, directional correctness, etc.
    - `visualize.py`: Generates plots of predictions vs. actuals.

### 4. Live Chart Service (`live_chart_prediction/`)
(Inferred) A web-based service to visualize predictions in real-time.
- Contains `main.py` (likely FastAPI/Flask) and `templates/` for the UI.

## Workflow
1.  **Data Prep & Training**: Use `model_training/` to fine-tune Kronos on QQQ data.
2.  **Evaluation**: Run `backtesting/run_backtest.py` to verify performance metrics.
3.  **Deployment/Usage**: Use `cli_kronos_prob_qqq.py` for daily/intraday predictions.

## Key Files
- `Kronos/model/kronos.py`: Core model definition.
- `cli_kronos_prob_qqq.py`: Main CLI for predictions.
- `model_training/train_predictor.py`: Fine-tuning script.
- `backtesting/run_backtest.py`: Backtesting runner.
- `config.yaml` (or similar): Configuration for predictions.

## Current Status & Notes
- The system is set up for Python environments (win32 detected).
- **Action Item**: Verify and fix the hardcoded sample count in `cli_kronos_prob_qqq.py`.
- **Action Item**: Confirm model loading mechanisms in `backtesting` and `live_chart_prediction`.
