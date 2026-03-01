# Intraday Prediction Project

## Project Overview
An agentic AI trading platform for intraday financial market analysis and prediction. Combines the Kronos foundation model for probabilistic forecasting with LLM-powered analysis (Google Gemini) and real-time market data streaming.

## Project Structure

```
intraday_predication/
├── platform/                  # PRODUCTION: Live web app (FastAPI + TradingView)
│   ├── main.py                # FastAPI entry point
│   ├── config.yaml            # App configuration (API keys, model settings)
│   ├── requirements.txt       # Python dependencies
│   ├── trading_rules.md       # AI-enforced trading rules
│   ├── services/              # Backend service modules
│   │   ├── prediction_service.py   # Kronos model integration
│   │   ├── llm_service.py         # Gemini LLM analysis
│   │   ├── news_service.py        # Alpaca News API
│   │   └── websocket_manager.py   # Real-time Alpaca streaming
│   ├── static/js/chart.js    # TradingView chart + WebSocket client
│   ├── static/css/style.css   # Dark theme styling
│   └── templates/index.html   # Main dashboard template
│
├── research/                  # RESEARCH: Training, experiments, POCs
│   ├── kronos/                # Kronos fine-tuning & testing
│   ├── pattern_recognition/   # YOLO pattern detection
│   ├── market_regime/         # HMM regime detection
│   ├── backtesting/           # Backtest engine
│   ├── model_training/        # Custom model training
│   └── scripts/               # One-off debug/POC scripts
│
├── data/                      # DATA: All artifacts (gitignored)
│   ├── historical/            # Historical OHLCV data
│   ├── kronos_training/       # Kronos training datasets
│   ├── kronos_checkpoints/    # Fine-tuned model weights
│   ├── yolo_datasets/         # YOLO training images/labels
│   ├── yolo_models/           # YOLO .pt weights
│   ├── model_checkpoints/     # Other model checkpoints
│   └── test_outputs/          # Test/backtest results
│
├── Kronos/                    # Git submodule: Kronos foundation model
├── venv/                      # Python virtual environment
└── .gitignore
```

## Development Environment
- **Platform**: Windows 11 (win32)
- **Python**: 3.12 with virtual environment at `./venv/`
- **GPU**: CUDA available for Kronos inference

## Common Commands

### Running the Platform
```bash
# Activate virtual environment
. venv/Scripts/activate

# Start the live trading dashboard
cd platform && python main.py
# Opens at http://127.0.0.1:5000
```

### Dependencies
```bash
cd platform && pip install -r requirements.txt
```

## Key Technologies
- **Backend**: FastAPI with WebSocket support
- **Frontend**: TradingView Lightweight Charts v4.1.0
- **ML Model**: Kronos-base (102.3M params) for probabilistic forecasting
- **LLM**: Google Gemini (gemini-2.5-pro) for market analysis
- **Data**: Alpaca Markets API (stocks + crypto)
- **Database**: SQLite (via SQLAlchemy) for agent data

## Important Reminders
1. Never commit API keys or credentials (config.yaml is gitignored)
2. The `data/` directory is gitignored - contains large model weights and datasets
3. Always check for existing code conventions before making changes
4. Consider market hours and timezone handling (Eastern time for RTH)
5. The Kronos submodule must be initialized: `git submodule update --init`
