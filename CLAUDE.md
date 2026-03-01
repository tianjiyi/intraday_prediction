# Intraday Prediction Project

## Project Overview
An agentic AI trading platform for intraday financial market analysis and prediction. Combines the Kronos foundation model for probabilistic forecasting with LLM-powered analysis (Google Gemini) and real-time market data streaming.

## Project Structure

```
intraday_predication/
├── platform/                  # PRODUCTION: Live web app (FastAPI + React)
│   ├── main.py                # FastAPI entry point (serves React SPA + API)
│   ├── config.yaml            # App configuration (no secrets - those go in .env)
│   ├── requirements.txt       # Python dependencies
│   ├── trading_rules.md       # AI-enforced trading rules
│   ├── services/              # Backend service modules
│   │   ├── prediction_service.py   # Kronos model integration (optional)
│   │   ├── llm_service.py         # Gemini LLM analysis
│   │   ├── news_service.py        # Alpaca News API
│   │   └── websocket_manager.py   # Real-time Alpaca streaming
│   ├── frontend/              # React SPA (Vite + TypeScript)
│   │   ├── src/
│   │   │   ├── components/chart/   # TradingChart, VolumeChart, StatsPanel, ChartToolbar
│   │   │   ├── components/         # Header, Layout, AiChat
│   │   │   ├── pages/              # ChartPage, HomePage
│   │   │   ├── hooks/              # useChartData, useResizableDivider, useWebSocket
│   │   │   ├── stores/             # Zustand stores (marketStore, uiStore, chatStore, newsStore)
│   │   │   ├── utils/              # chartHelpers, formatters
│   │   │   ├── api/                # API client + market endpoints
│   │   │   └── types/              # TypeScript interfaces (Candle, Prediction, etc.)
│   │   └── dist/                   # Built output (served by FastAPI)
│   ├── static/                # Legacy vanilla JS (bypassed by React SPA)
│   └── templates/             # Legacy Jinja templates (bypassed by React SPA)
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
│
├── Kronos/                    # Git submodule: Kronos foundation model
├── Dockerfile                 # CPU Docker image
├── docker-compose.yml         # App + Cloudflare Tunnel
├── .env.example               # Template for secrets
├── venv/                      # Python virtual environment (local dev)
└── .gitignore
```

## Development Environment
- **Platform**: Windows 11 (win32)
- **Python**: 3.12 with virtual environment at `./venv/`
- **GPU**: CUDA available for Kronos inference

## Common Commands

### Running the Platform (Local)
```bash
# Activate virtual environment
. venv/Scripts/activate

# Start the live trading dashboard
cd platform && python main.py
# Opens at http://localhost:5000
```

### Docker
```bash
docker compose up -d          # Start app + Cloudflare tunnel
docker compose logs -f app    # View logs
docker compose down           # Stop
```

### Cloud Deployment
```bash
# Any VM with Docker installed:
# 1. Clone repo
# 2. Copy .env.example -> .env, fill in secrets
# 3. Set CLOUDFLARE_TUNNEL_TOKEN in .env
# 4. docker compose up -d
```

### Frontend (React)
```bash
cd platform/frontend
npm install                   # Install dependencies
npm run dev                   # Dev server (proxies API to :5000)
npm run build                 # Build to dist/ (served by FastAPI)
```

### Dependencies
```bash
cd platform && pip install -r requirements.txt
```

## Key Technologies
- **Backend**: FastAPI with WebSocket support
- **Frontend**: React + Vite + TypeScript, TradingView Lightweight Charts v5
- **State Management**: Zustand (marketStore, uiStore, chatStore, newsStore)
- **ML Model**: Kronos-base (102.3M params) for probabilistic forecasting
- **LLM**: Google Gemini (gemini-2.5-pro) for market analysis
- **Data**: Alpaca Markets API (stocks + crypto)
- **Database**: SQLite (via SQLAlchemy) for agent data

## Configuration
- **Secrets**: All API keys live in `.env` (gitignored). See `.env.example` for the template.
- **Config**: `platform/config.yaml` has non-secret settings (model params, sampling, etc.)
- **Kronos**: Optional - set `model.enabled: false` in config.yaml to disable. Platform runs with data-only mode (no forecast bands).

## Remote Access (Cloudflare Tunnel)
- **URL**: https://ai.dnthetatechnologies.com
- **Auth**: Cloudflare Access (email OTP, jtian@dnthetatechnologies.com only)
- **Tunnel**: `my-pc` (ID `b551fdbf-fdb1-403f-8e8f-932ef3812609`)
- **Config**: `C:\Users\skysn\.cloudflared\config.yml`
- **Service**: `cloudflared` Windows service (Automatic, Running)
- **Routes**:
  - `rdp.dnthetatechnologies.com` → `tcp://localhost:3389` (RDP)
  - `ai.dnthetatechnologies.com` → `http://localhost:5000` (Chart app)
- **WebSocket**: Uses `wss://` automatically when page is served over HTTPS

## Important Reminders
1. Never commit API keys or credentials (.env is gitignored)
2. The `data/` directory is gitignored - contains large model weights and datasets
3. Always check for existing code conventions before making changes
4. Consider market hours and timezone handling (Eastern time for RTH)
5. The Kronos submodule must be initialized: `git submodule update --init`
