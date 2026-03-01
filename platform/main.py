#!/usr/bin/env python3
"""
FastAPI server with native WebSocket support for real-time QQQ predictions
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging
import logging.config

import yaml
from dotenv import load_dotenv

# FastAPI and async imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for Kronos submodule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Kronos"))

# Load .env file (supports both platform/.env and project root .env)
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path)

from services.prediction_service import PredictionService
from services.websocket_manager import AlpacaWebSocketManager
from services.llm_service import LLMService
from services.news_service import NewsService
from services.twitter_service import TwitterService
from services.agent_memory_service import AgentMemoryService
from services.polymarket_service import PolymarketService
from services.news_monitor_service import NewsMonitorService

# Load config once at module level - shared by all services
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path) as _f:
    app_config = yaml.safe_load(_f)


# Setup logging with timestamps
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": "%(asctime)s %(levelname)s %(client_addr)s - \"%(request_line)s\" %(status_code)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        "__main__": {"handlers": ["default"], "level": "INFO"},
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Kronos QQQ Prediction API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# React frontend build directory
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
HAS_REACT_BUILD = os.path.isdir(FRONTEND_DIST) and os.path.isfile(
    os.path.join(FRONTEND_DIST, "index.html")
)
if HAS_REACT_BUILD:
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="react-assets")

# Global services
prediction_service: Optional[PredictionService] = None
websocket_manager: Optional[AlpacaWebSocketManager] = None
llm_service: Optional[LLMService] = None
news_service: Optional[NewsService] = None
twitter_service: Optional[TwitterService] = None
agent_memory: Optional[AgentMemoryService] = None
polymarket_service: Optional[PolymarketService] = None
news_monitor: Optional[NewsMonitorService] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()


def parse_draw_commands(response_text: str) -> List[Dict]:
    """
    Parse drawing commands from LLM response.

    Supports multiple formats:
    - ```DRAW_COMMAND {...} ```
    - `DRAW_COMMAND {...} `
    - {"commands": [...]} array format
    - Individual {"type": "..."} objects

    Returns:
        List of drawing command dictionaries
    """
    import re
    commands = []

    # Pattern 1: Triple backticks with DRAW_COMMAND
    pattern1 = r'```DRAW_COMMAND\s*\n?(.*?)\n?```'
    matches = re.findall(pattern1, response_text, re.DOTALL | re.IGNORECASE)

    # Pattern 2: Single backticks with DRAW_COMMAND
    pattern2 = r'`DRAW_COMMAND\s*\n?(.*?)\n?`'
    matches += re.findall(pattern2, response_text, re.DOTALL | re.IGNORECASE)

    # Pattern 3: Just look for JSON blocks that look like draw commands
    # This catches cases where LLM doesn't use proper DRAW_COMMAND tags
    pattern3 = r'`\s*(\{[^`]*"(?:type|commands)"[^`]*\})\s*`'
    matches += re.findall(pattern3, response_text, re.DOTALL)

    for match in matches:
        try:
            # Parse the JSON
            data = json.loads(match.strip())

            # Handle {"commands": [...]} array format
            if isinstance(data, dict) and 'commands' in data:
                for cmd in data['commands']:
                    if isinstance(cmd, dict) and 'type' in cmd:
                        commands.append(cmd)
                        logger.info(f"Parsed draw command from array: {cmd}")
            # Handle single command {"type": "..."}
            elif isinstance(data, dict) and 'type' in data:
                commands.append(data)
                logger.info(f"Parsed draw command: {data}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse draw command JSON: {e}")
            continue

    return commands


# Pydantic models
class StartStreamRequest(BaseModel):
    symbol: str = "BTC/USD"
    timeframe: str = "1Min"

async def initialize_prediction_service():
    """Initialize the prediction service and related services"""
    global prediction_service, llm_service, news_service, twitter_service
    try:
        prediction_service = PredictionService(app_config)
        logger.info(f"Prediction service initialized (model_available={prediction_service.model_available})")

        # Initialize LLM service using shared config
        try:
            llm_service = LLMService(app_config)
            if llm_service.is_available():
                logger.info("LLM service initialized successfully")
            else:
                logger.warning("LLM service initialized but not available (check API key)")
        except Exception as e:
            logger.warning(f"LLM service initialization failed: {e}")

        # Initialize News service using shared config
        try:
            news_service = NewsService(app_config)
            if news_service.is_available():
                logger.info("News service initialized successfully")
            else:
                logger.warning("News service initialized but not available")
        except Exception as e:
            logger.warning(f"News service initialization failed: {e}")

        # Initialize Twitter/X service
        try:
            twitter_service = TwitterService(app_config)
        except Exception as e:
            logger.warning(f"Twitter service initialization failed: {e}")

        return True
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        return False

async def initialize_websocket_manager():
    """Initialize WebSocket manager for Alpaca streaming"""
    global websocket_manager
    try:
        # Create callback for FastAPI WebSocket broadcasting
        async def websocket_callback(message):
            await manager.broadcast(message)
        # Capture the FastAPI event loop so the stream thread can safely emit
        loop = asyncio.get_running_loop()
        websocket_manager = AlpacaWebSocketManager(
            fastapi_callback=websocket_callback,
            fastapi_loop=loop
        )
        logger.info("WebSocket manager initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket manager: {e}")
        return False

async def initialize_memory_service():
    """Initialize the agent memory service (PostgreSQL + embeddings)."""
    global agent_memory
    try:
        agent_memory = AgentMemoryService(app_config)
        ok = await agent_memory.initialize()
        if ok:
            logger.info(
                f"Agent memory service initialized "
                f"(db={agent_memory.is_available()}, "
                f"embeddings={agent_memory.embeddings_available()})"
            )
        else:
            logger.warning("Agent memory service disabled or unavailable")
            agent_memory = None
    except Exception as e:
        logger.warning(f"Agent memory service init failed (non-fatal): {e}")
        agent_memory = None


async def initialize_news_monitor():
    """Initialize the background news monitoring service."""
    global polymarket_service, news_monitor
    try:
        polymarket_service = PolymarketService(app_config)
        news_monitor = NewsMonitorService(
            config=app_config,
            news_service=news_service,
            twitter_service=twitter_service,
            polymarket_service=polymarket_service,
            broadcast_callback=manager.broadcast,
        )
        await news_monitor.start()
    except Exception as e:
        logger.warning(f"News monitor init failed (non-fatal): {e}")
        news_monitor = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting FastAPI server...")

    # Initialize services
    await initialize_prediction_service()
    await initialize_websocket_manager()
    await initialize_memory_service()
    await initialize_news_monitor()

    logger.info("FastAPI server startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down FastAPI server...")

    # Stop WebSocket manager if running
    if websocket_manager:
        try:
            websocket_manager.stop()
        except Exception as e:
            logger.warning(f"Error stopping WebSocket manager: {e}")

    # Stop news monitor
    if news_monitor:
        try:
            await news_monitor.stop()
        except Exception as e:
            logger.warning(f"Error stopping news monitor: {e}")

    # Close memory service DB connections
    if agent_memory:
        try:
            await agent_memory.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down memory service: {e}")

    logger.info("FastAPI server shutdown complete")

# Health check endpoint (for Docker HEALTHCHECK / load balancers)
@app.get("/api/health")
async def health():
    """Return service health status"""
    return {
        "status": "ok",
        "services": {
            "prediction": prediction_service is not None,
            "kronos_model": getattr(prediction_service, 'model_available', False),
            "llm": llm_service.is_available() if llm_service else False,
            "news": news_service.is_available() if news_service else False,
            "twitter": twitter_service.is_available() if twitter_service else False,
            "websocket": websocket_manager is not None,
            "memory": agent_memory.is_available() if agent_memory else False,
            "embeddings": agent_memory.embeddings_available() if agent_memory else False,
            "news_monitor": news_monitor is not None and news_monitor._running,
            "polymarket": polymarket_service.is_available() if polymarket_service else False,
        }
    }


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve React SPA or legacy template"""
    if HAS_REACT_BUILD:
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/initial_data")
async def get_initial_data(symbol: Optional[str] = None, timeframe: Optional[int] = None):
    """Get initial historical data and latest prediction"""
    try:
        if not prediction_service:
            await initialize_prediction_service()
            if not prediction_service:
                return JSONResponse({"error": "Prediction service not available"}, status_code=503)
        
        # Update settings if provided
        if symbol or timeframe:
            prediction_service.update_settings(symbol=symbol, timeframe_minutes=timeframe)
        
        # Get historical data
        historical_data = prediction_service.get_historical_data()
        
        # Get latest prediction
        prediction = prediction_service.get_latest_prediction()
        
        return {
            "historical": historical_data,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting initial data: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/latest_prediction")
async def get_latest_prediction():
    """Get the latest prediction"""
    try:
        if not prediction_service:
            await initialize_prediction_service()
            if not prediction_service:
                return JSONResponse({"error": "Prediction service not available"}, status_code=503)
        
        prediction = prediction_service.get_latest_prediction()
        return {
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

 

@app.post("/api/start_stream")
async def start_stream(request: StartStreamRequest):
    """Start WebSocket stream for a symbol"""
    try:
        if not websocket_manager:
            await initialize_websocket_manager()
            if not websocket_manager:
                return JSONResponse({"error": "WebSocket manager not available"}, status_code=503)
        
        symbol = request.symbol
        timeframe = request.timeframe
        
        # Stop existing stream first
        try:
            websocket_manager.stop()
        except Exception as e:
            logger.warning(f"Error stopping existing stream: {e}")
        
        # Subscribe to the symbol
        success = await websocket_manager.subscribe_to_symbol(symbol, timeframe)
        if success:
            # Start the stream
            await websocket_manager.start()
            logger.info(f"Started WebSocket stream for {symbol}")
            
            # Broadcast to all connected WebSocket clients
            await manager.broadcast({
                "type": "stream_started",
                "symbol": symbol,
                "timeframe": timeframe
            })
            
            return {"status": "started", "symbol": symbol}
        else:
            return JSONResponse({"error": f"Failed to subscribe to {symbol}"}, status_code=500)
            
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/stop_stream")
async def stop_stream():
    """Stop current WebSocket stream"""
    try:
        if websocket_manager:
            websocket_manager.stop()
            logger.info("WebSocket stream stopped")
            
            # Broadcast to all connected WebSocket clients
            await manager.broadcast({
                "type": "stream_stopped"
            })
            
            return {"status": "stopped"}
        else:
            return {"status": "not running"}
            
    except Exception as e:
        logger.error(f"Error stopping stream: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/generate_prediction")
async def generate_prediction_endpoint():
    """Generate a new prediction on demand"""
    try:
        if not prediction_service:
            await initialize_prediction_service()
            if not prediction_service:
                return JSONResponse({"error": "Prediction service not available"}, status_code=503)
        
        # Generate new prediction
        prediction = prediction_service.generate_new_prediction()
        historical_data = prediction_service.get_historical_data()
        
        # Broadcast to all connected WebSocket clients
        await manager.broadcast({
            "type": "prediction_update",
            "prediction": prediction,
            "historical": historical_data,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "historical": historical_data,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============== LLM Analysis Endpoints ==============

class AnalysisRequest(BaseModel):
    """Request model for LLM analysis"""
    symbol: Optional[str] = None  # Trading symbol (defaults to current)
    analysis_type: str = "full"  # "technical", "rules", "sentiment", "full", "highlights"
    proposed_action: Optional[str] = None  # For rule checking

@app.post("/api/analyze")
async def analyze_endpoint(request: AnalysisRequest):
    """
    Generate LLM analysis based on current market data and Kronos predictions

    analysis_type options:
    - "technical": Technical analysis with Kronos predictions
    - "rules": Check trading rules compliance
    - "sentiment": News sentiment analysis
    - "full": All of the above
    """
    try:
        if not prediction_service:
            await initialize_prediction_service()
            if not prediction_service:
                return JSONResponse({"error": "Prediction service not available"}, status_code=503)

        if not llm_service or not llm_service.is_available():
            return JSONResponse({
                "error": "LLM service not available. Please configure GEMINI_API_KEY in config.yaml or environment."
            }, status_code=503)

        # Use provided symbol or get from prediction service
        symbol = request.symbol
        if symbol:
            # Update prediction service to use the requested symbol
            prediction_service.update_settings(symbol=symbol)

        # Get current prediction data
        prediction = prediction_service.get_latest_prediction()
        if not prediction:
            prediction = prediction_service.generate_new_prediction()

        if not prediction:
            return JSONResponse({"error": "No prediction data available"}, status_code=500)

        # Use symbol from prediction if not explicitly provided
        if not symbol:
            symbol = prediction.get('symbol', 'QQQ')
        results = {}

        if request.analysis_type in ["technical", "full"]:
            results['technical'] = await llm_service.analyze_technical(
                symbol, prediction, {}
            )

        if request.analysis_type in ["rules", "full"]:
            results['rules'] = await llm_service.check_trading_rules(
                symbol, prediction, request.proposed_action
            )

        # Fetch news for sentiment and highlights
        news_items = []
        if request.analysis_type in ["sentiment", "full", "highlights"]:
            if news_service and news_service.is_available():
                llm_config = prediction_service.config.get('llm', {})
                news_hours = llm_config.get('news_hours_back', 72)
                news_limit = llm_config.get('news_limit', 25)

                # Fetch symbol-specific news
                news_items = await news_service.get_news(symbol, limit=news_limit, hours_back=news_hours)

                # If sparse results, also fetch general market news
                if len(news_items) < 5:
                    logger.info(f"Only {len(news_items)} news for {symbol}, fetching general market news...")
                    market_news = await news_service.get_market_news(limit=news_limit, hours_back=news_hours)
                    # Combine and deduplicate by id
                    existing_ids = {item['id'] for item in news_items}
                    for item in market_news:
                        if item['id'] not in existing_ids:
                            news_items.append(item)
                    logger.info(f"Combined total: {len(news_items)} news items")

            # Merge Twitter/X posts into news feed
            if twitter_service and twitter_service.is_available():
                twitter_config = app_config.get('twitter', {})
                tweet_limit = twitter_config.get('search_limit', 20)
                tweet_hours = twitter_config.get('hours_back', 24)
                tweets = await twitter_service.search_tweets(symbol, limit=tweet_limit, hours_back=tweet_hours)
                if tweets:
                    existing_ids = {item['id'] for item in news_items}
                    for tweet in tweets:
                        if tweet['id'] not in existing_ids:
                            news_items.append(tweet)
                    # Sort combined feed by created_at descending
                    news_items.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                    logger.info(f"Added {len(tweets)} tweets. Total news+tweets: {len(news_items)}")

        if request.analysis_type in ["sentiment", "full"]:
            results['sentiment'] = await llm_service.analyze_sentiment(symbol, news_items)
            results['news_count'] = len(news_items)

        if request.analysis_type in ["highlights", "full"]:
            results['highlights'] = await llm_service.generate_daily_highlights(
                symbol, prediction, news_items
            )
            results['news_count'] = len(news_items)

        results['timestamp'] = datetime.now().isoformat()
        results['symbol'] = symbol

        return results

    except Exception as e:
        logger.error(f"Error generating LLM analysis: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


class ChatRequest(BaseModel):
    """Request model for chat"""
    message: str
    symbol: Optional[str] = None
    session_id: Optional[str] = None  # UUID string for persistent session
    chat_history: Optional[List[Dict[str, str]]] = None
    chart_screenshot: Optional[str] = None  # base64-encoded PNG screenshot of the chart


class DrawCommand(BaseModel):
    """Request model for drawing commands"""
    type: str  # 'hline', 'trendline', 'clear', 'remove'
    price: Optional[float] = None
    startTime: Optional[int] = None
    startPrice: Optional[float] = None
    endTime: Optional[int] = None
    endPrice: Optional[float] = None
    color: Optional[str] = None
    label: Optional[str] = None
    id: Optional[str] = None


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat with the LLM using current market context

    The LLM has access to:
    - Current price and Kronos predictions
    - All technical indicators (SMAs, VWAP, Bollinger Bands)
    - Daily fundamentals (RSI, CCI, trend)
    - Key price levels (prev day high/low, 3-day high/low)
    - Trading rules for compliance checking
    """
    try:
        if not prediction_service:
            await initialize_prediction_service()
            if not prediction_service:
                return JSONResponse({"error": "Prediction service not available"}, status_code=503)

        if not llm_service or not llm_service.is_available():
            return JSONResponse({
                "error": "LLM service not available. Please configure GEMINI_API_KEY."
            }, status_code=503)

        # Use provided symbol or get from prediction service
        symbol = request.symbol
        if symbol:
            prediction_service.update_settings(symbol=symbol)

        # Get current prediction data
        prediction = prediction_service.get_latest_prediction()
        if not prediction:
            prediction = prediction_service.generate_new_prediction()

        if not prediction:
            return JSONResponse({"error": "No prediction data available"}, status_code=500)

        # Use symbol from prediction if not explicitly provided
        if not symbol:
            symbol = prediction.get('symbol', 'QQQ')

        # Build agent memory context (graceful if unavailable)
        memory_context = ""
        session_ctx = None
        if agent_memory and agent_memory.is_available():
            try:
                # Get or create session for Tier C
                session_ctx = agent_memory.get_or_create_session(
                    session_id=request.session_id, symbol=symbol
                )
                # Build combined memory context from all tiers
                memory_context = await agent_memory.build_memory_context(
                    query=request.message,
                    symbol=symbol,
                    session_id=str(session_ctx.session_id),
                )
            except Exception as e:
                logger.warning(f"Memory context build failed (non-fatal): {e}")

        # Get chat response
        response = await llm_service.chat_with_context(
            symbol=symbol,
            user_message=request.message,
            prediction_data=prediction,
            chat_history=request.chat_history,
            chart_screenshot=request.chart_screenshot,
            memory_context=memory_context,
        )

        # Store messages in memory (async, non-blocking)
        if agent_memory and agent_memory.is_available() and session_ctx:
            try:
                sid = session_ctx.session_id
                await agent_memory.store_chat_message(sid, "user", request.message, symbol)
                await agent_memory.store_chat_message(sid, "assistant", response, symbol)
                agent_memory.add_turn(sid, "user", request.message)
                agent_memory.add_turn(sid, "assistant", response)
            except Exception as e:
                logger.warning(f"Failed to store chat messages: {e}")

        # Parse and execute any drawing commands in the response
        draw_commands = parse_draw_commands(response)
        for cmd in draw_commands:
            try:
                await manager.broadcast({
                    "type": "draw_command",
                    "command": cmd
                })
                logger.info(f"Executed draw command from chat: {cmd}")
            except Exception as e:
                logger.error(f"Failed to execute draw command: {e}")

        return {
            "response": response,
            "symbol": symbol,
            "session_id": str(session_ctx.session_id) if session_ctx else None,
            "timestamp": datetime.now().isoformat(),
            "draw_commands": draw_commands
        }

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# Store for drawings (in-memory for now)
chart_drawings = []


@app.post("/api/draw")
async def draw_on_chart(command: DrawCommand):
    """
    Send a drawing command to all connected chart clients.

    This endpoint broadcasts drawing commands via WebSocket to render
    drawings on the chart. Can be called by MCP server or directly.

    Supported command types:
    - hline: Horizontal line at a price
    - trendline: Line between two points
    - clear: Clear all drawings
    - remove: Remove a specific drawing by ID
    """
    global chart_drawings

    try:
        # Convert command to dict
        command_dict = command.dict(exclude_none=True)
        logger.info(f"Drawing command received: {command_dict}")

        # Track drawings in memory
        if command.type == 'clear':
            chart_drawings = []
        elif command.type == 'remove':
            chart_drawings = [d for d in chart_drawings if d.get('id') != command.id]
        elif command.type in ['hline', 'trendline']:
            chart_drawings.append(command_dict)

        # Broadcast to all WebSocket clients
        await manager.broadcast({
            "type": "draw_command",
            "command": command_dict
        })

        return {"success": True, "command": command_dict}

    except Exception as e:
        logger.error(f"Error processing draw command: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/drawings")
async def get_drawings():
    """Get all current drawings on the chart."""
    return {"drawings": chart_drawings}


@app.get("/api/news")
async def get_news_endpoint(symbol: Optional[str] = None, limit: int = 10, hours: int = 24):
    """
    Fetch recent news for a symbol

    Args:
        symbol: Trading symbol (defaults to current symbol)
        limit: Maximum number of news items
        hours: How many hours back to search
    """
    global news_service
    try:
        if not news_service:
            if prediction_service:
                news_service = NewsService(prediction_service.config)

        if not news_service or not news_service.is_available():
            return JSONResponse({
                "error": "News service not available",
                "news": []
            }, status_code=503)

        # Use provided symbol or get from prediction service
        if not symbol and prediction_service:
            symbol = prediction_service.symbol
        symbol = symbol or "QQQ"

        news_items = await news_service.get_news(symbol, limit=limit, hours_back=hours)

        # Merge Twitter/X posts
        if twitter_service and twitter_service.is_available():
            twitter_config = app_config.get('twitter', {})
            tweets = await twitter_service.search_tweets(
                symbol,
                limit=twitter_config.get('search_limit', 20),
                hours_back=min(hours, twitter_config.get('hours_back', 24))
            )
            if tweets:
                existing_ids = {item['id'] for item in news_items}
                for tweet in tweets:
                    if tweet['id'] not in existing_ids:
                        news_items.append(tweet)
                news_items.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # Get simple sentiment keywords
        sentiment = news_service.get_sentiment_keywords(news_items)

        return {
            "symbol": symbol,
            "news": news_items,
            "count": len(news_items),
            "sentiment_keywords": sentiment,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return JSONResponse({"error": str(e), "news": []}, status_code=500)


@app.get("/api/llm_status")
async def llm_status_endpoint():
    """Check LLM and News service status"""
    available = llm_service.is_available() if llm_service else False

    # Determine reason if not available
    reason = None
    if not available:
        if not llm_service:
            reason = "LLM service not initialized"
        elif not llm_service.api_key:
            reason = "GEMINI_API_KEY not configured"
        else:
            reason = "LLM model not loaded"

    return {
        "available": available,
        "model": llm_service.model_name if llm_service else None,
        "provider": llm_service.provider if llm_service else None,
        "reason": reason,
        "news_available": news_service.is_available() if news_service else False,
        "twitter_available": twitter_service.is_available() if twitter_service else False,
        "trading_rules_loaded": bool(llm_service.trading_rules) if llm_service else False
    }


# ============== News Monitor Endpoints ==============


@app.get("/api/news/feed")
async def get_news_feed(category: str = "all", limit: int = 50):
    """Get current news buffer (for initial page load / polling)."""
    if not news_monitor:
        return JSONResponse({"error": "News monitor not running"}, status_code=503)
    items = news_monitor.get_buffer(category=category, limit=limit)
    return {
        "items": items,
        "count": len(items),
        "unread_count": news_monitor.get_unread_count(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/news/monitor/status")
async def news_monitor_status():
    """Health check for the news monitoring service."""
    return {
        "running": news_monitor is not None and news_monitor._running,
        "buffer_size": len(news_monitor._buffer) if news_monitor else 0,
        "sources": {
            "alpaca": news_service.is_available() if news_service else False,
            "twitter": twitter_service.is_available() if twitter_service else False,
            "polymarket": polymarket_service.is_available() if polymarket_service else False,
        },
    }


# ============== Agent Memory Endpoints ==============


class MemorySearchRequest(BaseModel):
    query: str
    symbol: Optional[str] = None
    memory_type: Optional[str] = None
    limit: int = 5


class StoreDecisionRequest(BaseModel):
    decision_text: str
    parsed_rule: Optional[Dict[str, Any]] = None
    source: str = "manual"


class StoreMemoryRequest(BaseModel):
    content: str
    memory_type: str = "experience"
    source: str = "manual"
    symbol: Optional[str] = None
    importance_score: float = 0.50


class SetPreferenceRequest(BaseModel):
    category: str
    key: str
    value: Any


@app.get("/api/memory/search")
async def memory_search(query: str, symbol: Optional[str] = None, limit: int = 5):
    """Semantic search over agent memories (vector RAG)."""
    if not agent_memory or not agent_memory.is_available():
        return JSONResponse({"error": "Memory service not available"}, status_code=503)
    results = await agent_memory.recall_memories(query, limit=limit, symbol=symbol)
    return {"results": results, "count": len(results)}


@app.post("/api/memory/store")
async def memory_store(request: StoreMemoryRequest):
    """Store a new memory manually."""
    if not agent_memory or not agent_memory.is_available():
        return JSONResponse({"error": "Memory service not available"}, status_code=503)
    result = await agent_memory.store_memory(
        content=request.content,
        memory_type=request.memory_type,
        source=request.source,
        symbol=request.symbol,
        importance_score=request.importance_score,
    )
    if result:
        return result
    return JSONResponse({"error": "Failed to store memory (embeddings unavailable?)"}, status_code=500)


@app.get("/api/memory/decisions")
async def memory_decisions(symbol: Optional[str] = None, active_only: bool = True):
    """Get trading decisions."""
    if not agent_memory or not agent_memory.is_available():
        return JSONResponse({"error": "Memory service not available"}, status_code=503)
    decisions = await agent_memory.get_active_decisions(symbol=symbol)
    return {"decisions": decisions, "count": len(decisions)}


@app.post("/api/memory/decisions")
async def memory_store_decision(request: StoreDecisionRequest):
    """Store a new trading decision."""
    if not agent_memory or not agent_memory.is_available():
        return JSONResponse({"error": "Memory service not available"}, status_code=503)
    result = await agent_memory.store_decision(
        decision_text=request.decision_text,
        parsed_rule=request.parsed_rule,
        source=request.source,
    )
    return result


@app.get("/api/memory/chat_history")
async def memory_chat_history(session_id: str, limit: int = 50):
    """Get chat history for a session."""
    if not agent_memory or not agent_memory.is_available():
        return JSONResponse({"error": "Memory service not available"}, status_code=503)
    import uuid as _uuid
    history = await agent_memory.get_chat_history(_uuid.UUID(session_id), limit=limit)
    return {"messages": history, "count": len(history)}


@app.get("/api/memory/preferences/{category}")
async def memory_preferences(category: str):
    """Get user preferences by category."""
    if not agent_memory or not agent_memory.is_available():
        return JSONResponse({"error": "Memory service not available"}, status_code=503)
    prefs = await agent_memory.get_preferences_by_category(category)
    return {"category": category, "preferences": prefs}


@app.put("/api/memory/preferences")
async def memory_set_preference(request: SetPreferenceRequest):
    """Set a user preference."""
    if not agent_memory or not agent_memory.is_available():
        return JSONResponse({"error": "Memory service not available"}, status_code=503)
    await agent_memory.set_user_preference(request.category, request.key, request.value)
    return {"status": "ok", "category": request.category, "key": request.key}


@app.get("/api/memory/signals")
async def memory_signals(symbol: Optional[str] = None, limit: int = 20):
    """Get recent trading signals."""
    if not agent_memory or not agent_memory.is_available():
        return JSONResponse({"error": "Memory service not available"}, status_code=503)
    signals = await agent_memory.get_recent_signals(symbol=symbol, limit=limit)
    return {"signals": signals, "count": len(signals)}


# ============== Market Regime Endpoints ==============


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await manager.send_personal_message({
            "type": "connected",
            "message": "Connected to prediction server"
        }, websocket)
        
        while True:
            # Listen for client messages
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "request_update":
                # Handle manual update request
                try:
                    if not prediction_service:
                        await initialize_prediction_service()
                    
                    if prediction_service:
                        # Generate new prediction
                        prediction = prediction_service.generate_new_prediction()
                        
                        # Broadcast to all connected clients
                        await manager.broadcast({
                            "type": "prediction_update",
                            "prediction": prediction,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Prediction service not available"
                        }, websocket)
                        
                except Exception as e:
                    logger.error(f"Error handling update request: {e}")
                    await manager.send_personal_message({
                        "type": "error",
                        "message": str(e)
                    }, websocket)
            
            elif message_type == "settings_changed":
                # Handle settings changes (ticker, timeframe, etc.)
                try:
                    logger.info(f"Settings changed: {data}")
                    
                    if not prediction_service:
                        await initialize_prediction_service()
                    
                    if prediction_service:
                        # Update settings in prediction service
                        symbol = data.get('ticker')
                        timeframe = data.get('timeframe')
                        
                        if symbol or timeframe:
                            prediction_service.update_settings(
                                symbol=symbol,
                                timeframe_minutes=timeframe
                            )
                        
                        # Generate new prediction with updated settings
                        prediction = prediction_service.generate_new_prediction()
                        
                        # Broadcast to all connected clients
                        await manager.broadcast({
                            "type": "prediction_update",
                            "historical": prediction_service.get_historical_data(),
                            "prediction": prediction,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        logger.info("Prediction updated due to settings change")
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Prediction service not available"
                        }, websocket)
                        
                except Exception as e:
                    logger.error(f"Error handling settings change: {e}")
                    await manager.send_personal_message({
                        "type": "error",
                        "message": str(e)
                    }, websocket)
            
            elif message_type == "check_for_new_data":
                # Check for new bar data and update prediction if available
                try:
                    if not prediction_service:
                        await initialize_prediction_service()
                    
                    if prediction_service:
                        # Check if we have new data
                        # Use configured days_to_fetch value
                        days_to_fetch = prediction_service.config['data'].get('days_to_fetch', 3)
                        current_data = prediction_service.fetch_historical_data(days_to_fetch)
                        
                        if current_data:
                            # Generate new prediction with fresh data
                            prediction = prediction_service.generate_prediction()
                            
                            # Broadcast update to all clients
                            await manager.broadcast({
                                "type": "prediction_update",
                                "historical": current_data,
                                "prediction": prediction,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            logger.info("Prediction updated with new bar data")
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Prediction service not available"
                        }, websocket)
                        
                except Exception as e:
                    logger.error(f"Error checking for new data: {e}")
                    await manager.send_personal_message({
                        "type": "error",
                        "message": str(e)
                    }, websocket)

            elif message_type == "news_ack":
                # Client acknowledges reading news, reset unread counter
                if news_monitor:
                    news_monitor.reset_unread_count()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# SPA catch-all — serves React index.html for client-side routes
# MUST be the last route to avoid shadowing /api/* and /ws
@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    if HAS_REACT_BUILD:
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with uvicorn...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_config=LOG_CONFIG,
        log_level="info",
    )
