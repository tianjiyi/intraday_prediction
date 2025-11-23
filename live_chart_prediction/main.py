#!/usr/bin/env python3
"""
FastAPI server with native WebSocket support for real-time QQQ predictions
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import logging.config

# FastAPI and async imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append("..")

from prediction_service import PredictionService
from websocket_manager import AlpacaWebSocketManager
from llm_service import LLMService
from news_service import NewsService

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

# Global services
prediction_service: Optional[PredictionService] = None
websocket_manager: Optional[AlpacaWebSocketManager] = None
llm_service: Optional[LLMService] = None
news_service: Optional[NewsService] = None

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

# Pydantic models
class StartStreamRequest(BaseModel):
    symbol: str = "BTC/USD"
    timeframe: str = "1Min"

async def initialize_prediction_service():
    """Initialize the prediction service and related services"""
    global prediction_service, llm_service, news_service
    try:
        prediction_service = PredictionService()
        logger.info("Prediction service initialized successfully")

        # Initialize LLM service using prediction service's config
        try:
            llm_service = LLMService(prediction_service.config)
            if llm_service.is_available():
                logger.info("LLM service initialized successfully")
            else:
                logger.warning("LLM service initialized but not available (check API key)")
        except Exception as e:
            logger.warning(f"LLM service initialization failed: {e}")

        # Initialize News service
        try:
            news_service = NewsService(prediction_service.config)
            if news_service.is_available():
                logger.info("News service initialized successfully")
            else:
                logger.warning("News service initialized but not available")
        except Exception as e:
            logger.warning(f"News service initialization failed: {e}")

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

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting FastAPI server...")
    
    # Initialize services
    await initialize_prediction_service()
    await initialize_websocket_manager()
    
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
    
    logger.info("FastAPI server shutdown complete")

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chart page"""
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

        if request.analysis_type in ["sentiment", "full"]:
            results['sentiment'] = await llm_service.analyze_sentiment(symbol, news_items)
            results['news_count'] = len(news_items)

        if request.analysis_type in ["highlights"]:
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
    chat_history: Optional[List[Dict[str, str]]] = None


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

        # Get chat response
        response = await llm_service.chat_with_context(
            symbol=symbol,
            user_message=request.message,
            prediction_data=prediction,
            chat_history=request.chat_history
        )

        return {
            "response": response,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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
        "trading_rules_loaded": bool(llm_service.trading_rules) if llm_service else False
    }


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
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with uvicorn...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5000,
        reload=False,
        log_config=LOG_CONFIG,
        log_level="info",
    )
