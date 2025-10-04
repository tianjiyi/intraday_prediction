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
    """Initialize the prediction service"""
    global prediction_service
    try:
        prediction_service = PredictionService()
        logger.info("Prediction service initialized successfully")
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
async def get_initial_data():
    """Get initial historical data and latest prediction"""
    try:
        if not prediction_service:
            await initialize_prediction_service()
            if not prediction_service:
                return JSONResponse({"error": "Prediction service not available"}, status_code=503)
        
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
