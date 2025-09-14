#!/usr/bin/env python3
"""
Flask server with WebSocket support for real-time QQQ predictions
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import logging

# Add parent directory to path for imports
sys.path.append("..")

from prediction_service import PredictionService

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'kronos-qqq-prediction-key'
CORS(app)

# Initialize SocketIO with async mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize prediction service
prediction_service = None

def initialize_prediction_service():
    """Initialize the prediction service"""
    global prediction_service
    try:
        prediction_service = PredictionService()
        logger.info("Prediction service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        return False

@app.route('/')
def index():
    """Serve the main chart page"""
    return render_template('index.html')

@app.route('/api/initial_data')
def get_initial_data():
    """Get initial historical data and latest prediction"""
    try:
        if not prediction_service:
            if not initialize_prediction_service():
                return jsonify({"error": "Prediction service not available"}), 503
        
        # Get historical data
        historical_data = prediction_service.get_historical_data()
        
        # Get latest prediction
        prediction = prediction_service.get_latest_prediction()
        
        return jsonify({
            "historical": historical_data,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting initial data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest_prediction')
def get_latest_prediction():
    """Get the latest prediction"""
    try:
        if not prediction_service:
            if not initialize_prediction_service():
                return jsonify({"error": "Prediction service not available"}), 503
        
        prediction = prediction_service.get_latest_prediction()
        return jsonify({
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to prediction server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_update')
def handle_update_request():
    """Handle manual update request from client"""
    try:
        if not prediction_service:
            if not initialize_prediction_service():
                emit('error', {'message': 'Prediction service not available'})
                return
        
        # Generate new prediction
        prediction = prediction_service.generate_new_prediction()
        
        # Emit to all connected clients
        socketio.emit('prediction_update', {
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling update request: {e}")
        emit('error', {'message': str(e)})

@socketio.on('settings_changed')
def handle_settings_change(data):
    """Handle settings changes (ticker, timeframe, etc.)"""
    try:
        logger.info(f"Settings changed: {data}")
        
        if not prediction_service:
            if not initialize_prediction_service():
                emit('error', {'message': 'Prediction service not available'})
                return
        
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
        
        # Emit to all connected clients
        socketio.emit('prediction_update', {
            'historical': prediction_service.get_historical_data(),
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info("Prediction updated due to settings change")
        
    except Exception as e:
        logger.error(f"Error handling settings change: {e}")
        emit('error', {'message': str(e)})

@socketio.on('check_for_new_data')
def handle_data_check():
    """Check for new bar data and update prediction if available"""
    try:
        if not prediction_service:
            if not initialize_prediction_service():
                emit('error', {'message': 'Prediction service not available'})
                return
        
        # Check if we have new data (this could be enhanced to check timestamps)
        current_data = prediction_service.fetch_historical_data()
        
        if current_data:
            # Generate new prediction with fresh data
            prediction = prediction_service.generate_prediction()
            
            # Emit update to all clients
            socketio.emit('prediction_update', {
                'historical': current_data,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("Prediction updated with new bar data")
        
    except Exception as e:
        logger.error(f"Error checking for new data: {e}")
        emit('error', {'message': str(e)})

@app.route('/api/generate_prediction')
def generate_prediction_endpoint():
    """Generate a new prediction on demand"""
    try:
        if not prediction_service:
            if not initialize_prediction_service():
                return jsonify({"error": "Prediction service not available"}), 503
        
        # Generate new prediction
        prediction = prediction_service.generate_new_prediction()
        historical_data = prediction_service.get_historical_data()
        
        return jsonify({
            "historical": historical_data,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize prediction service on startup
    if initialize_prediction_service():
        logger.info("Starting Flask server with WebSocket support...")
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to start server - prediction service initialization failed")