#!/usr/bin/env python3
"""
WebSocket Manager for Alpaca Real-Time Data Streams
Handles real-time bar and trade updates for both stocks and crypto
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
import pytz
import yaml

# Add parent directory to path
sys.path.append("..")

# Import Alpaca streaming - using modern alpaca-py library
try:
    from alpaca.data.live import CryptoDataStream, StockDataStream
    from alpaca.data.enums import CryptoFeed
    from alpaca_trade_api.stream import Stream
    from alpaca_trade_api.common import URL
    ALPACA_PY_AVAILABLE = True
except ImportError:
    print("Warning: alpaca-py not installed. Install with: pip install alpaca-py")
    CryptoDataStream = None
    StockDataStream = None
    CryptoFeed = None
    ALPACA_PY_AVAILABLE = False
    # Fallback to old library
    try:
        from alpaca_trade_api.stream import Stream
        from alpaca_trade_api.common import URL
    except ImportError:
        print("Warning: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
        Stream = None

# Setup logging with timestamps (module-level)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class AlpacaWebSocketManager:
    """Manages WebSocket connections to Alpaca for real-time data"""
    
    def __init__(self, config_path="../config.yaml", fastapi_callback=None, fastapi_loop=None):
        """Initialize WebSocket manager for FastAPI native WebSockets"""
        self.config = self._load_config(config_path)
        self.fastapi_callback = fastapi_callback  # FastAPI WebSocket callback
        self.stream = None
        self.current_symbol = None
        self.current_timeframe = "1Min"
        self.is_connected = False
        self.is_connecting = False  # Track connection state
        self.timezone = pytz.timezone('US/Eastern')
        
        # Crypto connection tracking
        self.crypto_connection_success = False
        self.crypto_connection_timeout = False
        
        # Determine data/feed and trading environment
        alpaca_cfg = self.config.get('alpaca', {})
        # Single switch for market data feed (True => SIP/live, False => IEX)
        self.use_live_data = alpaca_cfg.get('use_live_data', True)
        # Trading env (only relevant if placing orders)
        self.account_env = alpaca_cfg.get('account_env', 'paper')  # 'paper' | 'live'
        self.is_paper_account = (self.account_env == 'paper')
        logger.info(f"Config loaded: use_live_data={self.use_live_data}, account_env={self.account_env}")
        
        # Initialize modern alpaca-py streams
        self.crypto_stream = None
        self.stock_stream = None
        self.fastapi_loop = fastapi_loop
        
        # Track current bar being formed
        self.current_bar = {}
        self.bar_start_time = None
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def _emit_to_clients(self, event_type, data):
        """Emit data to FastAPI WebSocket clients"""
        try:
            if self.fastapi_callback:
                # Ensure event type key cannot be overridden by payload keys
                message = {**data, "type": event_type}
                # If we have the FastAPI event loop (main app loop), schedule there
                if self.fastapi_loop:
                    import asyncio as _asyncio
                    try:
                        _asyncio.run_coroutine_threadsafe(self.fastapi_callback(message), self.fastapi_loop)
                    except Exception as e:
                        logger.error(f"Threadsafe emit failed: {e}")
                else:
                    await self.fastapi_callback(message)
                
        except Exception as e:
            logger.error(f"Error emitting to clients: {e}")
    
    def _is_crypto_symbol(self, symbol):
        """Check if symbol is crypto"""
        return '/' in symbol
    
    async def initialize_crypto_stream_modern(self):
        """Initialize modern alpaca-py crypto stream"""
        if not ALPACA_PY_AVAILABLE or not CryptoDataStream:
            logger.error("alpaca-py not available")
            return False
        
        try:
            # Create modern crypto stream with minimal parameters (match documentation exactly)
            self.crypto_stream = CryptoDataStream(
                self.config['ALPACA_KEY_ID'],        # api_key (positional)
                self.config['ALPACA_SECRET_KEY'],    # secret_key (positional)
                feed=CryptoFeed.US
            )
            
            logger.info("Modern alpaca-py crypto stream initialized with correct parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize modern crypto stream: {e}")
            return False
    
    async def initialize_stream(self):
        """Initialize Alpaca stream connection with proper cleanup"""
        if not Stream:
            logger.error("alpaca-trade-api not available")
            return False
        
        # Clean up any existing stream completely first
        await self._cleanup_existing_stream()
            
        try:
            # Configure base_url by trading env; data_feed by use_live_data
            base_url = 'https://paper-api.alpaca.markets' if self.is_paper_account else 'https://api.alpaca.markets'
            data_feed = 'sip' if self.use_live_data else 'iex'
            logger.info(f"Initializing WebSocket: env={self.account_env}, data_feed={data_feed}")
            
            # Create stream instance with appropriate configuration
            self.stream = Stream(
                key_id=self.config['ALPACA_KEY_ID'],
                secret_key=self.config['ALPACA_SECRET_KEY'],
                base_url=URL(base_url),
                data_feed=data_feed,
                raw_data=False
            )
            
            logger.info("Alpaca WebSocket stream initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize stream: {e}")
            return False
    
    async def _cleanup_existing_stream(self):
        """Properly cleanup existing stream connections"""
        if self.stream:
            try:
                logger.info("Cleaning up existing stream...")
                
                # Unsubscribe from all current subscriptions first
                await self.unsubscribe_all()
                
                # Stop the stream
                self.stream.stop()
                
                # Wait for cleanup
                await asyncio.sleep(2)
                
                # Reset stream reference
                self.stream = None
                self.is_connected = False
                
                logger.info("Stream cleanup completed")
                
            except Exception as e:
                logger.error(f"Error during stream cleanup: {e}")
                # Force reset even if cleanup failed
                self.stream = None
                self.is_connected = False
    
    async def subscribe_to_symbol(self, symbol, timeframe="1Min"):
        """Subscribe to real-time data for a symbol"""
        # Prevent concurrent connections
        if self.is_connecting:
            logger.warning(f"Already connecting to another symbol, skipping {symbol}")
            return False
            
        self.is_connecting = True
        
        try:
            # Clean up previous symbol subscription
            if self.current_symbol and self.current_symbol != symbol:
                await self.unsubscribe_all()
            
            self.current_symbol = symbol
            self.current_timeframe = timeframe
            
            # Determine if crypto or stock
            is_crypto = self._is_crypto_symbol(symbol)
            
            # Initialize appropriate stream based on symbol type and account type
            if is_crypto and self.use_live_data:
                # For crypto on live accounts, don't initialize traditional stream
                # We'll use modern alpaca-py crypto stream only
                pass
            else:
                # For stocks or non-live crypto, initialize traditional stream if needed
                if not self.stream:
                    success = await self.initialize_stream()
                    if not success:
                        return False
            
            if is_crypto:
                # Subscribe to crypto bars using modern alpaca-py for live accounts
                logger.info(f"Subscribing to crypto bars for {symbol}")
                if not self.use_live_data:
                    # Non-live data feed (IEX) may have limited/unsupported crypto streaming
                    logger.info(f"Using non-live data feed crypto path for {symbol}")
                    try:
                        self.stream.subscribe_crypto_bars(self._handle_crypto_bar, symbol)
                        self.stream.subscribe_crypto_trades(self._handle_crypto_trade, symbol)
                        logger.info(f"Subscribed to crypto handlers for {symbol} via non-live data feed")
                        
                        # Notify client of limited crypto capabilities but continue with historical data
                        await self._emit_to_clients('stream_info', {
                            'message': f'Crypto streaming initialized for {symbol} with limited real-time capabilities',
                            'symbol': symbol,
                            'mode': 'historical_primary',
                            'info': 'Using historical data with limited real-time updates'
                        })
                        
                    except Exception as e:
                        logger.warning(f"Crypto streaming not available on non-live data feed: {e}")
                        # Don't return False - continue with historical data mode
                        await self._emit_to_clients('stream_error', {
                            'message': 'Crypto real-time streaming unavailable on current data feed - using historical data only',
                            'symbol': symbol,
                            'suggestion': 'Charts will update with historical data and predictions',
                            'fallback': True
                        })
                        logger.info(f"Continuing with historical data mode for {symbol}")
                else:
                    # Live accounts - use modern alpaca-py crypto streaming
                    logger.info(f"Using live data crypto path for {symbol}")
                    success = await self.initialize_crypto_stream_modern()
                    if not success:
                        logger.error("Failed to initialize modern crypto stream")
                        return False
                    
                    # Use correct crypto symbol format for alpaca-py (keep slash)
                    alpaca_symbol = symbol
                    
                    # Subscribe using modern alpaca-py
                    self.crypto_stream.subscribe_bars(self._handle_crypto_bar_modern, alpaca_symbol)
                    self.crypto_stream.subscribe_trades(self._handle_crypto_trade_modern, alpaca_symbol)
                    # Increase intra-minute updates with quotes
                    try:
                        self.crypto_stream.subscribe_quotes(self._handle_crypto_quote_modern, alpaca_symbol)
                        logger.info(f"Subscribed to crypto quotes for {alpaca_symbol}")
                    except Exception as e:
                        logger.warning(f"Quote subscribe failed for {alpaca_symbol}: {e}")
                    logger.info(f"Subscribed to modern crypto stream for {alpaca_symbol}")
            else:
                # Subscribe to stock bars
                logger.info(f"Subscribing to stock bars for {symbol}")
                self.stream.subscribe_bars(self._handle_stock_bar, symbol)
                self.stream.subscribe_trades(self._handle_stock_trade, symbol)
            
            self.is_connected = True
            
            # Notify client of subscription
            await self._emit_to_clients('stream_status', {
                'connected': True,
                'symbol': symbol,
                'asset_type': 'crypto' if is_crypto else 'stock'
            })
                
            logger.info(f"Successfully subscribed to {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            self.is_connected = False
            return False
        finally:
            self.is_connecting = False
    
    async def _handle_stock_bar(self, bar):
        """Handle incoming stock bar data"""
        try:
            logger.debug(f"Stock bar received: {bar}")

            # Handle timestamp - could be datetime object or unix timestamp
            if hasattr(bar.timestamp, 'isoformat'):
                timestamp_str = bar.timestamp.isoformat()
            elif isinstance(bar.timestamp, (int, float)):
                try:
                    # Detect timestamp precision by number of digits
                    timestamp_str_len = len(str(int(abs(bar.timestamp))))

                    if timestamp_str_len >= 19:  # Nanoseconds (19-21 digits)
                        timestamp_val = bar.timestamp / 1_000_000_000
                    elif timestamp_str_len >= 13:  # Milliseconds (13-15 digits)
                        timestamp_val = bar.timestamp / 1000
                    elif timestamp_str_len >= 10:  # Seconds (10-12 digits)
                        timestamp_val = bar.timestamp
                    else:
                        raise ValueError(f"Timestamp too short: {bar.timestamp}")

                    # Validate timestamp is reasonable (between 1970 and 2100)
                    if timestamp_val < 0 or timestamp_val > 4102444800:
                        raise ValueError(f"Invalid timestamp value: {timestamp_val}")

                    timestamp_str = datetime.fromtimestamp(timestamp_val).isoformat()
                except (ValueError, OSError) as e:
                    logger.warning(f"Invalid timestamp {bar.timestamp}: {e}, using current time")
                    timestamp_str = datetime.now(pytz.timezone('US/Eastern')).isoformat()
            else:
                # Try to convert to string
                timestamp_str = str(bar.timestamp)

            # Reset current bar tracking when a completed bar arrives
            # This prevents mixing old trade data with new bars
            if bar.symbol in self.current_bar:
                del self.current_bar[bar.symbol]
                logger.debug(f"Reset current bar tracking for {bar.symbol}")

            # Convert bar to dict format
            bar_data = {
                'symbol': bar.symbol,
                'timestamp': timestamp_str,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'asset_type': 'stock'
            }

            # Emit completed bar to clients
            await self._emit_to_clients('bar_complete', bar_data)
            logger.info(f"Emitted stock bar for {bar.symbol}")

        except Exception as e:
            logger.error(f"Error handling stock bar: {e}")
    
    async def _handle_crypto_bar(self, bar):
        """Handle incoming crypto bar data"""
        try:
            logger.debug(f"Crypto bar received: {bar}")
            
            # Convert bar to dict format
            bar_data = {
                'symbol': bar.symbol,
                'timestamp': bar.timestamp.isoformat(),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume),
                'asset_type': 'crypto'
            }
            
            # Emit completed bar to clients
            await self._emit_to_clients('bar_complete', bar_data)
            logger.info(f"Emitted crypto bar for {bar.symbol}")
                
        except Exception as e:
            logger.error(f"Error handling crypto bar: {e}")
    
    async def _handle_stock_trade(self, trade):
        """Handle incoming stock trade for current bar updates"""
        try:
            # Update current bar with latest trade
            price = float(trade.price)
            symbol = trade.symbol

            # Get current minute boundary
            trade_time = trade.timestamp
            if hasattr(trade_time, 'timestamp'):
                minute_boundary = int(trade_time.timestamp() // 60) * 60
            else:
                minute_boundary = int(trade_time // 60) * 60

            # Check if we need to reset for a new minute
            if symbol in self.current_bar:
                if self.current_bar[symbol].get('minute_boundary') != minute_boundary:
                    # New minute started, emit the old bar as complete first
                    old_bar = self.current_bar[symbol].copy()
                    old_bar.pop('minute_boundary', None)
                    await self._emit_to_clients('bar_complete', {
                        'symbol': symbol,
                        'timestamp': trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp),
                        **old_bar,
                        'asset_type': 'stock'
                    })

                    # Reset for new minute
                    self.current_bar[symbol] = {
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': trade.size if hasattr(trade, 'size') else 0,
                        'minute_boundary': minute_boundary
                    }
                else:
                    # Update existing bar within same minute
                    bar = self.current_bar[symbol]
                    bar['high'] = max(bar['high'], price)
                    bar['low'] = min(bar['low'], price)
                    bar['close'] = price
                    if hasattr(trade, 'size'):
                        bar['volume'] += trade.size
            else:
                # Initialize new bar
                self.current_bar[symbol] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': trade.size if hasattr(trade, 'size') else 0,
                    'minute_boundary': minute_boundary
                }

            # Emit current bar update (excluding minute_boundary)
            bar_data = self.current_bar[symbol].copy()
            bar_data.pop('minute_boundary', None)
            await self._emit_to_clients('bar_update', {
                'symbol': symbol,
                'timestamp': trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp),
                **bar_data,
                'asset_type': 'stock'
            })
                
        except Exception as e:
            logger.error(f"Error handling stock trade: {e}")
    
    async def _handle_crypto_trade(self, trade):
        """Handle incoming crypto trade for current bar updates"""
        try:
            # Update current bar with latest trade
            price = float(trade.price)
            
            # Initialize or update current bar
            if not self.current_bar.get(trade.symbol):
                self.current_bar[trade.symbol] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0
                }
                self.bar_start_time = trade.timestamp
            else:
                bar = self.current_bar[trade.symbol]
                bar['high'] = max(bar['high'], price)
                bar['low'] = min(bar['low'], price)
                bar['close'] = price
                bar['volume'] += trade.size
            
            # Emit current bar update
            await self._emit_to_clients('bar_update', {
                'symbol': trade.symbol,
                'timestamp': trade.timestamp.isoformat(),
                **self.current_bar[trade.symbol],
                'asset_type': 'crypto'
            })
                
        except Exception as e:
            logger.error(f"Error handling crypto trade: {e}")
    
    async def _handle_crypto_bar_modern(self, bar):
        """Handle incoming crypto bar data from modern alpaca-py"""
        try:
            logger.debug(f"Modern crypto bar received: {bar}")
            
            # Extract data from modern alpaca-py bar format
            bar_data = {
                'symbol': bar.symbol,
                'timestamp': bar.timestamp.isoformat(),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume),
                'asset_type': 'crypto'
            }
            
            # Emit completed bar to clients
            await self._emit_to_clients('bar_complete', bar_data)
            logger.info(f"Emitted modern crypto bar for {bar.symbol}")
                
        except Exception as e:
            logger.error(f"Error handling modern crypto bar: {e}")
    
    async def _handle_crypto_trade_modern(self, trade):
        """Handle incoming crypto trade from modern alpaca-py"""
        try:
            # Update current bar with latest trade
            price = float(trade.price)
            
            # Initialize or update current bar
            if not self.current_bar.get(trade.symbol):
                self.current_bar[trade.symbol] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0
                }
                self.bar_start_time = trade.timestamp
            else:
                bar = self.current_bar[trade.symbol]
                bar['high'] = max(bar['high'], price)
                bar['low'] = min(bar['low'], price)
                bar['close'] = price
                bar['volume'] += trade.size
            
            # Emit current bar update
            await self._emit_to_clients('bar_update', {
                'symbol': trade.symbol,
                'timestamp': trade.timestamp.isoformat(),
                **self.current_bar[trade.symbol],
                'asset_type': 'crypto'
            })
                
        except Exception as e:
            logger.error(f"Error handling modern crypto trade: {e}")

    def _handle_crypto_quote_modern(self, quote):
        """Handle incoming crypto quote to provide more frequent updates."""
        try:
            # Derive a mid price from bid/ask if available
            bid = float(getattr(quote, 'bid_price', 0) or 0)
            ask = float(getattr(quote, 'ask_price', 0) or 0)
            price = None
            if bid > 0 and ask > 0:
                price = (bid + ask) / 2.0
            elif ask > 0:
                price = ask
            elif bid > 0:
                price = bid
            else:
                return

            symbol = getattr(quote, 'symbol', self.current_symbol)
            ts = getattr(quote, 'timestamp', None) or getattr(quote, 'time', None)
            if not ts:
                return

            # Initialize or update current bar
            if not self.current_bar.get(symbol):
                self.current_bar[symbol] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0
                }
                self.bar_start_time = ts
            else:
                bar = self.current_bar[symbol]
                bar['high'] = max(bar['high'], price)
                bar['low'] = min(bar['low'], price)
                bar['close'] = price

            # Emit as bar_update (schedule onto FastAPI loop via _emit_to_clients)
            import asyncio
            payload = {
                'symbol': symbol,
                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                **self.current_bar[symbol],
                'asset_type': 'crypto'
            }
            if self.fastapi_loop:
                asyncio.run_coroutine_threadsafe(self._emit_to_clients('bar_update', payload), self.fastapi_loop)
        except Exception as e:
            logger.error(f"Error handling modern crypto quote: {e}")
    
    async def unsubscribe_all(self):
        """Unsubscribe from all current subscriptions"""
        if not self.stream or not self.current_symbol:
            return
            
        try:
            logger.info(f"Unsubscribing from {self.current_symbol}")
            
            if self._is_crypto_symbol(self.current_symbol):
                try:
                    self.stream.unsubscribe_crypto_bars(self.current_symbol)
                except Exception as e:
                    logger.warning(f"Error unsubscribing from crypto bars: {e}")
                try:
                    self.stream.unsubscribe_crypto_trades(self.current_symbol)
                except Exception as e:
                    logger.warning(f"Error unsubscribing from crypto trades: {e}")
            else:
                try:
                    self.stream.unsubscribe_bars(self.current_symbol)
                except Exception as e:
                    logger.warning(f"Error unsubscribing from stock bars: {e}")
                try:
                    self.stream.unsubscribe_trades(self.current_symbol)
                except Exception as e:
                    logger.warning(f"Error unsubscribing from stock trades: {e}")
            
            logger.info(f"Unsubscribed from {self.current_symbol}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {self.current_symbol}: {e}")
        finally:
            # Always reset state even if unsubscribe failed
            self.current_symbol = None
            self.is_connected = False
            
            # Notify client
            try:
                await self._emit_to_clients('stream_status', {
                    'connected': False,
                    'symbol': None
                })
            except Exception as e:
                logger.error(f"Error notifying client of disconnection: {e}")
    
    async def start(self):
        """Start the WebSocket stream in a persistent background thread."""
        try:
            import threading

            # Start modern crypto stream (live data) in its own thread
            if self.use_live_data and self.crypto_stream:
                if not hasattr(self, '_crypto_thread') or self._crypto_thread is None or not self._crypto_thread.is_alive():
                    def run_crypto():
                        try:
                            logger.info("Starting modern crypto stream run()")
                            self.crypto_stream.run()
                        except Exception as e:
                            logger.error(f"Crypto stream run() error: {e}")
                    self._crypto_thread = threading.Thread(target=run_crypto, daemon=True)
                    self._crypto_thread.start()
                    logger.info("Modern crypto stream started in background thread")
                self.is_connected = True
                return

            # Start traditional stream for stocks or paper crypto
            if self.stream:
                if not hasattr(self, '_stream_thread') or self._stream_thread is None or not self._stream_thread.is_alive():
                    def run_stream():
                        try:
                            logger.info("Starting traditional stream run()")
                            self.stream.run()
                        except Exception as e:
                            logger.error(f"Traditional stream run() error: {e}")
                    self._stream_thread = threading.Thread(target=run_stream, daemon=True)
                    self._stream_thread.start()
                    logger.info("Traditional stream started in background thread")
                self.is_connected = True
        except Exception as e:
            logger.error(f"Stream start error: {e}")
            self.is_connected = False
    
    def stop(self):
        """Stop the WebSocket stream"""
        try:
            if self.stream:
                # Skip unsubscribe during shutdown to avoid hanging
                # Just force stop the stream immediately
                self.stream.stop()
                logger.info("Alpaca WebSocket stream stopped")
                
            # Reset all state
            self.stream = None
            self.current_symbol = None
            self.is_connected = False
            self.is_connecting = False
            # Stop crypto stream if running
            if hasattr(self, '_crypto_thread') and self._crypto_thread is not None:
                try:
                    if self.crypto_stream:
                        self.crypto_stream.stop()
                except Exception as e:
                    logger.warning(f"Error stopping crypto stream: {e}")
                self._crypto_thread = None
            if hasattr(self, '_stream_thread'):
                self._stream_thread = None
            
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            # Force reset state even if stop failed
            self.stream = None
            self.current_symbol = None
            self.is_connected = False
            self.is_connecting = False
    
    async def _test_crypto_connection(self, alpaca_symbol, timeout=5):
        """Test crypto stream connection with timeout"""
        try:
            logger.info(f"Testing crypto connection for {alpaca_symbol} with {timeout}s timeout")
            
            # Skip test on non-live data feed
            if not self.use_live_data:
                logger.info("Non-live data feed detected - crypto streaming test skipped")
                return False
            
            # Quick connection test - try to create stream instance
            # This is a lightweight test before attempting full connection
            test_stream = CryptoDataStream(
                self.config['ALPACA_KEY_ID'],
                self.config['ALPACA_SECRET_KEY']
            )
            
            # If we can create the stream, assume connection will work
            # (actual connection is tested in the async task)
            logger.info(f"Crypto stream test passed for {alpaca_symbol}")
            return True
            
        except Exception as e:
            logger.warning(f"Crypto connection test failed for {alpaca_symbol}: {e}")
            return False
    
    async def _run_crypto_stream_async(self, alpaca_symbol, original_symbol):
        """Run crypto stream in async background task"""
        try:
            logger.info(f"Starting async crypto stream for {alpaca_symbol}")
            
            # This runs the blocking crypto stream in a separate thread pool
            import concurrent.futures
            import asyncio
            
            def run_blocking_stream():
                try:
                    self.crypto_stream.run()
                    return True
                except Exception as e:
                    logger.error(f"Crypto stream run error: {e}")
                    return False
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Start the stream but don't wait indefinitely
                future = executor.submit(run_blocking_stream)
                
                try:
                    # Wait for stream to start (or timeout after 10 seconds)
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, future.result), 
                        timeout=10.0
                    )
                    logger.info(f"Crypto stream started successfully for {alpaca_symbol}")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Crypto stream startup timeout for {alpaca_symbol}")
                    # Cancel the future and emit error
                    future.cancel()
                    await self._emit_to_clients('stream_error', {
                        'message': 'Crypto stream timeout - using historical data mode',
                        'symbol': original_symbol,
                        'fallback': True
                    })
                
        except Exception as e:
            logger.error(f"Async crypto stream error for {alpaca_symbol}: {e}")
            await self._emit_to_clients('stream_error', {
                'message': f'Crypto stream error: {str(e)}',
                'symbol': original_symbol,
                'fallback': True
            })
    
    async def _run_stream_async(self):
        """Run traditional Alpaca stream in async background task"""
        try:
            logger.info("Starting traditional stream background task...")
            
            # Run the blocking stream in a thread pool
            import concurrent.futures
            import asyncio
            
            def run_blocking_stream():
                try:
                    logger.info("Traditional stream: calling _run_forever()")
                    return asyncio.run(self.stream._run_forever())
                except Exception as e:
                    logger.error(f"Traditional stream run error: {e}")
                    return False
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Start the stream but don't wait indefinitely
                future = executor.submit(run_blocking_stream)
                
                try:
                    # Just start it and let it run - don't wait for completion
                    logger.info("Traditional stream task submitted to thread pool")
                    # Don't await - let it run in background
                    
                except Exception as e:
                    logger.warning(f"Traditional stream startup error: {e}")
                    # Don't fail completely - continue with prediction-only mode
                    await self._emit_to_clients('stream_error', {
                        'message': 'Real-time streaming unavailable - using prediction mode',
                        'fallback': True
                    })
                
        except Exception as e:
            logger.error(f"Async stream error: {e}")
            await self._emit_to_clients('stream_error', {
                'message': f'Stream error: {str(e)}',
                'fallback': True
            })
