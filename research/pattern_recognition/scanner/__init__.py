"""
Scanner Module - Premarket Scanner + Real-time Pattern Detection

This module provides:
1. Premarket Scanner: Finds unusual movers based on gap%, volume, market cap
2. Watchlist Manager: Manages default + floating watchlists in TimescaleDB
3. Real-time Scanner: Scans for ascending triangle patterns every minute during RTH

Usage:
    # Initialize database
    python -m pattern_recognition.scanner.run_scanner --init-db

    # Run premarket scanner
    python -m pattern_recognition.scanner.run_scanner --mode premarket

    # Run real-time pattern scanner
    python -m pattern_recognition.scanner.run_scanner --mode realtime

    # Run both (full day automation)
    python -m pattern_recognition.scanner.run_scanner --mode both

    # Show current watchlist
    python -m pattern_recognition.scanner.run_scanner --show-watchlist
"""

from .config import (
    SCANNER_CONFIG,
    get_db_connection_string,
    load_config_from_yaml,
    get_alpaca_client,
    get_trading_client,
)
from .watchlist_manager import WatchlistManager, get_watchlist_manager
from .premarket_scanner import PremarketScanner, PremarketMover
from .realtime_scanner import RealtimePatternScanner
from .db_models import (
    Base,
    DefaultWatchlist,
    FloatingWatchlist,
    PatternFinding,
    PremarketScanLog,
    RealtimeScanLog,
    get_engine,
    get_session,
    init_db,
)

__all__ = [
    # Configuration
    'SCANNER_CONFIG',
    'get_db_connection_string',
    'load_config_from_yaml',
    'get_alpaca_client',
    'get_trading_client',

    # Managers & Scanners
    'WatchlistManager',
    'get_watchlist_manager',
    'PremarketScanner',
    'RealtimePatternScanner',

    # Data classes
    'PremarketMover',

    # Database
    'Base',
    'get_engine',
    'get_session',
    'init_db',

    # ORM Models
    'DefaultWatchlist',
    'FloatingWatchlist',
    'PatternFinding',
    'PremarketScanLog',
    'RealtimeScanLog',
]

__version__ = '1.0.0'
