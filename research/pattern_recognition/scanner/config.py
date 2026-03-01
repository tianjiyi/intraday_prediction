"""
Scanner Module Configuration.

Centralized configuration for premarket scanning and real-time pattern detection.
"""

import os
from typing import Dict, Any


# Scanner Configuration
SCANNER_CONFIG: Dict[str, Any] = {
    # ==========================================================================
    # PREMARKET SCANNER SETTINGS
    # Runs from 4:00am - 9:30am ET to find unusual movers
    # ==========================================================================
    'premarket': {
        'start_time': '04:00',          # 4:00 AM ET
        'end_time': '09:30',            # 9:30 AM ET (market open)
        'min_market_cap': 1_000_000_000,  # $1 billion minimum
        'min_gap_percent': 5.0,         # 5% gap up/down from prev close
        'min_volume_ratio': 0.10,       # 10% of daily avg = unusual premarket volume
        'max_symbols_to_scan': 500,     # Limit API calls (0 = no limit)
        'batch_size': 100,              # Symbols per API batch
        'batch_delay_seconds': 0.1,       # Delay between batches
        'scan_interval_minutes': 15,    # Re-scan every 15 minutes
        'data_feed': 'sip',             # 'sip' for full market, 'iex' for free tier
    },

    # ==========================================================================
    # REAL-TIME SCANNER SETTINGS
    # Runs during RTH (9:30am - 4:00pm ET) for pattern detection
    # ==========================================================================
    'realtime': {
        'start_time': '09:30',          # RTH start
        'end_time': '16:00',            # RTH end
        'scan_interval_seconds': 60,    # Scan every minute
        'timeframe': '1min',            # Bar timeframe for pattern detection
        'lookback_bars': 200,           # Bars for pattern analysis
        'min_confidence': 0.5,          # Minimum pattern confidence to log
        'dedup_time_minutes': 30,       # Don't log same pattern within 30 mins
        'dedup_price_tolerance': 0.01,  # 1% price tolerance for deduplication
    },

    # ==========================================================================
    # DEFAULT WATCHLIST
    # Static list of symbols to always scan
    # ==========================================================================
    'default_symbols': [
        # Index ETFs
        'QQQ',   # Nasdaq 100
        'SPY',   # S&P 500

        # Tech Mega Caps
        'NVDA',  # NVIDIA
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'META',  # Meta/Facebook
        'GOOGL', # Alphabet
        'AMZN',  # Amazon

        # Momentum / High Beta
        'TSLA',  # Tesla
        'AMD',   # AMD
    ],

    # ==========================================================================
    # CATEGORY MAPPING
    # For database categorization
    # ==========================================================================
    'categories': {
        'QQQ': 'index_etf',
        'SPY': 'index_etf',
        'IWM': 'index_etf',
        'NVDA': 'tech_mega',
        'AAPL': 'tech_mega',
        'MSFT': 'tech_mega',
        'META': 'tech_mega',
        'GOOGL': 'tech_mega',
        'AMZN': 'tech_mega',
        'TSLA': 'momentum',
        'AMD': 'momentum',
    },

    # ==========================================================================
    # DATABASE CONFIGURATION
    # TimescaleDB connection settings
    # ==========================================================================
    'database': {
        'host': os.getenv('TIMESCALE_HOST', 'localhost'),
        'port': int(os.getenv('TIMESCALE_PORT', '5432')),
        'database': os.getenv('TIMESCALE_DB', 'market_data'),
        'user': os.getenv('TIMESCALE_USER', 'kronos'),
        'password': os.getenv('TIMESCALE_PASSWORD', 'kronos_dev'),
    },

    # ==========================================================================
    # ALPACA API CONFIGURATION
    # Data provider for market data
    # ==========================================================================
    'alpaca': {
        'api_key': os.getenv('ALPACA_KEY_ID', ''),
        'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
        'base_url': os.getenv('ALPACA_BASE_URL', 'https://api.alpaca.markets'),
        'data_url': os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets'),
        'paper': os.getenv('ALPACA_PAPER', 'false').lower() == 'true',
    },

    # ==========================================================================
    # POLYGON.IO API CONFIGURATION
    # Fundamentals data provider (market cap, shares outstanding)
    # ==========================================================================
    'polygon': {
        'api_key': os.getenv('POLYGON_API_KEY', ''),
        'base_url': 'https://api.polygon.io',
    },

    # ==========================================================================
    # PATTERN DETECTION SETTINGS
    # Ascending triangle detector configuration
    # ==========================================================================
    'pattern_detection': {
        'resistance_tolerance': None,   # Auto-select based on timeframe
        'min_support_slope': 0.00001,
        'zigzag_deviation': None,       # Auto-select based on timeframe
        'min_bars': None,               # Auto-select based on timeframe
        'pre_context_bars': 20,
        'post_context_bars': 20,
        'analyze_breakout': True,
    },

    # ==========================================================================
    # LOGGING
    # ==========================================================================
    'logging': {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
    },
}


def get_db_connection_string() -> str:
    """
    Build PostgreSQL connection string from config.

    Returns:
        Connection string for SQLAlchemy
    """
    db = SCANNER_CONFIG['database']
    return (
        f"postgresql://{db['user']}:{db['password']}"
        f"@{db['host']}:{db['port']}/{db['database']}"
    )


def load_config_from_yaml(config_path: str = None) -> Dict[str, Any]:
    """
    Load additional configuration from YAML file.

    Merges with default SCANNER_CONFIG, with YAML values taking precedence.

    Args:
        config_path: Path to config.yaml. If None, searches parent directories.

    Returns:
        Merged configuration dictionary
    """
    import yaml
    from pathlib import Path

    # Find config.yaml
    if config_path is None:
        # Search in parent directories
        current = Path(__file__).parent
        for _ in range(4):  # Search up to 4 levels
            config_file = current / 'config.yaml'
            if config_file.exists():
                config_path = str(config_file)
                break
            current = current.parent

    if config_path is None or not Path(config_path).exists():
        return SCANNER_CONFIG

    # Load YAML
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f) or {}

    # Merge with defaults (deep copy to avoid modifying original)
    import copy
    merged = copy.deepcopy(SCANNER_CONFIG)

    # Override Alpaca keys if present
    if 'ALPACA_KEY_ID' in yaml_config:
        merged['alpaca']['api_key'] = yaml_config['ALPACA_KEY_ID']
    if 'ALPACA_SECRET_KEY' in yaml_config:
        merged['alpaca']['secret_key'] = yaml_config['ALPACA_SECRET_KEY']

    # Override Polygon key if present (top-level or nested)
    if 'POLYGON_API_KEY' in yaml_config:
        merged['polygon']['api_key'] = yaml_config['POLYGON_API_KEY']
    elif 'polygon' in yaml_config and 'api_key' in yaml_config['polygon']:
        merged['polygon']['api_key'] = yaml_config['polygon']['api_key']

    # Override database settings if present
    if 'database' in yaml_config:
        merged['database'].update(yaml_config['database'])

    # Override scanner-specific settings if present
    if 'scanner' in yaml_config:
        if 'premarket' in yaml_config['scanner']:
            merged['premarket'].update(yaml_config['scanner']['premarket'])
        if 'realtime' in yaml_config['scanner']:
            merged['realtime'].update(yaml_config['scanner']['realtime'])
        if 'default_symbols' in yaml_config['scanner']:
            merged['default_symbols'] = yaml_config['scanner']['default_symbols']

    return merged


def get_alpaca_client():
    """
    Get Alpaca API client for data fetching.

    Returns:
        Tuple of (StockHistoricalDataClient, CryptoHistoricalDataClient)
    """
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient

    config = SCANNER_CONFIG['alpaca']
    api_key = config['api_key']
    secret_key = config['secret_key']

    stock_client = StockHistoricalDataClient(api_key, secret_key)
    crypto_client = CryptoHistoricalDataClient(api_key, secret_key)

    return stock_client, crypto_client


def get_trading_client():
    """
    Get Alpaca Trading API client for account/asset info.

    Returns:
        TradingClient instance
    """
    from alpaca.trading.client import TradingClient

    config = SCANNER_CONFIG['alpaca']
    return TradingClient(
        api_key=config['api_key'],
        secret_key=config['secret_key'],
        paper=config['paper']
    )
