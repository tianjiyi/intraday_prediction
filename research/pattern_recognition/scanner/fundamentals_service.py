"""
Stock Fundamentals Service.

Fetches and caches fundamental data (market cap, float, sector, etc.)
from Yahoo Finance. Refreshed weekly/monthly.

Usage:
    from .fundamentals_service import FundamentalsService

    service = FundamentalsService()

    # Get market cap for a symbol
    mcap = service.get_market_cap('AAPL')

    # Refresh fundamentals for symbols
    service.refresh(['AAPL', 'NVDA', 'GSIT'])

    # Refresh all tradable stocks
    service.refresh_all()
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
import requests
import yfinance as yf

from .db_models import StockFundamentals, get_session
from .config import load_config_from_yaml

logger = logging.getLogger(__name__)


class FundamentalsService:
    """
    Service for fetching and caching stock fundamentals from Yahoo Finance.
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize FundamentalsService.

        Args:
            max_workers: Max threads for parallel fetching
        """
        self.max_workers = max_workers
        self._cache: Dict[str, float] = {}  # In-memory cache for market cap

    def get_market_cap(self, symbol: str) -> float:
        """
        Get market cap for a symbol from database.

        Args:
            symbol: Ticker symbol

        Returns:
            Market cap in dollars, or 0 if not found
        """
        # Check in-memory cache first
        if symbol in self._cache:
            return self._cache[symbol]

        session = get_session()
        try:
            row = session.query(StockFundamentals).filter_by(symbol=symbol).first()
            if row and row.market_cap:
                mcap = float(row.market_cap)
                self._cache[symbol] = mcap
                return mcap
            return 0
        except Exception as e:
            logger.error(f"Error getting market cap for {symbol}: {e}")
            return 0
        finally:
            session.close()

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get all fundamentals for a symbol from database.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with fundamentals or None
        """
        session = get_session()
        try:
            row = session.query(StockFundamentals).filter_by(symbol=symbol).first()
            return row.to_dict() if row else None
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return None
        finally:
            session.close()

    def fetch_from_yahoo(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch fundamentals from Yahoo Finance.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with fundamentals or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Handle case where info is empty or has error
            if not info or info.get('regularMarketPrice') is None:
                logger.warning(f"No Yahoo data for {symbol}")
                return None

            return {
                'symbol': symbol,
                'market_cap': info.get('marketCap'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
            }
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
            return None

    def fetch_from_polygon(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch fundamentals from Polygon.io API.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with fundamentals or None if error
        """
        try:
            config = load_config_from_yaml()
            api_key = config.get('polygon', {}).get('api_key', '')

            if not api_key:
                logger.error("Polygon API key not configured. Add POLYGON_API_KEY to config.yaml")
                return None

            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {'apiKey': api_key}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 404:
                logger.warning(f"Symbol {symbol} not found in Polygon")
                return None

            if response.status_code != 200:
                logger.warning(f"Polygon API error for {symbol}: {response.status_code}")
                return None

            data = response.json().get('results', {})

            if not data:
                logger.warning(f"No Polygon data for {symbol}")
                return None

            return {
                'symbol': symbol,
                'market_cap': data.get('market_cap'),
                'shares_outstanding': data.get('share_class_shares_outstanding') or data.get('weighted_shares_outstanding'),
                'float_shares': None,  # Polygon doesn't provide float
                'sector': None,        # Polygon has different sector format
                'industry': None,
            }

        except requests.exceptions.Timeout:
            logger.error(f"Polygon API timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching Polygon data for {symbol}: {e}")
            return None

    def upsert(self, data: Dict[str, Any]) -> bool:
        """
        Insert or update fundamentals in database.

        Args:
            data: Dict with symbol, market_cap, etc.

        Returns:
            True if successful
        """
        session = get_session()
        try:
            existing = session.query(StockFundamentals).filter_by(
                symbol=data['symbol']
            ).first()

            if existing:
                # Update existing
                if data.get('market_cap'):
                    existing.market_cap = Decimal(str(data['market_cap']))
                if data.get('shares_outstanding'):
                    existing.shares_outstanding = data['shares_outstanding']
                if data.get('float_shares'):
                    existing.float_shares = data['float_shares']
                if data.get('sector'):
                    existing.sector = data['sector']
                if data.get('industry'):
                    existing.industry = data['industry']
                existing.updated_at = datetime.utcnow()
            else:
                # Insert new
                row = StockFundamentals(
                    symbol=data['symbol'],
                    market_cap=Decimal(str(data['market_cap'])) if data.get('market_cap') else None,
                    shares_outstanding=data.get('shares_outstanding'),
                    float_shares=data.get('float_shares'),
                    sector=data.get('sector'),
                    industry=data.get('industry'),
                    updated_at=datetime.utcnow()
                )
                session.add(row)

            session.commit()

            # Update in-memory cache
            if data.get('market_cap'):
                self._cache[data['symbol']] = float(data['market_cap'])

            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error upserting fundamentals for {data.get('symbol')}: {e}")
            return False
        finally:
            session.close()

    def refresh(self, symbols: List[str], source: str = 'polygon', show_progress: bool = True, delay: float = None) -> Dict[str, int]:
        """
        Refresh fundamentals for a list of symbols.

        Args:
            symbols: List of ticker symbols
            source: Data source - 'polygon' (default, faster) or 'yahoo' (has sector/industry)
            show_progress: Print progress updates
            delay: Delay between requests in seconds. Defaults: polygon=0.05s, yahoo=0.2s

        Returns:
            Dict with 'success' and 'failed' counts
        """
        import time

        # Set default delay based on source
        if delay is None:
            delay = 0.05 if source == 'polygon' else 0.2

        results = {'success': 0, 'failed': 0}

        if show_progress:
            print(f"\n  Refreshing fundamentals for {len(symbols)} symbols...")
            print(f"  Source: {source.upper()} (delay={delay}s)")

        for i, symbol in enumerate(symbols, 1):
            try:
                if source == 'polygon':
                    data = self.fetch_from_polygon(symbol)
                else:
                    data = self.fetch_from_yahoo(symbol)

                if data and self.upsert(data):
                    results['success'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results['failed'] += 1

            if show_progress and i % 100 == 0:
                print(f"    Processed {i}/{len(symbols)} symbols...")

            # Rate limiting delay
            time.sleep(delay)

        if show_progress:
            print(f"  Done: {results['success']} updated, {results['failed']} failed")

        return results

    def refresh_all(self, source: str = 'polygon', trading_client=None) -> Dict[str, int]:
        """
        Refresh fundamentals for all tradable stocks.

        Args:
            source: Data source - 'polygon' (default) or 'yahoo'
            trading_client: Alpaca TradingClient for fetching symbols

        Returns:
            Dict with 'success' and 'failed' counts
        """
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus

        if trading_client is None:
            config = load_config_from_yaml()
            api_key = config.get('alpaca', {}).get('api_key', '')
            secret_key = config.get('alpaca', {}).get('secret_key', '')

            if not api_key or not secret_key:
                raise ValueError("Alpaca API keys not configured. Check config.yaml")

            # Use live API (paper=False) for fetching asset list
            trading_client = TradingClient(api_key, secret_key, paper=False)

        # Get all tradable stocks
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE
        )
        assets = trading_client.get_all_assets(request)
        symbols = [asset.symbol for asset in assets if asset.tradable]

        print(f"\n  Found {len(symbols)} tradable stocks")

        return self.refresh(symbols, source=source)

    def load_all_to_cache(self) -> int:
        """
        Load all fundamentals from database to in-memory cache.

        Returns:
            Number of symbols loaded
        """
        session = get_session()
        try:
            rows = session.query(StockFundamentals).all()
            for row in rows:
                if row.market_cap:
                    self._cache[row.symbol] = float(row.market_cap)
            return len(self._cache)
        except Exception as e:
            logger.error(f"Error loading fundamentals cache: {e}")
            return 0
        finally:
            session.close()

    def clear_cache(self):
        """Clear in-memory cache."""
        self._cache.clear()


# Module-level singleton for convenience
_service = None


def get_fundamentals_service() -> FundamentalsService:
    """Get or create FundamentalsService singleton."""
    global _service
    if _service is None:
        _service = FundamentalsService()
    return _service


def get_market_cap(symbol: str) -> float:
    """
    Convenience function to get market cap.

    Args:
        symbol: Ticker symbol

    Returns:
        Market cap in dollars, or 0 if not found
    """
    return get_fundamentals_service().get_market_cap(symbol)
