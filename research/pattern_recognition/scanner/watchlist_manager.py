"""
Watchlist Manager for Scanner Module.

Manages default (static) and floating (dynamic) watchlists in TimescaleDB.
Provides CRUD operations and combined watchlist retrieval.
"""

import logging
from datetime import date, datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError

from .config import SCANNER_CONFIG, get_db_connection_string
from .db_models import (
    DefaultWatchlist,
    FloatingWatchlist,
    get_session,
    get_engine,
    init_db
)

logger = logging.getLogger(__name__)


class WatchlistManager:
    """
    Manages default + floating watchlists in TimescaleDB.

    Default watchlist: Static symbols that are always scanned (QQQ, NVDA, etc.)
    Floating watchlist: Dynamic premarket movers added daily

    Usage:
        manager = WatchlistManager()

        # Get all symbols to scan today
        symbols = manager.get_combined_watchlist()

        # Add premarket movers
        manager.add_to_floating([
            {'symbol': 'COIN', 'gap_percent': 8.5, 'volume_ratio': 3.2, ...},
            {'symbol': 'ARM', 'gap_percent': 6.2, 'volume_ratio': 2.5, ...},
        ])
    """

    def __init__(self, connection_string: str = None):
        """
        Initialize WatchlistManager.

        Args:
            connection_string: PostgreSQL connection string.
                              If None, uses config defaults.
        """
        self.connection_string = connection_string or get_db_connection_string()
        self._session = None

    def _get_session(self):
        """Get or create database session."""
        if self._session is None:
            self._session = get_session()
        return self._session

    def close(self):
        """Close database session."""
        if self._session:
            self._session.close()
            self._session = None

    # =========================================================================
    # DEFAULT WATCHLIST OPERATIONS
    # =========================================================================

    def get_default_watchlist(self, enabled_only: bool = True) -> List[str]:
        """
        Get symbols from default (static) watchlist.

        Args:
            enabled_only: If True, only return enabled symbols.

        Returns:
            List of ticker symbols
        """
        session = self._get_session()
        try:
            query = session.query(DefaultWatchlist.symbol)
            if enabled_only:
                query = query.filter(DefaultWatchlist.enabled == True)
            symbols = [row.symbol for row in query.all()]
            logger.debug(f"Default watchlist: {len(symbols)} symbols")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching default watchlist: {e}")
            # Fallback to config defaults
            return SCANNER_CONFIG['default_symbols']

    def add_to_default(
        self,
        symbol: str,
        category: str = None,
        notes: str = None,
        enabled: bool = True
    ) -> bool:
        """
        Add symbol to default watchlist.

        Args:
            symbol: Ticker symbol (uppercase)
            category: Category (e.g., 'index_etf', 'tech_mega', 'momentum')
            notes: Optional notes
            enabled: Whether to enable for scanning

        Returns:
            True if added successfully, False if already exists
        """
        session = self._get_session()
        try:
            entry = DefaultWatchlist(
                symbol=symbol.upper(),
                category=category,
                notes=notes,
                enabled=enabled
            )
            session.add(entry)
            session.commit()
            logger.info(f"Added {symbol} to default watchlist")
            return True
        except IntegrityError:
            session.rollback()
            logger.warning(f"Symbol {symbol} already in default watchlist")
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding to default watchlist: {e}")
            return False

    def remove_from_default(self, symbol: str) -> bool:
        """
        Remove symbol from default watchlist.

        Args:
            symbol: Ticker symbol

        Returns:
            True if removed, False if not found
        """
        session = self._get_session()
        try:
            result = session.query(DefaultWatchlist).filter(
                DefaultWatchlist.symbol == symbol.upper()
            ).delete()
            session.commit()
            if result > 0:
                logger.info(f"Removed {symbol} from default watchlist")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing from default watchlist: {e}")
            return False

    def toggle_default_symbol(self, symbol: str, enabled: bool) -> bool:
        """
        Enable or disable a symbol in the default watchlist.

        Args:
            symbol: Ticker symbol
            enabled: True to enable, False to disable

        Returns:
            True if updated, False if not found
        """
        session = self._get_session()
        try:
            entry = session.query(DefaultWatchlist).filter(
                DefaultWatchlist.symbol == symbol.upper()
            ).first()
            if entry:
                entry.enabled = enabled
                entry.updated_at = datetime.utcnow()
                session.commit()
                logger.info(f"Set {symbol} enabled={enabled}")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error toggling default symbol: {e}")
            return False

    def init_default_watchlist(self) -> int:
        """
        Initialize default watchlist from config.

        Adds all symbols from SCANNER_CONFIG['default_symbols'] if not exists.

        Returns:
            Number of symbols added
        """
        count = 0
        for symbol in SCANNER_CONFIG['default_symbols']:
            category = SCANNER_CONFIG['categories'].get(symbol, 'other')
            if self.add_to_default(symbol, category=category):
                count += 1
        logger.info(f"Initialized default watchlist with {count} new symbols")
        return count

    # =========================================================================
    # FLOATING WATCHLIST OPERATIONS
    # =========================================================================

    def get_floating_watchlist(self, target_date: date = None) -> List[str]:
        """
        Get symbols from floating (dynamic) watchlist for a date.

        Args:
            target_date: Date to fetch. Defaults to today.

        Returns:
            List of ticker symbols
        """
        if target_date is None:
            target_date = date.today()

        session = self._get_session()
        try:
            symbols = [
                row.symbol for row in
                session.query(FloatingWatchlist.symbol)
                .filter(FloatingWatchlist.date == target_date)
                .all()
            ]
            logger.debug(f"Floating watchlist for {target_date}: {len(symbols)} symbols")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching floating watchlist: {e}")
            return []

    def get_floating_watchlist_details(
        self,
        target_date: date = None
    ) -> List[Dict[str, Any]]:
        """
        Get detailed floating watchlist with metrics.

        Args:
            target_date: Date to fetch. Defaults to today.

        Returns:
            List of dicts with symbol, gap_percent, volume_ratio, etc.
        """
        if target_date is None:
            target_date = date.today()

        session = self._get_session()
        try:
            entries = session.query(FloatingWatchlist).filter(
                FloatingWatchlist.date == target_date
            ).order_by(FloatingWatchlist.gap_percent.desc()).all()

            return [entry.to_dict() for entry in entries]
        except Exception as e:
            logger.error(f"Error fetching floating watchlist details: {e}")
            return []

    def add_to_floating(
        self,
        movers: List[Dict[str, Any]],
        target_date: date = None
    ) -> int:
        """
        Add premarket movers to floating watchlist.

        Args:
            movers: List of dicts with keys:
                - symbol (required)
                - gap_percent
                - premarket_volume
                - avg_volume
                - volume_ratio
                - market_cap
                - prev_close
                - premarket_high
                - premarket_low
                - premarket_open
                - reason
            target_date: Date for entries. Defaults to today.

        Returns:
            Number of symbols added
        """
        if target_date is None:
            target_date = date.today()

        session = self._get_session()
        count = 0

        for mover in movers:
            try:
                entry = FloatingWatchlist(
                    date=target_date,
                    symbol=mover['symbol'].upper(),
                    gap_percent=mover.get('gap_percent'),
                    premarket_volume=mover.get('premarket_volume'),
                    avg_volume=mover.get('avg_volume'),
                    volume_ratio=mover.get('volume_ratio'),
                    market_cap=mover.get('market_cap'),
                    prev_close=mover.get('prev_close'),
                    premarket_high=mover.get('premarket_high'),
                    premarket_low=mover.get('premarket_low'),
                    premarket_open=mover.get('premarket_open'),
                    reason=mover.get('reason'),
                )
                session.add(entry)
                session.commit()
                count += 1
                logger.debug(f"Added {mover['symbol']} to floating watchlist")
            except IntegrityError:
                session.rollback()
                # Already exists, update instead
                try:
                    existing = session.query(FloatingWatchlist).filter(
                        and_(
                            FloatingWatchlist.date == target_date,
                            FloatingWatchlist.symbol == mover['symbol'].upper()
                        )
                    ).first()
                    if existing:
                        for key, value in mover.items():
                            if key != 'symbol' and hasattr(existing, key):
                                setattr(existing, key, value)
                        existing.scanned_at = datetime.utcnow()
                        session.commit()
                        count += 1
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error updating floating entry: {e}")
            except Exception as e:
                session.rollback()
                logger.error(f"Error adding to floating watchlist: {e}")

        logger.info(f"Added/updated {count} symbols to floating watchlist")
        return count

    def clear_floating(self, target_date: date = None) -> int:
        """
        Clear floating watchlist for a date.

        Useful for re-scanning premarket.

        Args:
            target_date: Date to clear. Defaults to today.

        Returns:
            Number of entries deleted
        """
        if target_date is None:
            target_date = date.today()

        session = self._get_session()
        try:
            count = session.query(FloatingWatchlist).filter(
                FloatingWatchlist.date == target_date
            ).delete()
            session.commit()
            logger.info(f"Cleared {count} entries from floating watchlist for {target_date}")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing floating watchlist: {e}")
            return 0

    def cleanup_old_floating(self, keep_days: int = 30) -> int:
        """
        Remove floating watchlist entries older than keep_days.

        Args:
            keep_days: Number of days to keep

        Returns:
            Number of entries deleted
        """
        session = self._get_session()
        cutoff = date.today().replace(day=date.today().day - keep_days)
        try:
            count = session.query(FloatingWatchlist).filter(
                FloatingWatchlist.date < cutoff
            ).delete()
            session.commit()
            logger.info(f"Cleaned up {count} old floating watchlist entries")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up floating watchlist: {e}")
            return 0

    # =========================================================================
    # COMBINED WATCHLIST
    # =========================================================================

    def get_combined_watchlist(self, target_date: date = None) -> List[str]:
        """
        Get combined watchlist (default + floating, deduplicated).

        Args:
            target_date: Date for floating watchlist. Defaults to today.

        Returns:
            List of unique ticker symbols
        """
        default = set(self.get_default_watchlist(enabled_only=True))
        floating = set(self.get_floating_watchlist(target_date))

        combined = list(default | floating)
        combined.sort()

        logger.info(
            f"Combined watchlist: {len(combined)} symbols "
            f"({len(default)} default + {len(floating)} floating)"
        )
        return combined

    def get_combined_watchlist_details(
        self,
        target_date: date = None
    ) -> List[Dict[str, Any]]:
        """
        Get combined watchlist with source information.

        Args:
            target_date: Date for floating watchlist

        Returns:
            List of dicts with symbol, source ('default' or 'floating'), and details
        """
        result = []

        # Add default symbols
        session = self._get_session()
        try:
            default_entries = session.query(DefaultWatchlist).filter(
                DefaultWatchlist.enabled == True
            ).all()

            for entry in default_entries:
                result.append({
                    'symbol': entry.symbol,
                    'source': 'default',
                    'category': entry.category,
                    'gap_percent': None,
                    'volume_ratio': None,
                })
        except Exception as e:
            logger.error(f"Error fetching default entries: {e}")

        # Add floating symbols (that aren't already in default)
        default_symbols = {r['symbol'] for r in result}
        floating_details = self.get_floating_watchlist_details(target_date)

        for entry in floating_details:
            if entry['symbol'] not in default_symbols:
                result.append({
                    'symbol': entry['symbol'],
                    'source': 'floating',
                    'category': None,
                    'gap_percent': entry.get('gap_percent'),
                    'volume_ratio': entry.get('volume_ratio'),
                })

        return sorted(result, key=lambda x: x['symbol'])


# Convenience functions
def get_watchlist_manager() -> WatchlistManager:
    """Get a WatchlistManager instance."""
    return WatchlistManager()
