"""
HMM Market Regime Backtester

Backtests the HMM regime detection strategy:
1. Train model on historical data (walk-forward or fixed window)
2. Generate regime predictions
3. Simulate trading based on regime signals
4. Calculate performance metrics

Usage:
    python -m market_regime.backtest --symbol QQQ --days 60 --train-days 30
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime import (
    MarketRegimeService,
    RegimeDetector,
    PolygonDataFetcher,
    calculate_ofi,
    calculate_vpin_for_volume_bars,
    prepare_hmm_features
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)


class RegimeBacktester:
    """
    Backtester for HMM regime detection strategy.

    Strategies:
    - Long only in Bull Trend
    - Short only in Bear Trend
    - Flat in Range
    - Exit/reduce in Stress/Reversal
    """

    def __init__(
        self,
        polygon_api_key: str,
        initial_capital: float = 100000.0,
        position_size: float = 1.0,
        transaction_cost: float = 0.001
    ):
        self.api_key = polygon_api_key
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost

        self.service = MarketRegimeService(polygon_api_key=polygon_api_key)
        self.results: Optional[pd.DataFrame] = None

    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical data for backtesting."""
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

        return self.service.fetch_and_process(
            symbol, start_date, end_date,
            force_refresh=True,
            use_agg_bars=True
        )

    def walk_forward_backtest(
        self,
        symbol: str,
        total_days: int = 60,
        train_window: int = 30,
        test_window: int = 5
    ) -> pd.DataFrame:
        """
        Walk-forward backtesting with rolling training window.

        Args:
            symbol: Stock ticker
            total_days: Total days to backtest
            train_window: Days for training window
            test_window: Days for out-of-sample testing

        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Starting walk-forward backtest: {total_days} days, "
                    f"train={train_window}, test={test_window}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(total_days * 1.5))

        # Fetch all data
        all_data = self.fetch_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if all_data.empty:
            raise ValueError("No data available for backtesting")

        logger.info(f"Total bars: {len(all_data)}")

        # Calculate features for all data
        features_df = self.service._features_cache

        results = []
        position = 0
        capital = self.initial_capital
        entry_price = 0

        # Walk forward through data
        train_size = int(len(all_data) * (train_window / total_days))
        test_size = int(len(all_data) * (test_window / total_days))

        i = train_size
        while i < len(all_data):
            # Train on window
            train_start = max(0, i - train_size)
            train_features = features_df.iloc[train_start:i]

            if len(train_features) < 100:
                i += test_size
                continue

            # Train detector
            detector = RegimeDetector(n_states=4, random_state=42)
            try:
                detector.fit(train_features)
            except Exception as e:
                logger.warning(f"Training failed at bar {i}: {e}")
                i += test_size
                continue

            # Test on next window
            test_end = min(i + test_size, len(all_data))

            for j in range(i, test_end):
                current_features = features_df.iloc[j].values
                bar = all_data.iloc[j]

                # Get regime prediction
                try:
                    pred = detector.predict_live(current_features)
                    regime = pred['state']
                    confidence = pred['confidence']
                except Exception:
                    regime = 'Range'
                    confidence = 0.5

                price = bar['close']
                timestamp = bar['time'] if 'time' in bar else bar['timestamp']

                # Trading logic
                prev_position = position

                if regime == 'Bull Trend' and confidence > 0.6:
                    target_position = 1
                elif regime == 'Bear Trend' and confidence > 0.6:
                    target_position = -1
                elif regime == 'Stress/Reversal':
                    target_position = 0
                else:
                    target_position = position

                # Execute trade if position changes
                if target_position != position:
                    if position != 0:
                        pnl = position * (price - entry_price) * self.position_size * (capital / entry_price)
                        pnl -= abs(pnl) * self.transaction_cost
                        capital += pnl

                    position = target_position
                    if position != 0:
                        entry_price = price
                        capital -= capital * self.transaction_cost * self.position_size

                # Calculate unrealized P&L
                if position != 0:
                    unrealized_pnl = position * (price - entry_price) * self.position_size * (capital / entry_price)
                else:
                    unrealized_pnl = 0

                results.append({
                    'timestamp': timestamp,
                    'price': price,
                    'regime': regime,
                    'confidence': confidence,
                    'position': position,
                    'capital': capital,
                    'unrealized_pnl': unrealized_pnl,
                    'total_equity': capital + unrealized_pnl,
                    'ofi': bar['ofi'],
                    'vpin': bar['vpin']
                })

            i += test_size

        self.results = pd.DataFrame(results)
        return self.results

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics from backtest results."""
        if self.results is None or self.results.empty:
            return {}

        df = self.results

        df['returns'] = df['total_equity'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1

        total_return = (df['total_equity'].iloc[-1] / self.initial_capital) - 1
        trading_days = len(df) / 390
        annualized_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

        daily_returns = df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(390 * 252)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        cummax = df['total_equity'].cummax()
        drawdown = (df['total_equity'] - cummax) / cummax
        max_drawdown = drawdown.min()

        trades = df[df['position'].diff() != 0]
        if len(trades) > 1:
            trade_pnls = []
            for i in range(1, len(trades)):
                if trades.iloc[i-1]['position'] != 0:
                    pnl = trades.iloc[i]['capital'] - trades.iloc[i-1]['capital']
                    trade_pnls.append(pnl)

            if trade_pnls:
                win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
                avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
                avg_loss = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0
            else:
                win_rate = avg_win = avg_loss = 0
            num_trades = len(trade_pnls)
        else:
            win_rate = avg_win = avg_loss = num_trades = 0

        regime_counts = df['regime'].value_counts()

        return {
            'total_return': f"{total_return:.2%}",
            'annualized_return': f"{annualized_return:.2%}",
            'volatility': f"{volatility:.2%}",
            'sharpe_ratio': f"{sharpe:.2f}",
            'max_drawdown': f"{max_drawdown:.2%}",
            'num_trades': num_trades,
            'win_rate': f"{win_rate:.2%}",
            'avg_win': f"${avg_win:.2f}",
            'avg_loss': f"${avg_loss:.2f}",
            'final_capital': f"${df['total_equity'].iloc[-1]:,.2f}",
            'regime_distribution': regime_counts.to_dict(),
            'trading_days': f"{trading_days:.1f}"
        }

    def save_results(self, filepath: str):
        """Save backtest results to CSV."""
        if self.results is not None:
            self.results.to_csv(filepath, index=False)
            logger.info(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='HMM Regime Backtest')
    parser.add_argument('--symbol', default='QQQ', help='Stock ticker')
    parser.add_argument('--days', type=int, default=60, help='Total backtest days')
    parser.add_argument('--train-days', type=int, default=30, help='Training window days')
    parser.add_argument('--test-days', type=int, default=5, help='Test window days')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--output', default='backtest_results.csv', help='Output CSV file')

    args = parser.parse_args()

    # Get API key
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')

    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key and os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            api_key = config.get('polygon', {}).get('api_key')

    if not api_key:
        print("Error: POLYGON_API_KEY not found")
        sys.exit(1)

    # Run backtest
    backtester = RegimeBacktester(
        polygon_api_key=api_key,
        initial_capital=args.capital
    )

    results = backtester.walk_forward_backtest(
        symbol=args.symbol,
        total_days=args.days,
        train_window=args.train_days,
        test_window=args.test_days
    )

    # Print metrics
    metrics = backtester.calculate_metrics()

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for key, value in metrics.items():
        if key != 'regime_distribution':
            print(f"{key:20}: {value}")

    print("\nRegime Distribution:")
    for regime, count in metrics.get('regime_distribution', {}).items():
        print(f"  {regime:20}: {count}")

    backtester.save_results(args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
