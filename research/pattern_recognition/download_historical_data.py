"""
Download historical 1-min data from Alpaca and save locally.
This allows processing without additional API requests.

Usage:
    python -m pattern_recognition.download_historical_data --ticker QQQ --start 2021 --end 2025
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pytz
import yaml
import time

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("alpaca-py not installed")

# Output directory
DATA_DIR = Path(__file__).parent.parent / "historical_data"


def load_config():
    """Load Alpaca API keys from config."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_year_data(client, ticker: str, year: int) -> pd.DataFrame:
    """
    Download 1-min data for a full year.
    Alpaca allows fetching large date ranges in one request for SIP feed.
    """
    eastern = pytz.timezone('US/Eastern')

    start = datetime(year, 1, 1, tzinfo=eastern)
    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=eastern)

    # Don't fetch future dates
    now = datetime.now(eastern)
    if end > now:
        end = now

    if start > now:
        print(f"  {year}: Skipping (future year)")
        return None

    print(f"  {year}: Fetching {start.date()} to {end.date()}...")

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.SIP
    )

    bars = client.get_stock_bars(request)

    if ticker not in bars.data or len(bars.data[ticker]) == 0:
        print(f"  {year}: No data returned")
        return None

    # Convert to DataFrame
    records = []
    for bar in bars.data[ticker]:
        records.append({
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'vwap': bar.vwap if hasattr(bar, 'vwap') else None,
            'trade_count': bar.trade_count if hasattr(bar, 'trade_count') else None
        })

    df = pd.DataFrame(records)
    print(f"  {year}: Downloaded {len(df):,} bars")

    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Regular Trading Hours (9:30 AM - 4:00 PM ET)."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert to Eastern time
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp_et'] = df['timestamp'].dt.tz_convert('US/Eastern')

    # Filter RTH
    df['time'] = df['timestamp_et'].dt.time
    from datetime import time as dt_time
    rth_start = dt_time(9, 30)
    rth_end = dt_time(15, 59)

    df_rth = df[(df['time'] >= rth_start) & (df['time'] <= rth_end)].copy()
    df_rth = df_rth.drop(columns=['timestamp_et', 'time'])

    return df_rth


def main():
    parser = argparse.ArgumentParser(description="Download historical 1-min data")
    parser.add_argument('--ticker', type=str, default='QQQ', help='Ticker symbol')
    parser.add_argument('--start', type=int, default=2021, help='Start year')
    parser.add_argument('--end', type=int, default=2025, help='End year')
    parser.add_argument('--rth-only', action='store_true', default=True, help='RTH only')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Historical Data Downloader")
    print(f"Ticker: {args.ticker}, Years: {args.start}-{args.end}")
    print("=" * 60)

    if not ALPACA_AVAILABLE:
        print("ERROR: alpaca-py not installed")
        return

    # Setup
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()

    api_key = config.get('ALPACA_KEY_ID')
    secret_key = config.get('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("ERROR: Alpaca API keys not found in config.yaml")
        return

    client = StockHistoricalDataClient(api_key, secret_key)

    # Download each year
    all_data = []

    for year in range(args.start, args.end + 1):
        try:
            df = download_year_data(client, args.ticker, year)
            if df is not None and not df.empty:
                all_data.append(df)

            # Small delay between years to avoid rate limits
            time.sleep(1)

        except Exception as e:
            print(f"  {year}: ERROR - {e}")
            time.sleep(5)  # Wait longer on error

    if not all_data:
        print("\nNo data downloaded!")
        return

    # Combine all years
    print("\nCombining data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['timestamp'])

    print(f"Total bars (all hours): {len(combined_df):,}")

    # Filter RTH if requested
    if args.rth_only:
        print("Filtering to RTH only...")
        combined_df = filter_rth(combined_df)
        print(f"Total bars (RTH only): {len(combined_df):,}")

    # Save to parquet (efficient storage)
    suffix = "rth" if args.rth_only else "all"
    output_file = DATA_DIR / f"{args.ticker}_{args.start}_{args.end}_1min_{suffix}.parquet"
    combined_df.to_parquet(output_file, index=False)

    # Also save as CSV for easy inspection
    csv_file = DATA_DIR / f"{args.ticker}_{args.start}_{args.end}_1min_{suffix}.csv"
    combined_df.to_csv(csv_file, index=False)

    # Summary
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Ticker: {args.ticker}")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"Total bars: {len(combined_df):,}")
    print(f"Parquet file: {output_file} ({file_size_mb:.1f} MB)")
    print(f"CSV file: {csv_file}")

    # Show bars per year
    combined_df['year'] = pd.to_datetime(combined_df['timestamp']).dt.year
    print("\nBars per year:")
    for year, count in combined_df.groupby('year').size().items():
        print(f"  {year}: {count:,}")


if __name__ == "__main__":
    main()
