"""
Polygon.io S3 Flat Files Fetcher

Downloads historical trades and quotes from Polygon's S3 flat files.
Much faster than API calls (~10-30 sec per day vs 6-7 min).

S3 Endpoint: https://files.polygon.io
Bucket: flatfiles
Paths:
  - Trades: us_stocks_sip/trades_v1/YYYY/MM/YYYY-MM-DD.csv.gz
  - Quotes: us_stocks_sip/quotes_v1/YYYY/MM/YYYY-MM-DD.csv.gz
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from pathlib import Path
import logging
import gzip
import io
import os

try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None

logger = logging.getLogger(__name__)


class PolygonS3Fetcher:
    """
    Fetches historical trades and quotes from Polygon.io S3 flat files.

    Much faster than REST API for bulk historical data.
    Downloads entire market day files and filters for specific ticker.
    """

    # Default values (can be overridden via config)
    DEFAULT_ENDPOINT = "https://files.polygon.io"
    DEFAULT_BUCKET = "flatfiles"

    # S3 paths
    TRADES_PATH = "us_stocks_sip/trades_v1"
    QUOTES_PATH = "us_stocks_sip/quotes_v1"

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        bucket: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize S3 client for Polygon flat files.

        Args:
            access_key: Polygon S3 Access Key (or env POLYGON_S3_ACCESS_KEY)
            secret_key: Polygon S3 Secret Key (or env POLYGON_S3_SECRET_KEY)
            endpoint_url: S3 endpoint URL (default: https://files.polygon.io)
            bucket: S3 bucket name (default: flatfiles)
            cache_dir: Directory to cache downloaded files
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is not installed. Install with: pip install boto3"
            )

        # Get credentials from params or environment
        self.access_key = access_key or os.environ.get('POLYGON_S3_ACCESS_KEY')
        self.secret_key = secret_key or os.environ.get('POLYGON_S3_SECRET_KEY')

        if not self.access_key or not self.secret_key:
            raise ValueError(
                "Polygon S3 credentials required. Set POLYGON_S3_ACCESS_KEY and "
                "POLYGON_S3_SECRET_KEY environment variables or pass to constructor."
            )

        # S3 endpoint and bucket (configurable)
        self.endpoint_url = endpoint_url or self.DEFAULT_ENDPOINT
        self.bucket = bucket or self.DEFAULT_BUCKET

        # Configure boto3 client
        config = Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=config
        )

        # Cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "data" / "s3_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Polygon S3 client initialized (endpoint: {self.endpoint_url}, bucket: {self.bucket})")

    def _get_s3_key(self, data_type: str, date: str) -> str:
        """
        Build S3 key for a specific date and data type.

        Args:
            data_type: 'trades' or 'quotes'
            date: Date string YYYY-MM-DD

        Returns:
            S3 key path
        """
        dt = datetime.strptime(date, '%Y-%m-%d')
        year = dt.strftime('%Y')
        month = dt.strftime('%m')

        base_path = self.TRADES_PATH if data_type == 'trades' else self.QUOTES_PATH
        return f"{base_path}/{year}/{month}/{date}.csv.gz"

    def _download_and_filter(
        self,
        s3_key: str,
        ticker: str,
        data_type: str
    ) -> pd.DataFrame:
        """
        Download S3 file and filter for specific ticker.

        Args:
            s3_key: S3 object key
            ticker: Stock ticker to filter for
            data_type: 'trades' or 'quotes'

        Returns:
            DataFrame with filtered data
        """
        logger.info(f"Downloading {s3_key}...")

        try:
            # Download file to memory
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            compressed_data = response['Body'].read()

            file_size_mb = len(compressed_data) / (1024 * 1024)
            logger.info(f"Downloaded {file_size_mb:.1f} MB, decompressing...")

            # Decompress and read as CSV
            with gzip.open(io.BytesIO(compressed_data), 'rt') as f:
                # Read only rows for our ticker using chunked reading
                chunks = []
                chunk_size = 500000  # Read 500k rows at a time

                for chunk in pd.read_csv(f, chunksize=chunk_size):
                    # Filter for ticker
                    filtered = chunk[chunk['ticker'] == ticker]
                    if not filtered.empty:
                        chunks.append(filtered)

                if not chunks:
                    logger.warning(f"No data found for ticker {ticker}")
                    return pd.DataFrame()

                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Filtered {len(df)} rows for {ticker}")

                return df

        except self.s3.exceptions.NoSuchKey:
            logger.warning(f"S3 key not found: {s3_key}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            raise

    def fetch_trades(
        self,
        ticker: str,
        date: str
    ) -> pd.DataFrame:
        """
        Fetch trades for a ticker on a specific date.

        Args:
            ticker: Stock ticker (e.g., 'QQQ')
            date: Date string YYYY-MM-DD

        Returns:
            DataFrame with columns: [timestamp, price, size, exchange, conditions]
        """
        s3_key = self._get_s3_key('trades', date)
        df = self._download_and_filter(s3_key, ticker, 'trades')

        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'price', 'size', 'exchange', 'conditions'])

        # Convert timestamp (nanoseconds) to datetime
        df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns', utc=True)

        # Select and rename columns to match existing API format
        result = pd.DataFrame({
            'timestamp': df['timestamp'],
            'price': df['price'].astype(float),
            'size': df['size'].astype(int),
            'exchange': df['exchange'],
            'conditions': df['conditions'].fillna('[]')
        })

        return result.sort_values('timestamp').reset_index(drop=True)

    def fetch_quotes(
        self,
        ticker: str,
        date: str
    ) -> pd.DataFrame:
        """
        Fetch quotes for a ticker on a specific date.

        Args:
            ticker: Stock ticker (e.g., 'QQQ')
            date: Date string YYYY-MM-DD

        Returns:
            DataFrame with columns: [timestamp, bid_price, bid_size, ask_price, ask_size, ...]
        """
        s3_key = self._get_s3_key('quotes', date)
        df = self._download_and_filter(s3_key, ticker, 'quotes')

        if df.empty:
            return pd.DataFrame(columns=[
                'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size',
                'bid_exchange', 'ask_exchange'
            ])

        # Convert timestamp (nanoseconds) to datetime
        df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns', utc=True)

        # Select and rename columns to match existing API format
        result = pd.DataFrame({
            'timestamp': df['timestamp'],
            'bid_price': df['bid_price'].astype(float),
            'bid_size': df['bid_size'].astype(int),
            'ask_price': df['ask_price'].astype(float),
            'ask_size': df['ask_size'].astype(int),
            'bid_exchange': df['bid_exchange'],
            'ask_exchange': df['ask_exchange']
        })

        return result.sort_values('timestamp').reset_index(drop=True)

    def list_available_dates(
        self,
        data_type: str = 'trades',
        year: int = None,
        month: int = None
    ) -> list:
        """
        List available dates in S3.

        Args:
            data_type: 'trades' or 'quotes'
            year: Optional year filter
            month: Optional month filter (requires year)

        Returns:
            List of available date strings
        """
        base_path = self.TRADES_PATH if data_type == 'trades' else self.QUOTES_PATH

        if year and month:
            prefix = f"{base_path}/{year}/{month:02d}/"
        elif year:
            prefix = f"{base_path}/{year}/"
        else:
            prefix = f"{base_path}/"

        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1000
            )

            dates = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.csv.gz'):
                    # Extract date from filename
                    filename = key.split('/')[-1]
                    date = filename.replace('.csv.gz', '')
                    dates.append(date)

            return sorted(dates)

        except Exception as e:
            logger.error(f"Error listing S3 objects: {e}")
            return []

    def test_connection(self) -> bool:
        """
        Test S3 connection and credentials.

        Returns:
            True if connection successful
        """
        try:
            # Try to list a small prefix
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix="us_stocks_sip/trades_v1/2024/",
                MaxKeys=1
            )

            if 'Contents' in response:
                logger.info("S3 connection test successful")
                return True
            else:
                logger.warning("S3 connection OK but no data found")
                return True

        except Exception as e:
            logger.error(f"S3 connection test failed: {e}")
            return False


def test_s3_fetcher():
    """Quick test of S3 fetcher functionality."""
    import time

    logging.basicConfig(level=logging.INFO)

    # Check for credentials
    access_key = os.environ.get('POLYGON_S3_ACCESS_KEY')
    secret_key = os.environ.get('POLYGON_S3_SECRET_KEY')

    if not access_key or not secret_key:
        print("Set POLYGON_S3_ACCESS_KEY and POLYGON_S3_SECRET_KEY to test")
        return

    fetcher = PolygonS3Fetcher(access_key, secret_key)

    # Test connection
    print("\n=== Testing S3 Connection ===")
    if not fetcher.test_connection():
        print("Connection failed!")
        return

    # List recent dates
    print("\n=== Recent Available Dates ===")
    dates = fetcher.list_available_dates('trades', 2024, 12)
    print(f"Available dates in Dec 2024: {dates[-5:] if dates else 'None'}")

    # Test fetching trades for a recent date
    if dates:
        test_date = dates[-1]
        print(f"\n=== Fetching QQQ Trades for {test_date} ===")

        start_time = time.time()
        trades_df = fetcher.fetch_trades('QQQ', test_date)
        elapsed = time.time() - start_time

        print(f"Fetched {len(trades_df)} trades in {elapsed:.1f} seconds")
        if not trades_df.empty:
            print(f"First trade: {trades_df.iloc[0].to_dict()}")
            print(f"Last trade: {trades_df.iloc[-1].to_dict()}")


if __name__ == "__main__":
    test_s3_fetcher()
