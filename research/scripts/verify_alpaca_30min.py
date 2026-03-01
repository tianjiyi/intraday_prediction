
import os
import sys
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pytz
import yaml

# Load config to get keys
def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

config = load_config()
api_key = config['ALPACA_KEY_ID']
secret_key = config['ALPACA_SECRET_KEY']

client = StockHistoricalDataClient(api_key, secret_key)

# Test 30 minute timeframe
timeframe = TimeFrame(30, TimeFrameUnit.Minute)
symbol = "QQQ"

end_time = datetime.now(pytz.timezone('US/Eastern'))
# Start at a weird time (e.g. 9:45)
start_time = (end_time - timedelta(days=2)).replace(minute=45, second=0, microsecond=0)

print(f"Requesting {symbol} 30min data starting at {start_time}...")

request = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=timeframe,
    start=start_time,
    end=end_time
)

print(f"Requesting {symbol} 30min data...")
try:
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    print(f"Received {len(df)} bars")
    print(df[['timestamp', 'open', 'close']].head(10))
    
    # Check intervals
    df['diff'] = df['timestamp'].diff()
    print("\nTime differences:")
    print(df['diff'].value_counts())
    
except Exception as e:
    print(f"Error: {e}")
