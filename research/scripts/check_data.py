#!/usr/bin/env python3
"""Check what data the API is returning"""

import requests
import json

response = requests.get("http://localhost:5000/api/initial_data")
data = response.json()

print("Keys in response:", data.keys())
print()

if 'historical' in data:
    if data['historical']:
        print(f"Historical data: {len(data['historical'])} bars")
        print(f"First bar: {data['historical'][0]}")
        print(f"Last bar: {data['historical'][-1]}")
    else:
        print("Historical data is empty or None")
else:
    print("No historical key in response")

print()

if 'prediction' in data:
    if data['prediction']:
        print("Prediction keys:", data['prediction'].keys())
        print(f"Current close: {data['prediction'].get('current_close')}")
        print(f"P(up): {data['prediction'].get('p_up_30m')}")
    else:
        print("Prediction is None")
else:
    print("No prediction key in response")