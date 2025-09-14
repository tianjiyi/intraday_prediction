#!/usr/bin/env python3
"""
Test script to verify the API endpoints are working
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_homepage():
    """Test if homepage is accessible"""
    print("Testing homepage...")
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        print("[OK] Homepage accessible")
        return True
    else:
        print(f"[FAIL] Homepage error: {response.status_code}")
        return False

def test_initial_data():
    """Test initial data endpoint"""
    print("\nTesting initial data endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/initial_data", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'historical' in data:
                print(f"[OK] Historical data: {len(data.get('historical', []))} bars")
            if 'prediction' in data:
                pred = data['prediction']
                if pred:
                    print(f"[OK] Prediction data received")
                    print(f"  - Current price: ${pred.get('current_close', 0):.2f}")
                    print(f"  - P(Up): {pred.get('p_up_30m', 0)*100:.1f}%")
                    print(f"  - Expected return: {pred.get('exp_ret_30m', 0)*100:.3f}%")
                else:
                    print("[WARN] Prediction data is None")
            return True
        else:
            print(f"[FAIL] API error: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("[FAIL] Request timed out (model may be loading)")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def test_latest_prediction():
    """Test latest prediction endpoint"""
    print("\nTesting latest prediction endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/latest_prediction", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'prediction' in data:
                print("[OK] Latest prediction endpoint working")
                return True
        else:
            print(f"[FAIL] API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Live Chart Prediction API")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Run tests
    tests = [
        test_homepage(),
        test_initial_data(),
        test_latest_prediction()
    ]
    
    # Summary
    print("\n" + "=" * 50)
    if all(tests):
        print("[OK] All tests passed!")
        print("\nYou can now open http://localhost:5000 in your browser")
    else:
        print("[WARN] Some tests failed - check the server logs")
    print("=" * 50)