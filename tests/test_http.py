#!/usr/bin/env python3
import requests
import json

def test_backend():
    try:
        # Test basic HTTP endpoint
        response = requests.get('http://localhost:8000/status')
        print(f"Status endpoint: {response.json()}")
        
        # Test start endpoint
        response = requests.post('http://localhost:8000/start')
        print(f"Start endpoint: {response.json()}")
        
        # Test pause endpoint  
        response = requests.post('http://localhost:8000/pause')
        print(f"Pause endpoint: {response.json()}")
        
        # Test reset endpoint
        response = requests.post('http://localhost:8000/reset')
        print(f"Reset endpoint: {response.json()}")
        
    except Exception as e:
        print(f"HTTP test failed: {e}")

if __name__ == "__main__":
    test_backend()
