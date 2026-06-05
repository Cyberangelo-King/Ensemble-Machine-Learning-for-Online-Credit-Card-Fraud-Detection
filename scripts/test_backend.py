import requests
import json
from websocket import create_connection

def test_rest():
    print("Testing REST endpoints...")
    # Health check/root not defined but let's try explain
    try:
        response = requests.get("http://localhost:8000/explain/0")
        print(f"Explain response status: {response.status_code}")
        if response.status_code == 200:
            print("Explain endpoint works.")
    except Exception as e:
        print(f"REST test failed: {e}")

def test_websocket():
    print("Testing WebSocket...")
    try:
        ws = create_connection("ws://localhost:8000/ws/stream")
        ws.send(json.dumps({"action": "start", "interval": 0.1}))
        result = ws.recv()
        print(f"Received message: {result[:100]}...")
        ws.close()
        print("WebSocket test successful.")
    except Exception as e:
        print(f"WebSocket test failed: {e}")

if __name__ == "__main__":
    # Note: Backend must be running
    test_rest()
    test_websocket()
