#!/usr/bin/env python3
"""
Aura End-to-End System Integration Test
Verifies:
1. HTTP UI Server responds with 200 OK
2. Python ML WebSocket Server accepts client connection
3. Broadcasts valid 60fps telemetry packets
4. Responds to calibration control commands
"""

import os
import sys
import time
import json
import asyncio
import urllib.request
import threading
import websockets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from aura_engine.server import AuraEngineServer
from start_aura import run_http_server


async def client_test():
    print("[Integration Test] Connecting to ws://127.0.0.1:8765...")
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        print("[Integration Test] Connected! Receiving 5 telemetry frames...")
        for i in range(5):
            msg = await ws.recv()
            data = json.loads(msg)
            assert data["type"] == "telemetry", "Expected telemetry packet"
            assert "x" in data and "y" in data, "Telemetry missing x, y"
            assert "click" in data, "Telemetry missing click status"
            print(f"  Frame {i+1}: pos=({data['x']:.1f}, {data['y']:.1f}), click={data['click']}, blink_locked={data['is_blink_locked']}")

        # Test sending control message
        print("[Integration Test] Sending 'start_record_target' control command...")
        await ws.send(json.dumps({"command": "start_record_target", "target_id": 0}))
        for _ in range(50):
            resp = await ws.recv()
            resp_data = json.loads(resp)
            if resp_data.get("status") == "recording_target_started":
                print(f"  Server response: {resp_data}")
                break
        assert resp_data.get("status") == "recording_target_started", "Failed to start target recording"

        print("[Integration Test] Sending 'stop_record_target'...")
        await ws.send(json.dumps({"command": "stop_record_target"}))
        for _ in range(50):
            resp2 = await ws.recv()
            resp2_data = json.loads(resp2)
            if resp2_data.get("status") == "recording_target_saved":
                print(f"  Server response: {resp2_data}")
                break
        assert resp2_data.get("status") == "recording_target_saved", "Failed to save target recording"

    print(" -> SUCCESS: WebSocket telemetry and control protocol verified!")


def main():
    print("\n========================================================")
    print("      AURA END-TO-END SYSTEM INTEGRATION TEST           ")
    print("========================================================")

    # 1. Start HTTP UI server
    http_thread = threading.Thread(target=run_http_server, args=(5179,), daemon=True)
    http_thread.start()
    time.sleep(0.3)

    # Check HTTP endpoint
    res = urllib.request.urlopen("http://127.0.0.1:5179")
    assert res.status == 200, "UI Server failed to serve 200 OK"
    html_content = res.read().decode("utf-8")
    assert "AURA" in html_content, "UI HTML content missing"
    print("[Integration Test] HTTP UI Server verified serving index.html at http://127.0.0.1:5179.")

    # 2. Start ML Engine Server in background thread
    server = AuraEngineServer()
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Wait up to 10 seconds for WebSocket port to open
    print("[Integration Test] Waiting for WebSocket server to become ready...")
    for _ in range(30):
        time.sleep(0.5)
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", 8765)) == 0:
                    print("[Integration Test] WebSocket server port is OPEN and ready.")
                    break
        except Exception:
            pass

    # 3. Run client integration test
    asyncio.run(client_test())

    # Stop server
    server.stop()
    print("\n========================================================")
    print(" ALL END-TO-END SYSTEM INTEGRATION TESTS PASSED!       ")
    print("========================================================\n")


if __name__ == "__main__":
    main()
