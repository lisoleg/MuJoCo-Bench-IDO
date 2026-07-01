"""WebSocket streaming test for webviz dashboard."""
import asyncio
import json
import requests
import time
import websockets

received_messages = []

async def ws_listener():
    uri = "ws://localhost:8080/ws/stream"
    async with websockets.connect(uri) as ws:
        # Read initial message
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(msg)
        print(f"Initial WS msg: {data.get('type')}")

        # Now start a benchmark run
        payload = {
            "task": "reacher-easy",
            "episodes": 1,
            "max_steps": 50,
            "eval_mode": "standard",
            "kappa_thresh": 0.05,
            "evolution_rounds": 3,
        }
        r = requests.post("http://localhost:8080/api/run", json=payload)
        print(f"Run started: {r.status_code} {r.json()}")

        # Collect messages for up to 15 seconds
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                data = json.loads(msg)
                received_messages.append(data)
                msg_type = data.get("type", "step_data")
                if "step" in data and "eta" in data:
                    print(f"Step {data['step']}: eta={data['eta']:.4f}, nv={data['noether_violations']}")
                elif msg_type == "run_start":
                    print(f"Run start: {data}")
                elif msg_type == "episode_complete":
                    print(f"Episode complete: steps={data.get('metrics', {}).get('steps_to_goal')}")
                elif msg_type == "run_complete":
                    print("Run complete!")
                    break
                elif msg_type == "error":
                    print(f"Error: {data.get('message')}")
                    break
                else:
                    print(f"Other msg: {msg_type}")
        except asyncio.TimeoutError:
            print("Timeout waiting for messages")

        print(f"Total WS messages received: {len(received_messages)}")
        # Check if step-level data was received
        step_msgs = [m for m in received_messages if "step" in m and "eta" in m]
        print(f"Step-level messages with eta: {len(step_msgs)}")

asyncio.run(ws_listener())
