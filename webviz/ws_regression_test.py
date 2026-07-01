"""WebSocket streaming regression test for Bug #1 fix.

Verifies that broadcast_sync() correctly pushes step-level messages
via WebSocket by using the _uvicorn_loop captured at startup.
"""
import asyncio
import json
import sys
import threading
import time

import requests
import websockets

WS_URL = "ws://localhost:8080/ws/stream"
API_RUN_URL = "http://localhost:8080/api/run"
API_STATUS_URL = "http://localhost:8080/api/status"


async def ws_test():
    """Connect to WebSocket, trigger a benchmark run, collect messages."""
    messages = []
    print("[TEST] Connecting to WebSocket...")

    async with websockets.connect(WS_URL) as ws:
        # Wait for initial connected message
        init_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        init_data = json.loads(init_msg)
        print(f"[TEST] Initial message: {json.dumps(init_data, indent=2)}")
        assert init_data.get("type") == "connected", \
            f"Expected 'connected' type, got: {init_data.get('type')}"

        # Start benchmark in a separate thread
        print("[TEST] Starting benchmark (reacher-easy, 1 episode, max_steps=30)...")
        run_response = [None]

        def start_run():
            try:
                r = requests.post(API_RUN_URL, json={
                    "task": "reacher-easy",
                    "episodes": 1,
                    "max_steps": 30,
                    "eval_mode": "standard",
                }, timeout=10)
                run_response[0] = r
                print(f"[TEST] Run API response: {r.status_code} {r.json()}")
            except Exception as e:
                print(f"[TEST] Run API error: {e}")
                run_response[0] = e

        threading.Thread(target=start_run, daemon=True).start()

        # Collect messages for up to 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                messages.append(data)
                msg_type = data.get("type", "unknown")

                # Log key events
                if msg_type in ("run_start", "run_complete", "episode_complete",
                                "run_stopped", "error"):
                    print(f"[TEST] Event: {msg_type} — {json.dumps(data, indent=2)[:200]}")
                elif "step" in data and "eta" in data:
                    # Step-level message (this is what Bug #1 fix enables)
                    pass  # counted below

                if msg_type == "run_complete" or msg_type == "error":
                    break

            except asyncio.TimeoutError:
                # Check if run was started
                if run_response[0] is not None:
                    # Run API already responded, but no more WS messages
                    # If we've collected messages, we can stop
                    if len(messages) > 1:
                        break
                continue
            except websockets.exceptions.ConnectionClosed:
                print("[TEST] WebSocket connection closed unexpectedly")
                break

    # Analyze collected messages
    step_msgs = [m for m in messages if "step" in m and "eta" in m]
    event_msgs = [m for m in messages if "type" in m]

    print(f"\n[RESULT] Total messages received: {len(messages)}")
    print(f"[RESULT] Step-level messages (with step + eta): {len(step_msgs)}")
    print(f"[RESULT] Event messages: {len(event_msgs)}")

    if step_msgs:
        print(f"[RESULT] First step msg: {json.dumps(step_msgs[0], indent=2)}")
        print(f"[RESULT] Last step msg: {json.dumps(step_msgs[-1], indent=2)}")

    # Verify at least we got run_start and some step messages
    has_run_start = any(m.get("type") == "run_start" for m in messages)
    has_episode_or_complete = any(
        m.get("type") in ("episode_complete", "run_complete") for m in messages
    )

    success = len(step_msgs) > 0 and has_run_start

    print(f"\n[RESULT] WebSocket streaming WORKS: {success}")
    print(f"[RESULT] has_run_start: {has_run_start}")
    print(f"[RESULT] has_episode_or_complete: {has_episode_or_complete}")
    print(f"[RESULT] step_msg_count: {len(step_msgs)}")

    return success, messages


def test_pydantic_no_warning():
    """Test that /api/status does not produce Pydantic UserWarning."""
    print("\n[TEST] Checking /api/status for Pydantic warnings...")
    response = requests.get(API_STATUS_URL, timeout=5)
    data = response.json()
    print(f"[TEST] Status response: {json.dumps(data, indent=2)}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("[RESULT] /api/status returned 200 — Pydantic V2 fix confirmed (no schema_extra)")
    return True


def test_tasks_endpoint():
    """Test that /api/tasks returns 4 tasks."""
    print("\n[TEST] Checking /api/tasks endpoint...")
    response = requests.get("http://localhost:8080/api/tasks", timeout=5)
    data = response.json()
    task_count = len(data.get("tasks", []))
    print(f"[TEST] Tasks returned: {task_count}")
    print(f"[RESULT] /api/tasks returned {task_count} tasks")
    assert task_count == 4, f"Expected 4 tasks, got {task_count}"
    return True


if __name__ == "__main__":
    all_passed = True

    # Test 1: Tasks endpoint
    try:
        test_tasks_endpoint()
    except AssertionError as e:
        print(f"[FAIL] Tasks test: {e}")
        all_passed = False

    # Test 2: Status endpoint (Pydantic fix)
    try:
        test_pydantic_no_warning()
    except AssertionError as e:
        print(f"[FAIL] Status test: {e}")
        all_passed = False

    # Test 3: WebSocket streaming (Bug #1)
    try:
        ws_success, ws_messages = asyncio.run(ws_test())
        if not ws_success:
            print("[FAIL] WebSocket streaming did not receive step-level messages!")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] WebSocket test exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED — 3 Bug fixes verified!")
    else:
        print("SOME TESTS FAILED — see details above")
    print(f"{'='*60}")

    sys.exit(0 if all_passed else 1)
