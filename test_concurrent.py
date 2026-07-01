"""Test concurrent run conflict detection."""
import requests
import time

BASE = "http://localhost:8080"

# Start a longer run (5 episodes, 500 steps) to ensure it's still running when we try the second
payload = {
    "task": "reacher-easy",
    "episodes": 5,
    "max_steps": 500,
    "eval_mode": "standard",
    "kappa_thresh": 0.05,
    "evolution_rounds": 3,
}
r1 = requests.post(f"{BASE}/api/run", json=payload)
print(f"First run: {r1.status_code} {r1.json()}")

if r1.status_code == 200:
    # Immediately try a second run
    r2 = requests.post(f"{BASE}/api/run", json=payload)
    print(f"Second run (concurrent): {r2.status_code} {r2.json()}")
    if r2.status_code == 409:
        print("[PASS] Concurrent run conflict: 409 returned correctly")
    else:
        print(f"[WARN] Expected 409, got {r2.status_code}")

    # Stop the run
    r3 = requests.post(f"{BASE}/api/stop")
    print(f"Stop: {r3.status_code} {r3.json()}")
    time.sleep(2)
else:
    # Maybe already running
    print(f"First run returned {r1.status_code} - may already be running")
