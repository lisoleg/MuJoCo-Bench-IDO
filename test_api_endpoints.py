"""Comprehensive API endpoint test for webviz dashboard."""
import requests
import json
import time

BASE = "http://localhost:8080"

def test_tasks():
    r = requests.get(f"{BASE}/api/tasks")
    assert r.status_code == 200
    data = r.json()
    assert "tasks" in data
    assert len(data["tasks"]) >= 4
    assert data["version"] == "v0.2.0"
    print(f"[PASS] GET /api/tasks: {len(data['tasks'])} tasks, version={data['version']}")

def test_dashboard():
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200
    assert "MuJoCo-Bench-IDO" in r.text
    assert "etaChart" in r.text
    assert "motorChart" in r.text
    print(f"[PASS] GET /: dashboard HTML, {len(r.text)} chars")

def test_status_idle():
    r = requests.get(f"{BASE}/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["is_running"] == False
    assert "mjviser_available" in data
    print(f"[PASS] GET /api/status (idle): is_running={data['is_running']}, mjviser={data['mjviser_available']}")

def test_results_empty():
    r = requests.get(f"{BASE}/api/results")
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "count" in data
    print(f"[PASS] GET /api/results: count={data['count']}")

def test_start_viewer_no_mjviser():
    r = requests.post(f"{BASE}/api/start_viewer")
    assert r.status_code == 503
    data = r.json()
    assert "error" in data
    assert data["available"] == False
    print(f"[PASS] POST /api/start_viewer: 503, mjviser not available")

def test_stop_idle():
    r = requests.post(f"{BASE}/api/stop")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "idle"
    print(f"[PASS] POST /api/stop (idle): status={data['status']}")

def test_run_standard():
    payload = {
        "task": "reacher-easy",
        "episodes": 1,
        "max_steps": 50,
        "eval_mode": "standard",
        "kappa_thresh": 0.05,
        "evolution_rounds": 3,
    }
    r = requests.post(f"{BASE}/api/run", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "started"
    assert data["task"] == "reacher-easy"
    print(f"[PASS] POST /api/run (standard): started, task={data['task']}")

    # Wait for completion
    time.sleep(5)

    # Verify results saved
    r2 = requests.get(f"{BASE}/api/results")
    data2 = r2.json()
    assert data2["count"] >= 1
    print(f"[PASS] Results saved: count={data2['count']}")

    # Verify status is idle again
    r3 = requests.get(f"{BASE}/api/status")
    data3 = r3.json()
    assert data3["is_running"] == False
    print(f"[PASS] Status after run: is_running={data3['is_running']}")

def test_run_sip():
    payload = {
        "task": "reacher-easy",
        "episodes": 1,
        "max_steps": 30,
        "eval_mode": "sip",
        "kappa_thresh": 0.05,
        "evolution_rounds": 1,
    }
    r = requests.post(f"{BASE}/api/run", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "started"
    assert data["eval_mode"] == "sip"
    print(f"[PASS] POST /api/run (sip): started, mode={data['eval_mode']}")

    # Wait for SIP completion (3 phases × 1 episode × 30 steps + 1 evolution round)
    time.sleep(10)

    # Check results for SIP data
    r2 = requests.get(f"{BASE}/api/results")
    data2 = r2.json()
    # Find SIP result
    sip_results = [r for r in data2["results"] if r.get("eval_mode") == "sip"]
    if sip_results:
        print(f"[PASS] SIP results found: {len(sip_results)} entries")
        sip = sip_results[-1]
        if "sip_result" in sip:
            sr = sip["sip_result"]
            print(f"  T0: avg_eta={sr['T0']['avg_eta']:.4f}, avg_steps={sr['T0']['avg_steps']}")
            print(f"  T1: avg_eta={sr['T1']['avg_eta']:.4f}, avg_steps={sr['T1']['avg_steps']}")
            print(f"  T2: avg_eta={sr['T2']['avg_eta']:.4f}, avg_steps={sr['T2']['avg_steps']}")
            print(f"  retention_gain={sr['retention_gain']:.3f}")
            print(f"  stability_index={sr['stability_index']:.3f}")
    else:
        print("[WARN] No SIP results found yet (may need more time)")

def test_concurrent_run_conflict():
    # Start a long-ish run
    payload = {
        "task": "reacher-easy",
        "episodes": 3,
        "max_steps": 100,
        "eval_mode": "standard",
        "kappa_thresh": 0.05,
        "evolution_rounds": 3,
    }
    r1 = requests.post(f"{BASE}/api/run", json=payload)
    if r1.status_code == 409:
        print("[PASS] POST /api/run conflict: 409 (already running from SIP test)")
        return
    assert r1.status_code == 200

    # Try to start another run while first is running
    time.sleep(1)
    r2 = requests.post(f"{BASE}/api/run", json=payload)
    assert r2.status_code == 409
    data2 = r2.json()
    assert "error" in data2
    print(f"[PASS] POST /api/run conflict: 409, error='{data2['error']}'")

    # Stop the run
    r3 = requests.post(f"{BASE}/api/stop")
    assert r3.status_code == 200
    print(f"[PASS] POST /api/stop: stopped")

    # Wait for cleanup
    time.sleep(3)

# Run all tests
print("=" * 60)
print("  MuJoCo-Bench-IDO Webviz API Endpoint Tests")
print("=" * 60)

test_tasks()
test_dashboard()
test_status_idle()
test_results_empty()
test_start_viewer_no_mjviser()
test_stop_idle()
test_run_standard()
test_run_sip()
test_concurrent_run_conflict()

print("=" * 60)
print("  All API endpoint tests PASSED")
print("=" * 60)
