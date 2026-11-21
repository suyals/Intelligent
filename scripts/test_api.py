import time
import requests
import subprocess
import sys

def test_api():
    print("=" * 60)
    print("STARTING FASTAPI BACKEND SERVER FOR INTEGRATION TESTING")
    print("=" * 60)

    # Start uvicorn server process
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    base_url = "http://127.0.0.1:8000"
    server_ready = False

    # Wait for server up to 15 seconds
    print("Waiting for server to initialize on http://127.0.0.1:8000 ...")
    for attempt in range(15):
        time.sleep(1)
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                server_ready = True
                print(f"Server is LIVE! (Attempt {attempt + 1})")
                break
        except Exception:
            continue

    if not server_ready:
        proc.terminate()
        out, err = proc.communicate(timeout=5)
        print("Server failed to start in time!")
        print("STDOUT:", out)
        print("STDERR:", err)
        sys.exit(1)

    try:
        # 1. Health Check
        print("\n1. Testing GET /health...")
        r = requests.get(f"{base_url}/health")
        print("   Status:", r.status_code)
        print("   Response:", r.json())
        assert r.status_code == 200
        assert r.json()["models_loaded"] is True

        # 2. Analytics Check
        print("\n2. Testing GET /analytics...")
        r = requests.get(f"{base_url}/analytics")
        print("   Status:", r.status_code)
        data = r.json()
        print(f"   Train Records: {data['dataset_summary']['train_total']}")
        print(f"   Test Records:  {data['dataset_summary']['test_total']}")
        print(f"   Priority Model Test Acc: {data['final_evaluations']['Priority']['test_accuracy'] * 100}%")
        assert r.status_code == 200

        # 3. Test Predictions on Real Dataset Cases
        print("\n3. Testing POST /predict on sample dataset queries...")
        test_queries = [
            ("Internship deadline today, need NOC approval.", "Registrar & Student Records", "High"),
            ("My fee payment failed but amount deducted.", "Finance & Accounts", "High"),
            ("I cannot submit my exam form and today is the last date.", "Academics & Examination", "High"),
            ("How to apply for revaluation?", "Academics & Examination", "Medium"),
            ("How can I apply for hostel change?", "Hostel & Housing", "Medium"),
            ("WiFi password for campus?", "IT & Technical Support", "Low"),
            ("What are library timings?", "Campus Facilities & Transport", "Low"),
            ("How to join sports club?", "Student Affairs & Clubs", "Low"),
        ]

        for query, exp_cat, exp_prio in test_queries:
            r = requests.post(f"{base_url}/predict", json={"complaint": query})
            assert r.status_code == 200, f"Failed for query: {query}"
            res = r.json()
            print(f"\n   Query: '{query}'")
            print(f"     -> Category: {res['predicted_category']} (conf: {res['category_confidence']})")
            print(f"     -> Priority: {res['predicted_priority']} (conf: {res['priority_confidence']})")
            print(f"     -> SLA: {res['resolution_sla']}")
            print(f"     -> Keywords: {res['contributing_keywords']}")
            assert res['predicted_category'] == exp_cat, f"Category mismatch: {res['predicted_category']} != {exp_cat}"
            assert res['predicted_priority'] == exp_prio, f"Priority mismatch: {res['predicted_priority']} != {exp_prio}"

        # 4. Batch Prediction Check
        print("\n4. Testing POST /predict-batch...")
        batch_queries = [q[0] for q in test_queries[:3]]
        r = requests.post(f"{base_url}/predict-batch", json={"complaints": batch_queries})
        assert r.status_code == 200
        batch_res = r.json()
        print(f"   Batch Total: {batch_res['total']}")
        assert batch_res['total'] == 3

        # 5. Frontend Serving Check
        print("\n5. Testing GET / (Frontend UI serving)...")
        r = requests.get(f"{base_url}/")
        assert r.status_code == 200
        assert "Intelligent Student Complaint" in r.text
        print("   Frontend HTML successfully served with 200 OK!")

        print("\n" + "=" * 60)
        print("ALL ENDPOINTS AND INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
        print("=" * 60)

    finally:
        print("\nStopping uvicorn test process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Uvicorn test process stopped.")

if __name__ == "__main__":
    test_api()

