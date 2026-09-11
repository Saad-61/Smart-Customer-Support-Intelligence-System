"""
Comprehensive Offline Test Suite for FastAPI REST Inference Service (Module 11)
Uses starlette/fastapi TestClient to test all endpoints without binding a socket port.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from api.app import app


def test_api_suite():
    print("=" * 80)
    print("  RUNNING FASTAPI REST SERVICE TEST SUITE")
    print("=" * 80)

    # Use TestClient with lifespan context
    with TestClient(app) as client:
        # 1. Test Root endpoint
        print("\n[1/6] Testing GET / ...")
        resp = client.get("/")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "online"
        print(" -> GET / PASSED:", data["service"])

        # 2. Test Health endpoint
        print("\n[2/6] Testing GET /health ...")
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        health_data = resp.json()
        assert health_data["status"] == "ok"
        assert health_data["models_loaded"] is True
        print(f" -> GET /health PASSED: device='{health_data['device']}', total_indexed={health_data['total_indexed_tickets']}")

        # 3. Test POST /predict with standard in-domain complaint
        print("\n[3/6] Testing POST /predict (In-Domain complaint) ...")
        payload = {
            "ticket_text": "I was charged twice for the same RP bundle",
            "product": "League of Legends",
            "previous_tickets": 1,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        pred = resp.json()
        print(f" -> Category: '{pred['category']}' (confidence: {pred['category_confidence']})")
        print(f" -> Priority: '{pred['priority']}' (confidence: {pred['priority_confidence']})")
        print(f" -> Uncertain (OOD): {pred['uncertain']}")
        print(f" -> Processing latency: {pred['processing_time_ms']} ms")
        print(f" -> Top explanation features: {[f['feature'] for f in pred['explanation'][:3]]}")
        print(f" -> Similar tickets retrieved: {len(pred['similar_tickets'])}")
        assert pred["category"] == "Missing RP / Purchase Issue"
        assert pred["uncertain"] is False
        assert len(pred["similar_tickets"]) > 0
        assert len(pred["explanation"]) > 0
        print(" -> POST /predict (In-Domain) PASSED!")

        # 4. Test POST /predict with empty/whitespace input (Expecting HTTP 422)
        print("\n[4/6] Testing POST /predict with empty/whitespace input (Validation check) ...")
        invalid_payload = {"ticket_text": "   ", "product": "League of Legends"}
        resp = client.post("/predict", json=invalid_payload)
        assert resp.status_code == 422, f"Expected 422 Unprocessable Entity, got {resp.status_code}: {resp.text}"
        print(" -> Validation check PASSED: Received HTTP 422 Unprocessable Entity as required.")

        # 5. Test POST /predict with Out-of-Distribution query
        print("\n[5/6] Testing POST /predict with Out-of-Distribution (OOD) query ...")
        ood_payload = {
            "ticket_text": "What is the weather in London today and how do I bake bread?",
            "product": "League of Legends",
        }
        resp = client.post("/predict", json=ood_payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        ood_pred = resp.json()
        print(f" -> OOD Category confidence: {ood_pred['category_confidence']}")
        print(f" -> OOD Uncertain flag: {ood_pred['uncertain']}")
        assert ood_pred["uncertain"] is True, "Expected uncertain=True for off-domain query"
        print(" -> OOD guardrail check PASSED!")

        # 6. Test standalone /similar and /explain endpoints
        print("\n[6/6] Testing POST /similar and POST /explain ...")
        sim_resp = client.post("/similar", json={"query_text": "game crashing every match on load", "top_k": 3})
        assert sim_resp.status_code == 200, f"Expected 200, got {sim_resp.status_code}: {sim_resp.text}"
        sim_data = sim_resp.json()
        assert len(sim_data["results"]) == 3
        print(f" -> POST /similar PASSED: Retrieved {len(sim_data['results'])} nearest tickets.")

        exp_resp = client.post("/explain", json={"ticket_text": "Permaban for third party script abuse", "top_n": 4})
        assert exp_resp.status_code == 200, f"Expected 200, got {exp_resp.status_code}: {exp_resp.text}"
        exp_data = exp_resp.json()
        assert len(exp_data["top_features"]) > 0
        print(f" -> POST /explain PASSED: Category '{exp_data['predicted_category']}' explained by {[f['feature'] for f in exp_data['top_features']]}.")

    print("\n" + "=" * 80)
    print("  ALL FASTAPI REST INFERENCE SERVICE TESTS PASSED PERFECTLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_api_suite()
