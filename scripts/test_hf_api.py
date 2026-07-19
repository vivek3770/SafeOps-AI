import json
import requests
import time

def test_huggingface_api():
    print("=========================================================")
    print("      TESTING LIVE HUGGING FACE AI ENGINE API            ")
    print("=========================================================")
    
    # We are testing the Gradio API endpoint
    url = "https://monkey3770-safeops-ai-engine.hf.space/gradio_api/call/eval"
    
    # 1. Create a mock plant state (simulate a gas spike + worker)
    mock_state = {
        "zone_id": "ZONE_3",
        "current_time": "2026-07-01T12:00:20Z",
        "shift_risk_factor": 0.1,
        # Spike in gas levels to trigger critical risk
        "sensor_raw_history": [[35.0, 28.0, 1.2] for _ in range(12)],
        "cv_raw_frame": {
            "zone_id": "ZONE_3",
            "workers_detected": 1,
            "violations": [{"worker_id": "W_101", "violation_type": "NO_HELMET", "confidence": 0.9}]
        },
        "active_permits_raw": [
            {
                "permit_id": "HW_042",
                "type": "HOT_WORK",
                "zone_id": "ZONE_3",
                "start_time": "2026-07-01T12:00:10Z",
                "expiry": "2026-07-01T20:00:00Z"
            }
        ]
    }
    
    # 2. Wrap in Gradio's expected format
    payload = {
        "data": [json.dumps(mock_state)]
    }
    
    print(f"Sending POST request to: {url}")
    print("Waiting for AI Engine to calculate risk (might take 15-30s if Hugging Face is waking up from sleep)...")
    
    try:
        # We use a 60 second timeout because Hugging Face ZeroGPU spaces sometimes take 20s to wake up if asleep
        response = requests.post(url, json=payload, timeout=60.0)
            
        if response.status_code == 200:
            print("\n[SUCCESS] Received 200 OK from Hugging Face.")
            response_json = response.json()
            
            # Gradio sometimes returns an EVENT ID that we have to poll, or the raw data directly.
            # Let's see what the space returns.
            if "event_id" in response_json:
                print(f"Gradio returned an Event ID: {response_json['event_id']}")
                print("Wait, this means we need to poll the /call/eval/{event_id} endpoint.")
                
                # Poll for the result
                poll_url = f"{url}/{response_json['event_id']}"
                print(f"Polling {poll_url} for results...")
                
                # Gradio streams Server-Sent Events (SSE) from the poll URL
                poll_resp = requests.get(poll_url, timeout=60.0)
                print("\n--- RAW SSE RESPONSE ---")
                print(poll_resp.text)
                
                # A quick hack to find the final data line in the SSE stream
                for line in poll_resp.text.split('\n'):
                    if line.startswith('data: ['):
                        final_data_str = line[6:] # Strip 'data: '
                        final_ai_results = json.loads(json.loads(final_data_str)[0])
                        print("\n--- PARSED AI RESULTS ---")
                        print(f"Risk Score: {final_ai_results.get('risk_fusion_out', {}).get('score')}")
                        print(f"Severity: {final_ai_results.get('risk_fusion_out', {}).get('severity')}")
                        print(f"Action: {final_ai_results.get('action_taken')}")
                        break
            
            elif "data" in response_json:
                final_ai_results = json.loads(response_json["data"][0])
                print("\n--- PARSED AI RESULTS ---")
                print(f"Risk Score: {final_ai_results.get('risk_fusion_out', {}).get('score')}")
                print(f"Severity: {final_ai_results.get('risk_fusion_out', {}).get('severity')}")
                print(f"Action: {final_ai_results.get('action_taken')}")
                
        else:
            print(f"\n[ERROR] Received status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n[FAILED TO CONNECT] {e}")

if __name__ == "__main__":
    test_huggingface_api()
