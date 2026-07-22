import json
import requests
import time

def test_huggingface_api():
    print("=========================================================")
    print("      TESTING LIVE HUGGING FACE AI ENGINE API            ")
    print("=========================================================")
    
    url = "https://monkey3770-safeops-ai-engine.hf.space/gradio_api/call/eval"
    
    mock_state = {
        "zone_id": "ZONE_3",
        "current_time": "2026-07-01T12:00:20Z",
        "shift_risk_factor": 0.1,
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
    
    payload = {
        "data": [json.dumps(mock_state)]
    }
    
    print(f"Sending POST request to: {url}")
    print("Waiting for AI Engine to calculate risk (might take 15-30s if Hugging Face is waking up)...")
    
    try:
        response = requests.post(url, json=payload, timeout=60.0)
            
        if response.status_code == 200:
            print("\n[SUCCESS] Connected to Hugging Face successfully.")
            response_json = response.json()
            
            if "event_id" in response_json:
                event_id = response_json["event_id"]
                poll_url = f"{url}/{event_id}"
                print(f"Gradio Event ID: {event_id}")
                print(f"Streaming results from: {poll_url}...")
                
                # Stream the Server-Sent Events (SSE)
                with requests.get(poll_url, stream=True, timeout=60.0) as r:
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_content = decoded_line[6:]
                                if data_content != "null":
                                    try:
                                        data_json = json.loads(data_content)
                                        final_ai_results = json.loads(data_json[0])
                                        print("\n--- PARSED AI RESULTS ---")
                                        print(f"Risk Score: {final_ai_results.get('risk_fusion_out', {}).get('score')}")
                                        print(f"Severity: {final_ai_results.get('risk_fusion_out', {}).get('severity')}")
                                        print(f"Action: {final_ai_results.get('action_taken')}")
                                        
                                        if "rag_compliance_out" in final_ai_results:
                                            rag_out = final_ai_results["rag_compliance_out"]
                                            print("\n--- RAG COMPLIANCE & SAFETY COPILOT REPORT ---")
                                            
                                            print("\nApplicable Regulations:")
                                            for reg in rag_out.get("applicable_regulations", []):
                                                print(f"  - [{reg.get('regulation_id')}] {reg.get('title')}: {reg.get('requirement')}")
                                                
                                            print("\nHistorical Precedents:")
                                            for inc in rag_out.get("similar_incidents", []):
                                                print(f"  - {inc.get('plant')} ({inc.get('date')}): {inc.get('description')}")
                                                print(f"    Outcome: {inc.get('outcome')}")
                                                
                                            print("\nActionable Recommendations:")
                                            for rec in rag_out.get("recommended_actions", []):
                                                print(f"  * {rec}")
                                            
                                            print(f"\nSources Cited: {rag_out.get('rag_sources_cited', [])}")
                                        return
                                    except Exception as e:
                                        # Skip lines that are not valid JSON arrays
                                        pass
                            elif decoded_line.startswith("event: error"):
                                print("\n[ERROR] Hugging Face execution encountered a container/ZeroGPU error.")
                                return
            else:
                print(f"\n[ERROR] Unexpected response format: {response_json}")
        else:
            print(f"\n[ERROR] Received status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n[FAILED TO CONNECT] {e}")

if __name__ == "__main__":
    test_huggingface_api()
