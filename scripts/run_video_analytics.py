import os
import sys
import time
import json
import datetime
import random
import requests
from pathlib import Path

try:
    import cv2
    from ultralytics import YOLO
except ImportError:
    print("Please install requirements first: pip install ultralytics opencv-python")
    sys.exit(1)

# Add project root to python path for imports if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

def main():
    print("=========================================================")
    print("     SAFEOPS AI - LIVE VIDEO ANALYTICS (YOLOv8)          ")
    print("=========================================================")
    print("Loading YOLOv8 model...")
    
    # Use the base yolov8n.pt model. 
    # For a real PPE detector, replace this with your custom 'best.pt' weights path!
    model_path = "yolov8n.pt" 
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model {model_path}. Error: {e}")
        sys.exit(1)

    # Initialize video capture. 0 for webcam, or provide a path to an .mp4 file.
    video_source = 0 
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        sys.exit(1)

    print(f"Video source '{video_source}' opened successfully. Press 'q' to quit.")
    
    zone_id = "ZONE_3" # Hardcoded for demo scenario
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to grab frame.")
            break
            
        frame_count += 1
        
        # Run YOLO inference
        # Conf=0.5 to reduce false positives
        results = model(frame, conf=0.5, verbose=False)
        
        workers_detected = 0
        violations = []
        
        # Parse results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # In base YOLOv8, class 0 is 'person'
                # If using a custom PPE model, check for your specific class IDs (e.g. 1 for 'no_helmet')
                if cls_id == 0:
                    workers_detected += 1
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # --- MOCK PPE VIOLATION LOGIC FOR DEMO ---
                    # Since base YOLO only detects people, we simulate a PPE violation randomly
                    # or based on position, just to trigger the compound risk engine!
                    if random.random() < 0.3: # 30% chance a worker is flagged without PPE
                        violation_type = random.choice(["NO_HELMET", "NO_VEST"])
                        violations.append({
                            "worker_id": f"W_VID_{workers_detected}",
                            "violation_type": violation_type,
                            "confidence": round(conf, 2),
                            "bbox": [x1, y1, x2, y2],
                            "frame_snapshot_path": f"/snapshots/live_{int(time.time())}.jpg"
                        })
                        
                        # Draw RED bounding box for violation
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{violation_type} ({conf:.2f})", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # Draw GREEN bounding box for compliant worker
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Worker ({conf:.2f})", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Build the final JSON Payload
        cv_payload = {
            "zone_id": zone_id,
            "workers_detected": workers_detected,
            "violations": violations
        }
        
        # Every 30 frames (approx 1 second), send the payload to the backend
        if frame_count % 30 == 0:
            print("\n[LIVE CV PAYLOAD EXTRACTED]")
            print(json.dumps(cv_payload, indent=2))
            
            # Send the JSON payload over HTTP to your local Java/Python Backend
            backend_url = "http://13.204.35.23:8080/api/cv/frame"
            try:
                # We use a short timeout so the video doesn't freeze if the backend is down
                response = requests.post(backend_url, json=cv_payload, timeout=2.0)
                print(f"-> Successfully sent to Backend! (Status: {response.status_code})")
            except requests.exceptions.RequestException as e:
                print(f"-> Warning: Could not connect to Backend at {backend_url}. Is it running?")
        
        # Show the video feed
        cv2.imshow("SafeOps AI - Live Video Analytics", frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
