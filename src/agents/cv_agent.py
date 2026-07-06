import datetime
from typing import List, Dict

class CVAgent:
    def __init__(self):
        pass
        
    def run(self, raw_cctv_frame_data: dict) -> dict:
        """Parses raw CCTV detection frames from YOLOv8 (processed by Backend 1)
        and formats them into the frozen schema for 'cv_safety'.
        
        Args:
            raw_cctv_frame_data (dict): Contains:
                - zone_id (str)
                - workers_detected (int)
                - violations (list)
                
        Returns:
            dict conforming to the frozen output schema for 'cv_safety'.
        """
        zone_id = raw_cctv_frame_data.get("zone_id", "ZONE_3")
        workers_detected = raw_cctv_frame_data.get("workers_detected", 0)
        violations = raw_cctv_frame_data.get("violations", [])
        
        formatted_violations = []
        for v in violations:
            formatted_violations.append({
                "worker_id": v.get("worker_id", "WORKER_UNKNOWN"),
                "violation_type": v.get("violation_type", "NO_HELMET"),
                "confidence": float(v.get("confidence", 0.85)),
                "bbox": v.get("bbox", [100, 150, 200, 300]),
                "frame_snapshot_path": v.get("frame_snapshot_path", "/snapshots/latest_violation.jpg")
            })
            
        now_str = datetime.datetime.utcnow().isoformat() + "Z"
        
        return {
            "agent": "cv_safety",
            "zone_id": zone_id,
            "frame_timestamp": now_str,
            "workers_detected": int(workers_detected),
            "violations": formatted_violations,
            "headcount_in_hazard_zone": int(workers_detected),
            "timestamp": now_str
        }

    def generate_mock_violations(self, count: int, zone_id: str = "ZONE_3") -> dict:
        """Helper to generate simulated violations for the Vizag demo scenario.
        Creates 'NO_HELMET' or 'NO_VEST' violations.
        """
        violations = []
        for i in range(count):
            violations.append({
                "worker_id": f"W_{100 + i}",
                "violation_type": "NO_HELMET" if i % 2 == 0 else "NO_VEST",
                "confidence": 0.89 - (i * 0.05),
                "bbox": [50 + i*40, 120, 120 + i*40, 280],
                "frame_snapshot_path": f"/snapshots/violation_zone3_{i}.jpg"
            })
            
        return self.run({
            "zone_id": zone_id,
            "workers_detected": count + 1,
            "violations": violations
        })
