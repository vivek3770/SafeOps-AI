import os
import sys
import time
import json
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.orchestrator.master_flow import flow_app

# Define terminal colors for beautiful reporting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"{Colors.HEADER}{Colors.BOLD}====================================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}                    SAFEOPS AI - DEMO SCENARIO REPLAY                  {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}               Vizag Coke Oven Battery Explosion Prevention            {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}====================================================================={Colors.ENDC}\n")

def run_step(step_num: int, duration_sec: str, description: str, state: dict):
    print(f"\n{Colors.BOLD}[T+{duration_sec}] STEP {step_num}: {description}{Colors.ENDC}")
    print("-" * 50)
    
    # Run the LangGraph flow
    start_time = time.time()
    result = flow_app.invoke(state)
    latency = (time.time() - start_time) * 1000
    
    # Extract outcomes
    risk_out = result["risk_fusion_out"]
    score = risk_out["score"]
    severity = risk_out["severity"]
    action = result["action_taken"]
    
    # Format severity with color
    severity_color = Colors.GREEN
    if severity == "MED":
        severity_color = Colors.WARNING
    elif severity == "HIGH":
        severity_color = Colors.FAIL
    elif severity == "CRITICAL":
        severity_color = f"{Colors.FAIL}{Colors.BOLD}"
        
    print(f"Current Gas Level   : {state['sensor_raw_history'][-1][0]}% LEL")
    print(f"Active Permits      : {[p['type'] for p in state['active_permits_raw']]}")
    print(f"PPE Violations      : {len(state['cv_raw_frame']['violations'])}")
    print(f"Compound Risk Score : {Colors.BOLD}{score}{Colors.ENDC} ({severity_color}{severity}{Colors.ENDC})")
    print(f"Orchestrator Action : {Colors.BLUE}{action}{Colors.ENDC}")
    print(f"Decision Latency    : {latency:.2f} ms")
    
    if "rag_compliance_out" in result and result["rag_compliance_out"]:
        rag_out = result["rag_compliance_out"]
        print(f"\n{Colors.CYAN}--- RAG Compliance & Incident Safety Copilot Report ---{Colors.ENDC}")
        
        # Print Regulations
        print(f"\n{Colors.BOLD}Applicable Regulations Violations:{Colors.ENDC}")
        for reg in rag_out.get("applicable_regulations", []):
            violation_status = f"{Colors.FAIL}VIOLATION DETECTED{Colors.ENDC}" if reg.get("violation_detected") else f"{Colors.GREEN}COMPLIANT{Colors.ENDC}"
            print(f"  - [{reg.get('regulation_id')}] {reg.get('title')} Clause {reg.get('clause')}: {reg.get('requirement')} ({violation_status})")
            
        # Print Precedents
        print(f"\n{Colors.BOLD}Similar Historical Precedents:{Colors.ENDC}")
        for inc in rag_out.get("similar_incidents", []):
            print(f"  - {Colors.WARNING}{inc.get('plant')} ({inc.get('date')}){Colors.ENDC}: {inc.get('description')}")
            print(f"    Outcome: {inc.get('outcome')}")
            
        # Print Recommendations
        print(f"\n{Colors.BOLD}Actionable Mitigation Recommendations:{Colors.ENDC}")
        for rec in rag_out.get("recommended_actions", []):
            print(f"  * {Colors.HEADER}{rec}{Colors.ENDC}")
            
        print(f"\nSources Cited: {rag_out.get('rag_sources_cited', [])}")

def main():
    print_banner()
    
    # 1. Normal state baseline at T+0:00
    state_t0 = {
        "zone_id": "ZONE_3",
        "current_time": "2026-07-01T12:00:00Z",
        "shift_risk_factor": 0.1,
        # Gas stable at 18%, normal baseline
        "sensor_raw_history": [[18.0, 28.0, 1.2] for _ in range(12)],
        "cv_raw_frame": {"zone_id": "ZONE_3", "workers_detected": 2, "violations": []},
        "active_permits_raw": []
    }
    run_step(1, "0:00", "Baseline plant operations are stable", state_t0)
    time.sleep(1)
    
    # 2. Hot work permit activated at T+0:10
    state_t10 = {
        "zone_id": "ZONE_3",
        "current_time": "2026-07-01T12:00:10Z",
        "shift_risk_factor": 0.1,
        "sensor_raw_history": [[18.0, 28.0, 1.2] for _ in range(12)],
        "cv_raw_frame": {"zone_id": "ZONE_3", "workers_detected": 2, "violations": []},
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
    run_step(2, "0:10", "Hot Work Permit #HW-042 is issued in Zone 3 (Coke Oven Battery)", state_t10)
    time.sleep(1)

    # 3. PPE violations detected at T+0:20
    state_t20 = {
        "zone_id": "ZONE_3",
        "current_time": "2026-07-01T12:00:20Z",
        "shift_risk_factor": 0.1,
        "sensor_raw_history": [[18.0, 28.0, 1.2] for _ in range(12)],
        "cv_raw_frame": {
            "zone_id": "ZONE_3",
            "workers_detected": 3,
            "violations": [
                {"worker_id": "W_101", "violation_type": "NO_HELMET", "confidence": 0.9},
                {"worker_id": "W_102", "violation_type": "NO_VEST", "confidence": 0.85}
            ]
        },
        "active_permits_raw": state_t10["active_permits_raw"]
    }
    run_step(3, "0:20", "CCTV detects 2 maintenance workers without helmets or vests in Zone 3", state_t20)
    time.sleep(1)

    # 4. Gas levels start to leak/drift at T+0:35
    state_t35 = {
        "zone_id": "ZONE_3",
        "current_time": "2026-07-01T12:00:35Z",
        "shift_risk_factor": 0.1,
        # Gas level drifts from 18% up to 25% LEL
        "sensor_raw_history": [[18.0, 28.0, 1.2] for _ in range(9)] + [[21.0, 28.5, 1.2], [23.5, 29.0, 1.21], [25.0, 29.2, 1.21]],
        "cv_raw_frame": state_t20["cv_raw_frame"],
        "active_permits_raw": state_t10["active_permits_raw"]
    }
    run_step(4, "0:35", "Gas sensor GAS_Z3_001 starts rising, reaching 25.0% LEL", state_t35)
    time.sleep(1)

    # 5. Critical gas levels reach 38% LEL at T+1:00
    state_t60 = {
        "zone_id": "ZONE_3",
        "current_time": "2026-07-01T12:01:00Z",
        "shift_risk_factor": 0.8,  # Shift changeover starts, adding human error risk
        # Gas spikes to 38% LEL, temperature and pressure rise
        "sensor_raw_history": [[18.0, 28.0, 1.2] for _ in range(6)] + [
            [21.0, 28.5, 1.2], [25.0, 29.2, 1.21], [29.0, 29.8, 1.22],
            [32.0, 30.2, 1.23], [35.0, 30.6, 1.23], [38.0, 31.0, 1.24]
        ],
        "cv_raw_frame": {
            "zone_id": "ZONE_3",
            "workers_detected": 3,
            # Add third violation to simulate high stress/critical situation
            "violations": state_t20["cv_raw_frame"]["violations"] + [
                {"worker_id": "W_103", "violation_type": "NO_HARNESS", "confidence": 0.8}
            ]
        },
        "active_permits_raw": state_t10["active_permits_raw"]
    }
    run_step(5, "1:00", "CRITICAL THREAT: Gas levels spike to 38% LEL during active Hot Work permit", state_t60)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}====================================================================={Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}                      DEMO RUN SUCCESSFULLY COMPLETED                 {Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}====================================================================={Colors.ENDC}\n")

if __name__ == "__main__":
    main()
