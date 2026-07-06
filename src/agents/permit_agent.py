import datetime
from typing import List, Dict

class PermitAgent:
    def __init__(self):
        pass
        
    def run(self, active_permits: List[Dict], current_time: str = None, current_gas: float = 0.0) -> dict:
        """Analyzes active permits to detect safety conflicts (overlaps, expired permits, high-risk SIMOPS).
        
        Args:
            active_permits (list): List of active permit dicts:
                - permit_id (str)
                - type (str): 'HOT_WORK', 'CONFINED_SPACE', 'ELECTRICAL', 'HEIGHT'
                - zone_id (str)
                - start_time (str): ISO8601 string
                - expiry (str): ISO8601 string
                - workers_assigned (list)
                - hazard_class (str)
            current_time (str): Optional override for checking expiry. If None, uses utcnow.
            current_gas (float): Current gas sensor reading in LEL%.
            
        Returns:
            dict conforming to the frozen output schema for 'permit_intel'.
        """
        if current_time is None:
            now = datetime.datetime.utcnow()
        else:
            try:
                now = datetime.datetime.fromisoformat(current_time.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                now = datetime.datetime.utcnow()
                
        conflicts = []
        
        # 1. Check for EXPIRED_ACTIVE permits
        for p in active_permits:
            expiry_str = p.get("expiry", "")
            if expiry_str:
                try:
                    expiry_dt = datetime.datetime.fromisoformat(expiry_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    if now > expiry_dt:
                        conflicts.append({
                            "conflict_type": "EXPIRED_ACTIVE",
                            "permit_ids": [p["permit_id"]],
                            "zones_affected": [p["zone_id"]],
                            "risk_description": f"Permit {p['permit_id']} ({p['type']}) is active past its expiry time ({expiry_str}).",
                            "severity": "MED"
                        })
                except Exception as e:
                    print(f"Error parsing permit expiry time: {e}")

        # 1.5 Check for OISD-105 Section 4.3 violation: Hot Work in presence of elevated gas
        for p in active_permits:
            if p["type"] == "HOT_WORK" and current_gas >= 20.0:
                conflicts.append({
                    "conflict_type": "SIMULTANEOUS_HIGH_RISK",
                    "permit_ids": [p["permit_id"]],
                    "zones_affected": [p["zone_id"]],
                    "risk_description": f"OISD-105 Clause 4.3 violation: HOT_WORK permit active in {p['zone_id']} with elevated gas concentration ({current_gas}% LEL).",
                    "severity": "CRITICAL"
                })

        # 2. Check for OVERLAP of high-risk permits in the same zone (SIMOPS conflicts)
        # Group permits by zone
        zone_permits = {}
        for p in active_permits:
            zone = p["zone_id"]
            if zone not in zone_permits:
                zone_permits[zone] = []
            zone_permits[zone].append(p)
            
        for zone, p_list in zone_permits.items():
            if len(p_list) >= 2:
                # We have multiple permits in the same zone. Check for dangerous combinations.
                types = [p["type"] for p in p_list]
                ids = [p["permit_id"] for p in p_list]
                
                # Check for HOT_WORK and CONFINED_SPACE overlap in same zone
                if "HOT_WORK" in types and "CONFINED_SPACE" in types:
                    conflicts.append({
                        "conflict_type": "SIMULTANEOUS_HIGH_RISK",
                        "permit_ids": ids,
                        "zones_affected": [zone],
                        "risk_description": f"Critical overlap: HOT_WORK and CONFINED_SPACE permits are active simultaneously in {zone}. High risk of flammable vapor ignition.",
                        "severity": "CRITICAL"
                    })
                # Check for HOT_WORK and HEIGHT (requiring safety harness) or ELECTRICAL overlap
                elif "HOT_WORK" in types and "ELECTRICAL" in types:
                    conflicts.append({
                        "conflict_type": "OVERLAP",
                        "permit_ids": ids,
                        "zones_affected": [zone],
                        "risk_description": f"Operational conflict: HOT_WORK and ELECTRICAL permits are active in {zone}.",
                        "severity": "HIGH"
                    })
                else:
                    # Generic overlap of multiple permits
                    conflicts.append({
                        "conflict_type": "OVERLAP",
                        "permit_ids": ids,
                        "zones_affected": [zone],
                        "risk_description": f"SIMOPS: Multiple active permits ({', '.join(types)}) overlapping in {zone}.",
                        "severity": "MED"
                    })
                    
        return {
            "agent": "permit_intel",
            "active_permits": active_permits,
            "conflicts": conflicts,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        
    def calculate_conflict_score(self, agent_output: dict) -> float:
        """Helper to calculate the conflict score (0.0 to 1.0) based on detected conflicts."""
        conflicts = agent_output.get("conflicts", [])
        if not conflicts:
            return 0.0
            
        severities = [c.get("severity", "LOW") for c in conflicts]
        
        if "CRITICAL" in severities:
            return 1.0
        if "HIGH" in severities:
            return 0.8
        if "MED" in severities:
            return 0.5
        return 0.3
