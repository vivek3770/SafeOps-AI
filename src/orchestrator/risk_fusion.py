from src.config import Config
from src.graph.plant_graph import PlantGraph

class RiskFusionEngine:
    def __init__(self, plant_graph: PlantGraph):
        self.graph = plant_graph
        self.weights = Config.RISK_WEIGHTS
        self.thresholds = Config.SEVERITY_THRESHOLDS
        
    def compute_score(self, 
                      sensor_anomaly_score: float, 
                      cv_violations: list, 
                      permit_conflicts: list, 
                      shift_risk_factor: float, 
                      zone_id: str) -> dict:
        """Computes the compound risk score and severity level based on inputs and KG context.
        
        Formula:
            CompoundRiskScore = (
                0.35 * sensor_anomaly_score
                + 0.28 * cv_violation_score
                + 0.22 * permit_conflict_score
                + 0.15 * shift_risk_factor
            )
            + 0.1 (if safety critical equipment in zone is active/critical)
            
        Args:
            sensor_anomaly_score (float): Anomaly score from LSTM (0.0 to 1.0)
            cv_violations (list): List of safety violations from CCTV
            permit_conflicts (list): List of permit conflict dicts
            shift_risk_factor (float): Time-based human error factor (0.0 to 1.0)
            zone_id (str): Target zone ID (e.g. 'ZONE_3')
            
        Returns:
            dict containing:
                - score (float)
                - severity (str): 'NORMAL', 'LOW', 'MED', 'HIGH', 'CRITICAL'
                - breakdown (dict)
        """
        # 1. Normalize sensor score
        s = max(0.0, min(1.0, sensor_anomaly_score))
        
        # 2. Normalize CV violations: 3+ violations = max (1.0)
        c = min(len(cv_violations) / 3.0, 1.0)
        
        # 3. Normalize Permit conflicts: Max severity among conflicts maps to score
        p = 0.0
        if permit_conflicts:
            severities = [conf.get("severity", "LOW") for conf in permit_conflicts]
            if "CRITICAL" in severities:
                p = 1.0
            elif "HIGH" in severities:
                p = 0.8
            elif "MED" in severities:
                p = 0.5
            else:
                p = 0.3
                
        # 4. Normalize Shift risk
        sh = max(0.0, min(1.0, shift_risk_factor))
        
        # Calculate base compound score
        compound = (
            self.weights['sensor'] * s
            + self.weights['cv'] * c
            + self.weights['permit'] * p
            + self.weights['shift'] * sh
        )
        
        # 5. Apply Knowledge Graph Amplifier
        # If there is critical equipment in the zone, add +0.1
        kg_amplified = False
        if self.graph.is_equipment_critical(zone_id):
            compound += 0.1
            kg_amplified = True
            
        # Clamp score between 0.0 and 1.0
        compound = max(0.0, min(1.0, compound))
        
        # 6. Determine Severity Level
        severity = "NORMAL"
        # Sort thresholds descending to match highest first
        for name, thresh in sorted(self.thresholds.items(), key=lambda item: item[1], reverse=True):
            if compound >= thresh:
                severity = name
                break
                
        return {
            "score": round(compound, 4),
            "severity": severity,
            "kg_amplified": kg_amplified,
            "breakdown": {
                "sensor_score": round(s, 4),
                "cv_score": round(c, 4),
                "permit_score": round(p, 4),
                "shift_score": round(sh, 4)
            }
        }
