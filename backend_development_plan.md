# SafeOps AI - Backend Development Plan & Integration Blueprint

This document outlines the detailed development plan, API routes, data pipelines, database schemas, simulators, and integration pathways required for the **Backend Developers** (Backend 1 and Backend 2 roles) to complete the SafeOps AI prototype.

The backend is built on **FastAPI**, storing relational operational data in **SQLite**, and integrates our pre-built **AI/ML LangGraph Master Flow** to calculate compound risk and generate safety compliance recommendations in real time.

---

## 1. Database Architecture (SQLite)

The SQLite database (`data/safeops.db`) stores the relational data for the plant layout, active workers, active work permits, raw sensor readings, generated alerts, and safety reports.

### 1.1 Schema Definitions
The backend team must create and initialize the following tables. Here are the SQL DDL statements:

```sql
-- 1. Plant Zones Table
CREATE TABLE IF NOT EXISTS zones (
    zone_id TEXT PRIMARY KEY,       -- e.g., 'ZONE_1', 'ZONE_3'
    name TEXT NOT NULL,             -- e.g., 'Coke Oven Battery'
    hazard_class TEXT NOT NULL,     -- e.g., 'A', 'B', 'SAFE'
    polygon_coords TEXT NOT NULL    -- JSON array of coordinates: '[{"x": 10, "y": 20}, ...]'
);

-- 2. Physical Sensors Table
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id TEXT PRIMARY KEY,     -- e.g., 'GAS_Z3_001'
    zone_id TEXT NOT NULL,          -- FK to zones
    type TEXT NOT NULL,             -- e.g., 'gas', 'temp', 'pressure'
    unit TEXT NOT NULL,             -- e.g., 'LEL%', 'C', 'bar'
    normal_min REAL DEFAULT 0.0,
    normal_max REAL NOT NULL,       -- Anomaly thresholds
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
);

-- 3. Plant Workers Table
CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,     -- e.g., 'W_101'
    name TEXT NOT NULL,
    shift TEXT NOT NULL,            -- e.g., 'MORNING', 'NIGHT'
    ppe_status TEXT NOT NULL,       -- e.g., 'COMPLIANT', 'VIOLATION'
    current_zone TEXT,              -- FK to zones (nullable if off-site)
    FOREIGN KEY (current_zone) REFERENCES zones(zone_id)
);

-- 4. Work Permits Table
CREATE TABLE IF NOT EXISTS permits (
    permit_id TEXT PRIMARY KEY,     -- e.g., 'PERMIT_HW_042'
    type TEXT NOT NULL,             -- e.g., 'HOT_WORK', 'CONFINED_SPACE', 'ELECTRICAL'
    zone_id TEXT NOT NULL,          -- FK to zones
    status TEXT NOT NULL,           -- e.g., 'APPROVED', 'ACTIVE', 'EXPIRED', 'SUSPENDED'
    start_time TEXT NOT NULL,       -- ISO8601 Timestamp
    expiry TEXT NOT NULL,           -- ISO8601 Timestamp
    workers_assigned TEXT NOT NULL, -- JSON array of worker IDs: '["W_101", "W_102"]'
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
);

-- 5. Time-Series Sensor Readings Table
CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_id TEXT NOT NULL,
    value REAL NOT NULL,
    anomaly_score REAL DEFAULT 0.0, -- Computed by LSTM agent
    ts TEXT NOT NULL,               -- ISO8601 Timestamp
    FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id)
);

-- 6. Generated Safety Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,      -- e.g., 'ALT_20260701_ZONE_3'
    zone_id TEXT NOT NULL,          -- FK to zones
    risk_score REAL NOT NULL,       -- Compound risk (0.0 to 1.0)
    severity TEXT NOT NULL,         -- e.g., 'NORMAL', 'LOW', 'MED', 'HIGH', 'CRITICAL'
    zones_affected TEXT NOT NULL,   -- JSON array of zone IDs
    trigger_summary TEXT NOT NULL,  -- Description of reasons
    status TEXT NOT NULL,           -- e.g., 'ACTIVE', 'ACKNOWLEDGED', 'RESOLVED'
    created_at TEXT NOT NULL,       -- ISO8601 Timestamp
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
);

-- 7. Incident Reports Table
CREATE TABLE IF NOT EXISTS reports (
    incident_id TEXT PRIMARY KEY,   -- FK to alerts or custom ID
    alert_id TEXT NOT NULL,         -- FK to alerts
    report_path TEXT NOT NULL,      -- Local file path to the generated PDF
    created_at TEXT NOT NULL,       -- ISO8601 Timestamp
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
);
```

---

## 2. Real-Time Background Simulators (Backend 1)

For the hackathon demonstration, we require background simulator tasks that periodically push fake data into our tables. This simulates a live, running industrial plant.

### 2.1 The Sensor Simulator Loop
Develop a continuous loop running in the background (using `asyncio.create_task` on startup). It should:
1. Fetch all sensors from the database.
2. Generate a new reading every 5 seconds.
3. Support three simulation modes (which can be toggled via REST APIs):
   * **`NORMAL`**: Gas level remains at $18.0 \pm 2\%$ LEL (random walk).
   * **`DRIFT`**: Gas level increases linearly by $+0.5\%$ LEL every 5 seconds towards the alarm threshold.
   * **`SPIKE`**: Gas level immediately jumps to $38.0\%$ LEL and stays there.
4. Write the reading into the `sensor_readings` table.

```python
# Example Simulator class structure
class SensorSimulator:
    def __init__(self, sensor_id: str, mode: str = "NORMAL"):
        self.sensor_id = sensor_id
        self.mode = mode
        self.current_val = 18.0
        
    def step(self) -> float:
        if self.mode == "NORMAL":
            self.current_val += random.gauss(0, 0.05) + (18.0 - self.current_val) * 0.01
        elif self.mode == "DRIFT":
            self.current_val += 0.5
        elif self.mode == "SPIKE":
            self.current_val = 38.0
        return max(0.0, self.current_val)
```

### 2.2 Worker & CCTV Simulator Loop
Simulates workers changing locations and PPE status:
1. Every 10 seconds, randomly update a worker's `current_zone` in the database.
2. In the demo scenario, inject **PPE Violations** (e.g., set `ppe_status` of `W_101` and `W_102` to `NO_HELMET` and `NO_VEST` inside `ZONE_3`).

---

## 3. Integrating the AI/ML LangGraph Engine

The backend must intercept every new sensor reading and evaluate the safety status using the AI/ML orchestrator.

### 3.1 Fetching Context and Invoking the Graph
When a new reading is stored (specifically for safety-critical zones like `ZONE_3`):
1. Query the last 12 readings for the zone's sensors from the database.
2. Query active permits in the zone from the `permits` table.
3. Query active workers and their PPE statuses in the zone from the `workers` table.
4. Invoke our pre-built LangGraph model (`src.orchestrator.master_flow.flow_app`) with the state dictionary.
5. Store the resulting `risk_fusion_out` and `rag_compliance_out` back into the `alerts` database.

Here is the exact integration code pattern the backend must use:

```python
import datetime
from src.orchestrator.master_flow import flow_app
from src.database.vector_db import VectorDB # If direct queries are needed

async def evaluate_safety_anomaly(zone_id: str):
    # 1. Fetch last 12 readings for [Gas, Temp, Pressure] from SQLite
    raw_history = await get_last_12_readings(zone_id) # Should return list of [gas, temp, pressure] lists
    
    # 2. Fetch active permits from SQLite
    active_permits = await get_active_permits(zone_id) # List of dicts matching schema
    
    # 3. Fetch CCTV / Worker frame data from SQLite
    cctv_frame = await get_latest_cctv_state(zone_id) # Dict with workers count and violations list
    
    # 4. Initialize State dictionary
    state = {
        "zone_id": zone_id,
        "current_time": datetime.datetime.utcnow().isoformat() + "Z",
        "shift_risk_factor": 0.1, # Base level. Can be elevated to 0.8 during shift handovers
        "sensor_raw_history": raw_history,
        "cv_raw_frame": cctv_frame,
        "active_permits_raw": active_permits
    }
    
    # 5. Invoke our LangGraph App (Thread-safe)
    result = flow_app.invoke(state)
    
    # 6. Parse results
    risk_out = result["risk_fusion_out"]
    rag_out = result.get("rag_compliance_out", None) # Present only if score >= 0.50
    action = result["action_taken"]
    
    # 7. Write alert to SQLite
    if risk_out["severity"] != "NORMAL":
        await save_alert_to_db(
            zone_id=zone_id,
            risk_score=risk_out["score"],
            severity=risk_out["severity"],
            action=action,
            summary=result.get("notifications_sent", []),
            rag_output=rag_out
        )
        
    # 8. Trigger notifications if needed
    if action in ["DISPATCH_ALERT", "TRIGGER_EVACUATION"]:
        trigger_external_alerting(action, risk_out, rag_out)
```

---

## 4. FastAPI Endpoints (REST APIs)

The backend team must expose the following endpoints for the React dashboard frontend.

### 4.1 GET `/health`
Returns service status.
* **Response**: `{"status": "healthy", "database": "connected"}`

### 4.2 GET `/api/zones`
Returns the status, details, and current compound risk score of all zones.
* **Response**:
  ```json
  [
    {
      "zone_id": "ZONE_3",
      "name": "Coke Oven Battery",
      "hazard_class": "A",
      "current_risk_score": 0.8717,
      "severity": "HIGH",
      "active_permits_count": 1,
      "workers_count": 3
    }
  ]
  ```

### 4.3 GET `/api/alerts`
Returns paginated list of generated safety alerts.
* **Query Parameters**: `limit` (default: 50), `zone_id` (optional).

### 4.4 GET `/api/permits/active`
Returns list of currently active work permits.
* **Query Parameters**: `zone_id` (optional).

### 4.5 POST `/api/scenario/trigger`
Triggers the hackathon demo scenarios. Sets the simulator modes to demonstrate system responses.
* **Request Body**:
  ```json
  {
    "scenario_type": "DRIFT" // Options: "NORMAL", "DRIFT", "SPIKE"
  }
  ```
* **Action**: Updates the background `SensorSimulator` mode and triggers mock worker/CCTV states to simulate the Vizag incident.

### 4.6 GET `/api/reports/{alert_id}`
Triggers the download of the auto-generated PDF report.
* **Action**: Fetches the report path from the `reports` table and returns the PDF file stream.

---

## 5. WebSockets Real-Time Pipelines

To support a dynamic UI that "feels alive", the backend must implement WebSockets instead of REST polling.

### 5.1 ws://localhost:8000/ws/sensors/{plant_id}
Broadcasts live sensor values and anomaly scores every 5 seconds to the dashboard charts.
* **Payload sent to client**:
  ```json
  {
    "type": "SENSOR_UPDATE",
    "timestamp": "2026-07-01T12:00:35Z",
    "zone_id": "ZONE_3",
    "sensor_id": "GAS_Z3_001",
    "sensor_type": "gas",
    "value": 25.0,
    "anomaly_score": 0.8717,
    "trend": "RISING"
  }
  ```

### 5.2 ws://localhost:8000/ws/alerts/{plant_id}
Broadcasts alert events immediately when the LangGraph orchestrator flags a threat.
* **Payload sent to client**:
  ```json
  {
    "type": "ALERT",
    "alert_id": "ALT_20260701_ZONE_3",
    "severity": "HIGH",
    "risk_score": 0.8717,
    "zone_id": "ZONE_3",
    "trigger_components": {
      "sensor": { "score": 0.87, "detail": "GAS_Z3_001 at 25% LEL, rising trend" },
      "cv": { "score": 0.67, "detail": "2 workers without helmets detected" },
      "permit": { "score": 1.0, "detail": "HOT_WORK permit active in gas leakage" }
    },
    "recommended_action": "DISPATCH_ALERT",
    "regulation_reference": "OISD-105 Clause 7.3.2"
  }
  ```

---

## 6. External Integrations (Services)

### 6.1 Twilio Alerting Engine (Backend 2)
When the action is `DISPATCH_ALERT` or `TRIGGER_EVACUATION`:
1. Format a message body summarizing the violations and citations:
   ```
   🚨 CRITICAL ALERT - SafeOps AI 🚨
   Evacuation triggered in ZONE_3 (Coke Oven Battery).
   Compound Risk Score: 1.0.
   Gas Concentration: 38% LEL.
   Active Violations: OISD-105 Clause 7.3.2 (Hot Work near gas leakage).
   Action Required: Evacuate immediately!
   ```
2. Send the message via Twilio's SMS and WhatsApp API channels to the configured phone numbers of the safety officers.

### 6.2 ReportLab PDF Generator (Backend 2)
Implement a script utilizing `reportlab` to write a formal compliance audit PDF report.
* **File Naming**: Save under `data/reports/incident_ALT_{alert_id}.pdf`.
* **Required Content Section Structure**:
  1. **Header Block**: Logo, Title, Timestamp, Alert Severity, and Zone Name.
  2. **Risk Breakdown Table**: Prints Sensor anomaly, CV safety counts, and active permit codes.
  3. **Evidence Snapshots**: Sensor history logs and mock CCTV frame snapshot file path references.
  4. **Compliance Citations**: Copy the RAG output containing OISD Standard 105 and Factories Act clauses.
  5. **Audit Sign-off**: Signature lines for the safety head and inspector.

---

## 7. Containerization & Deployment

Set up a `Dockerfile` and `docker-compose.yml` to package the FastAPI backend.

### 7.1 Dockerfile Example
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run FastAPI using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2 docker-compose.yml Example
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=AIzaSyBtWTydR3sspsdr2iG3Kb8oJBqejWGmEx4
      - GEMINI_MODEL=gemini-2.5-flash
    volumes:
      - ./data:/app/data
    restart: always
```
