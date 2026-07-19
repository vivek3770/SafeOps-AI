import os
import datetime
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.orchestrator.master_flow import (
    flow_app,
    sensor_agent,
    cv_agent,
    permit_agent,
    rag_agent,
)

app = FastAPI(
    title="SafeOps AI Engine API",
    description="Hugging Face microservice hosting LangGraph multi-agent safety orchestrator",
    version="1.0.0",
)

# Enable CORS for backend & frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas for API Requests
class SafetyEvalRequest(BaseModel):
    zone_id: str
    current_time: Optional[str] = None
    shift_risk_factor: Optional[float] = 0.10
    sensor_raw_history: List[Any]
    cv_raw_frame: Dict[str, Any]
    active_permits_raw: List[Dict[str, Any]]


class SensorOnlyRequest(BaseModel):
    sensor_raw_history: List[Any]


class CVOnlyRequest(BaseModel):
    zone_id: str
    workers_detected: int
    violations: List[Dict[str, Any]]


class PermitOnlyRequest(BaseModel):
    permits: List[Dict[str, Any]]
    current_time: Optional[str] = None
    current_gas: Optional[float] = 0.0


class RAGOnlyRequest(BaseModel):
    zone_id: str
    sensor_reading: float
    cv_violations: List[Any]
    active_permits: List[Any]


@app.get("/")
def read_root():
    return {
        "service": "SafeOps AI Agent Engine",
        "status": "RUNNING",
        "platform": "Hugging Face Spaces",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "agents": ["SensorAgent", "CVAgent", "PermitAgent", "RiskFusionEngine", "RAGAgent"],
    }


# -------------------------------------------------------------
# MASTER MULTI-AGENT EVALUATION ENDPOINT
# -------------------------------------------------------------
@app.post("/api/eval")
async def evaluate_safety(payload: SafetyEvalRequest):
    """Executes the complete LangGraph multi-agent flow."""
    try:
        state = payload.dict()
        if not state.get("current_time"):
            state["current_time"] = datetime.datetime.utcnow().isoformat() + "Z"

        result = flow_app.invoke(state)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent orchestration failed: {str(e)}")


# -------------------------------------------------------------
# STANDALONE SINGLE AGENT ENDPOINTS
# -------------------------------------------------------------
@app.post("/api/agents/sensor")
async def run_sensor_agent(payload: SensorOnlyRequest):
    """Runs only the Sensor LSTM Anomaly Detection agent."""
    try:
        return sensor_agent.run(payload.sensor_raw_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/cv")
async def run_cv_agent(payload: CVOnlyRequest):
    """Runs only the CV Safety Agent."""
    try:
        return cv_agent.run(payload.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/permit")
async def run_permit_agent(payload: PermitOnlyRequest):
    """Runs only the Permit Conflict & Spatial Graph agent."""
    try:
        now_str = payload.current_time or (datetime.datetime.utcnow().isoformat() + "Z")
        return permit_agent.run(payload.permits, current_time=now_str, current_gas=payload.current_gas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/rag")
async def run_rag_agent(payload: RAGOnlyRequest):
    """Runs only the RAG Regulatory Compliance agent."""
    try:
        return rag_agent.run(payload.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=port, reload=False)
