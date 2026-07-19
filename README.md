---
title: SafeOps AI Engine
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ SafeOps AI - Multi-Agent Industrial Safety Orchestrator

SafeOps AI is a multi-agent safety compliance and compound risk evaluation platform for industrial plant environments.

## API Endpoints
- **Master Evaluation API (Gradio)**: `POST /gradio_api/call/eval`
- **Gradio App Info**: `GET /info`
- **Payload Format**: `{"data": ["{JSON_STRING}"]}`

## Agents Orchestrated
1. **SensorAgent**: PyTorch LSTM Multivariate Anomaly Detection
2. **CVAgent**: YOLOv8 Vision & PPE Compliance Wrapper
3. **PermitAgent**: NetworkX Spatial Graph & Permit Conflict Intelligence
4. **RiskFusionEngine**: Dynamic Compound Risk Calculation Engine
5. **RAGAgent**: ChromaDB Vector Store & Gemini 2.5 Flash Regulatory Compliance Agent
