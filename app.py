import gradio as gr
import json
import datetime
import os
import spaces
from src.orchestrator.master_flow import flow_app

@spaces.GPU
def dummy_gpu_trigger():
    """Dummy function to satisfy ZeroGPU requirement without blocking evaluate_safety_api."""
    return "ZeroGPU Active"

def evaluate_safety_api(payload_str: str):
    """Gradio API wrapper for the master multi-agent flow."""
    try:
        state = json.loads(payload_str)
        if not state.get("current_time"):
            state["current_time"] = datetime.datetime.utcnow().isoformat() + "Z"
        result = flow_app.invoke(state)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

# Simple Gradio UI for browser testing & verification
with gr.Blocks(title="SafeOps AI Agent Engine") as demo:
    gr.Markdown("# 🛡️ SafeOps AI Multi-Agent Engine")
    gr.Markdown(
        "This Hugging Face Space hosts the **LangGraph Multi-Agent Orchestrator**.\n\n"
        "### 🔌 Integration\n"
        "The backend can call the API using the standard Gradio client or standard POST requests to `/call/eval`.\n"
        "Payload format: `{\"data\": [\"{\\\"zone_id\\\": \\\"...\\\"}\"]}`"
    )
    
    with gr.Row():
        zone_input = gr.Textbox(value="ZONE_3", label="Zone ID")
        status_output = gr.Textbox(label="Agent Engine Status", value="Ready to accept API requests from Backend.")
        
    # Hidden components for the API endpoint
    api_payload = gr.Textbox(visible=False, label="API Payload")
    api_result = gr.Textbox(visible=False, label="API Result")
    api_btn = gr.Button("Run API", visible=False)
    api_btn.click(
        fn=evaluate_safety_api, 
        inputs=[api_payload], 
        outputs=[api_result], 
        api_name="eval"
    )

if __name__ == "__main__":
    demo.launch()
