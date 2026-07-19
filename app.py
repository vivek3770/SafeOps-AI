import gradio as gr
from src.api_server import app as fastapi_app

# Simple Gradio UI for browser testing & verification
with gr.Blocks(title="SafeOps AI Agent Engine") as demo:
    gr.Markdown("# 🛡️ SafeOps AI Multi-Agent Engine")
    gr.Markdown(
        "This Hugging Face Space hosts the **LangGraph Multi-Agent Orchestrator** & REST API microservice.\n\n"
        "- **API Health Endpoint**: `/health`\n"
        "- **Master Evaluation API**: `/api/eval`\n"
        "- **Swagger Interactive Docs**: `/docs`\n"
    )
    
    with gr.Row():
        zone_input = gr.Textbox(value="ZONE_3", label="Zone ID")
        status_output = gr.Textbox(label="Agent Engine Status", value="Ready to accept API requests from Backend.")

# Mount FastAPI app (with all REST endpoints) onto the Gradio application
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
