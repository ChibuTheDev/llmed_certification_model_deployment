import gradio as gr
import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ---- Hugging Face client ----
class HuggingFaceEndpoint:
    def __init__(self, endpoint_url: str, hf_token: str = None):
        self.endpoint_url = endpoint_url
        self.headers = {}
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"

    def summarize(self, dialogue: str) -> str:
        payload = {"inputs": dialogue}
        try:
            response = requests.post(self.endpoint_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
            else:
                return json.dumps(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def summarize_batch(self, dialogues):
        return [self.summarize(d) for d in dialogues]

# ---- Initialize client ----
ENDPOINT_URL = os.getenv("MODEL_URL") or os.getenv("HF_API_URL")
HF_TOKEN = None
client = HuggingFaceEndpoint(ENDPOINT_URL, HF_TOKEN)

# ---- Gradio functions ----
def summarize_single(dialogue: str):
    start = time.time()
    summary = client.summarize(dialogue)
    latency = round((time.time() - start) * 1000, 2)  # ms
    return summary, f"{latency} ms"

def summarize_batch(dialogues: str):
    # Expect each dialogue separated by two newlines
    dialogue_list = [d.strip() for d in dialogues.split("\n\n") if d.strip()]
    start = time.time()
    summaries = client.summarize_batch(dialogue_list)
    latency = round((time.time() - start) * 1000, 2)
    return "\n\n".join(summaries), f"{latency} ms"

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("## LLM Dialogue Summarizer")
    gr.Markdown("Enter a dialogue to summarize or multiple dialogues separated by double line breaks.")

    with gr.Tab("Single Dialogue"):
        single_input = gr.Textbox(label="Dialogue", placeholder="Enter dialogue here...", lines=7)
        single_output = gr.Textbox(label="Summary", lines=5)
        single_latency = gr.Textbox(label="Latency (ms)")
        single_button = gr.Button("Summarize")
        single_button.click(summarize_single, inputs=single_input, outputs=[single_output, single_latency])

    with gr.Tab("Batch Dialogues"):
        batch_input = gr.Textbox(label="Dialogues", placeholder="Separate multiple dialogues with double newlines", lines=10)
        batch_output = gr.Textbox(label="Summaries", lines=10)
        batch_latency = gr.Textbox(label="Latency (ms)")
        batch_button = gr.Button("Summarize Batch")
        batch_button.click(summarize_batch, inputs=batch_input, outputs=[batch_output, batch_latency])

# ---- Launch ----
if __name__ == "__main__":
    demo.launch()
