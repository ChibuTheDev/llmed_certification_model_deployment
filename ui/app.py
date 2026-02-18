import gradio as gr
import requests
import json
import time


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
            response = requests.post(
                self.endpoint_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
            else:
                return json.dumps(result)

        except requests.exceptions.RequestException as e:
            return f"Request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def summarize_batch(self, dialogues):
        # Sequential calls (same behavior as original)
        return [self.summarize(d) for d in dialogues]


# ---- Gradio functions ----
def summarize_single(endpoint_url, hf_token, dialogue):
    if not endpoint_url:
        return "Please provide a valid endpoint URL.", ""

    if not dialogue.strip():
        return "Please enter a dialogue to summarize.", ""

    client = HuggingFaceEndpoint(endpoint_url, hf_token)

    start = time.time()
    summary = client.summarize(dialogue)
    latency = round((time.time() - start) * 1000, 2)

    return summary, f"{latency} ms"


def summarize_batch(endpoint_url, hf_token, dialogues):
    if not endpoint_url:
        return "Please provide a valid endpoint URL.", ""

    dialogue_list = [d.strip() for d in dialogues.split("\n\n") if d.strip()]

    if not dialogue_list:
        return "Please enter at least one dialogue.", ""

    client = HuggingFaceEndpoint(endpoint_url, hf_token)

    start = time.time()
    summaries = client.summarize_batch(dialogue_list)
    latency = round((time.time() - start) * 1000, 2)

    return "\n\n".join(summaries), f"{latency} ms"


# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("## LLM Dialogue Summarizer")

    gr.Markdown("""
This application works only with the following model:

ReadyTensorCertification/bart-highlightsum-merged

You must:

1. Deploy a Hugging Face Inference Endpoint for this exact model.
2. Paste the generated Endpoint URL below.
3. Provide your Hugging Face access token.

""")

    endpoint_input = gr.Textbox(
        label="Deployed Inference Endpoint URL",
        placeholder="https://xxxxxx.region.aws.endpoints.huggingface.cloud"
    )

    token_input = gr.Textbox(
        label="Hugging Face Token",
        type="password"
    )

    with gr.Tab("Single Dialogue"):
        single_input = gr.Textbox(
            label="Dialogue",
            placeholder="Enter dialogue here...",
            lines=7
        )

        single_output = gr.Textbox(label="Summary", lines=5)
        single_latency = gr.Textbox(label="Latency (ms)")
        single_button = gr.Button("Summarize")

        single_button.click(
            summarize_single,
            inputs=[endpoint_input, token_input, single_input],
            outputs=[single_output, single_latency]
        )

    with gr.Tab("Batch Dialogues"):
        batch_input = gr.Textbox(
            label="Dialogues",
            placeholder="Separate multiple dialogues with double newlines",
            lines=10
        )

        batch_output = gr.Textbox(label="Summaries", lines=10)
        batch_latency = gr.Textbox(label="Latency (ms)")
        batch_button = gr.Button("Summarize Batch")

        batch_button.click(
            summarize_batch,
            inputs=[endpoint_input, token_input, batch_input],
            outputs=[batch_output, batch_latency]
        )


# ---- Launch ----
if __name__ == "__main__":
    demo.launch()
