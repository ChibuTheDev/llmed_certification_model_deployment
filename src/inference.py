import requests
import json
from typing import List
import os 
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceEndpoint:
    def __init__(self, endpoint_url: str, hf_token: str = None):
        """
        Initialize the Hugging Face Inference Endpoint client.

        :param endpoint_url: my deployed endpoint URL
        :param hf_token: Hugging face token   """
        self.endpoint_url = endpoint_url
        self.headers = {}
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"

    def summarize(self, dialogue: str) -> str:
        """
        Summarize a single dialogue.

        :param dialogue: Multi-turn dialogue string
        :return: Summary string
        """
        payload = {"inputs": dialogue}
        response = requests.post(self.endpoint_url, headers=self.headers, json=payload)
        response.raise_for_status() 

        result = response.json()
        if isinstance(result, list) and "summary_text" in result[0]:
            return result[0]["summary_text"]
        else:
            return json.dumps(result) 

    def summarize_batch(self, dialogues: List[str]) -> List[str]:
        """
        Summarize a list of dialogues.

        :param dialogues: List of dialogue strings
        :return: List of summary strings
        """
        summaries = []
        for dialogue in dialogues:
            try:
                summary = self.summarize(dialogue)
            except Exception as e:
                summary = f"Error: {str(e)}"
            summaries.append(summary)
        return summaries


# Example usage
if __name__ == "__main__":
    ENDPOINT_URL = os.getenv('MODEL_URL')
    HF_TOKEN = os.getenv("HF_TOKEN")

    client = HuggingFaceEndpoint(ENDPOINT_URL, HF_TOKEN)

    # Single dialogue
    dialogue = """A: Hi Tom, are you busy tomorrow afternoon?
B: I think I am. Why?
A: I want to go to the animal shelter.
B: For what?
A: I'm getting a puppy for my son."""
    summary = client.summarize(dialogue)
    print("Single Dialogue Summary:\n", summary)

    # Batch dialogues
    batch = [
        dialogue,
        "A: What are you getting him? B: Something cool. A: What about a Lego? B: Too old for that. A: Then what? B: Not sure."
    ]
    summaries = client.summarize_batch(batch)
    print("\nBatch Summaries:")
    for i, s in enumerate(summaries, 1):
        print(f"{i}. {s}")
