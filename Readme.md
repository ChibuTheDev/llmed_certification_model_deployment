# Module 2 Capstone Project — LLM Deployment & Monitoring

## Overview

This project demonstrates the end-to-end deployment of a fine-tuned Large Language Model from experimentation to production. A dialogue summarization model fine-tuned from **BART-Large** was deployed using the **Hugging Face Inference API**, evaluated for latency, cost, and reliability, and monitored in real time using **Weights & Biases (W&B)**.

The goal of this project is to showcase practical LLM deployment skills, including cloud inference, performance evaluation, cost estimation, and basic observability.

---

## Model Details

* **Task:** Dialogue Summarization
* **Base Model:** `facebook/bart-large-cnn`
* **Fine-tuning Dataset:** HighlightSum (dialogue summarization)
* **Fine-tuning Method:** LoRA → merged into full FP16 model
* **Inference Setup:** No PEFT required at inference time

This merged model version was selected for deployment due to its superior performance and simplicity in production.

---

## 1. Cloud Deployment

### Deployment Platform

* **Platform:** Hugging Face Inference API
* **Deployment Type:** Managed inference endpoint
* **Billing Model:** Time-based billing ($0.27 / hour)

### Endpoint Characteristics

* Fully managed infrastructure
* Publicly accessible inference endpoint
* Supports programmatic API access

### Inference Access

Inference is demonstrated using a Python client that sends requests directly to the deployed Hugging Face endpoint.

Relevant files:

* `src/inference.py`
* `notebooks/inference_demo.ipynb`

The endpoint accepts raw dialogue text and returns a concise summary.

---

## 2. Performance & Cost Evaluation

A reproducible evaluation was conducted using **10 dialogue samples** to measure real-world inference behavior.

### Metrics Collected

* **Latency per request (ms)**
* **Response reliability**
* **Token usage (input + output)**
* **Estimated cost per 1K tokens (derived from time-based billing)**

### Results Summary

| Metric                             | Value                              |
| ---------------------------------- | ---------------------------------- |
| Average Latency                    | ~3,600 ms                          |
| Response Reliability               | 100%                               |
| Endpoint Cost                      | $0.27 / hour                       |
| Total Estimated Cost (10 requests) | ~$0.00077                          |
| Cost per 1K Tokens                 | Derived from runtime + token usage |

Latency values varied across requests, which is expected for managed cloud inference services due to cold starts and shared infrastructure.

### Reproducibility

Performance analysis can be reproduced by running:

* `notebooks/performance_analysis.ipynb`

This notebook:

* Sends inference requests
* Measures latency
* Counts tokens using the Hugging Face tokenizer
* Estimates cost based on endpoint runtime

---

## 3. Monitoring & Observability

### Monitoring Tool

* **Tool Used:** Weights & Biases (W&B)

### Logged Metrics

* `latency_ms` — per-request inference latency
* `error` — binary flag indicating request failure

### Observability Setup

Each inference request logs metrics to W&B in real time, enabling per-request visibility into model performance and reliability.

### Observed Behavior

* Latency ranged between ~2.9s and ~5.3s per request
* No errors were recorded across all test requests (0% error rate)
* One latency spike was observed, consistent with expected cloud inference variability

### Visualizations

Screenshots of the W&B dashboard are included in:

```
wandb_graphs/
├── Latency Plot.png
└── Error Plot.png
```

These plots show latency trends over successive inference requests and confirm stable endpoint behavior.

---

## 4. Repository Structure

```
.
├── UI/
│   └── app.py
│   └── requierements.txt         # Gradio UI for HF Space
├── requirements.txt        # Space dependencies
├── logs/
│   └── demo_summaries.xlsx
├── notebooks/
│   ├── inference_demo.ipynb
│   ├── performance_analysis.ipynb
│   └── wandb/
├── src/
│   └── inference.py
├── wandb_graphs/
│   ├── Latency Plot.png
│   └── Error Plot.png
├── .env
└── README.md

```

---

## 5. How to Run & Test

### Environment Setup

1. Create a virtual environment
2. Install dependencies (transformers, requests, wandb, pandas, torch)
3. Set environment variables in `.env`:

   * `HF_TOKEN`
   * `MODEL_URL`
   * `WANDB_API_KEY`

### Running Inference

* Open `notebooks/inference_demo.ipynb`
* Execute cells to send requests to the deployed endpoint

### Running HF Space

You can also choose to run this on the hugging face spaces with this [link](https://huggingface.co/spaces/Chibuu/llm-inference-demo) 

### Running Performance & Monitoring Tests

* Open `notebooks/performance_analysis.ipynb`
* Run all cells to:

  * Collect latency metrics
  * Log results to W&B
  * Estimate cost and reliability

---

## Conclusion

This project demonstrates a complete LLM deployment workflow:

* Fine-tuned model deployed to a cloud inference service
* Realistic performance and cost evaluation
* Integrated monitoring with real-time observability
* Reproducible testing and clear documentation

The system is production-ready, testable, and observable, satisfying all requirements for the Module 2 Capstone Project.
