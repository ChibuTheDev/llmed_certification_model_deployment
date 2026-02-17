# Module 2 Capstone Project — LLM Deployment & Monitoring

## Overview

This project (repository) demonstrates a complete production-style deployment of a [fine-tuned Large Language Model (LLM)](https://app.readytensor.ai/publications/llm-engineering-and-deployment-certification-peft-of-bart-for-dialogue-summarization-FRZ4ZyEHbobq) for dialogue summarization.

The deployed model is a fine-tuned version of [**facebook/bart-large-cnn**](https://huggingface.co/facebook/bart-large-cnn), optimized for summarizing multi-speaker conversations.

This repository supports the Ready Tensor publication: **Deployment Plan for a Fine-Tuned BART Dialogue Summarization Model**

The project demonstrates:

- Model deployment using Hugging Face Inference Endpoints
- Real-time inference via API client
- Performance and cost evaluation
- Monitoring and observability integration
- Deployment configuration documentation
- Reproducible testing

---

## Model Information

* **Task:** Dialogue Summarization
* **Base Model:** `facebook/bart-large-cnn`
* **Fine-tuning Dataset:** HighlightSum (dialogue summarization)
* **Fine-tuning Method:** LoRA → merged into full FP16 model
* **Inference Setup:** No PEFT required at inference time

This merged model version was selected for deployment due to its superior performance and simplicity in production.

| Attribute | Value |
|------------|--------|
| Task | Dialogue Summarization |
| Base Model | facebook/bart-large-cnn |
| Architecture | Encoder–Decoder (BART) |
| Parameter Count | 406M |
| Fine-Tuning | LoRA (merged to full FP16 model) |
| Context Length | 1024 tokens |
| Max Output Length | 128 tokens |

The LoRA adapters were merged into the full model prior to deployment to simplify inference and eliminate PEFT dependencies in production.

---

##  Deployment Strategy

### Deployment Platform

- **Platform:** Hugging Face Inference Endpoint
- **Deployment Type:** Managed real-time inference
- **vCPU:** Intel Sapphire Rapids (16GB)
- **Billing Model:** Time-based billing (~$0.27/hour)
- **Region:** US-East
- **Scaling:** 1–3 replicas (auto-scaling enabled)

Deployment configuration details are documented in `deployment/hf_endpoint_config.md` 

---

## Why Hugging Face Inference Endpoint?

- Fully managed infrastructure
- Minimal DevOps overhead
- Integrated autoscaling
- Secure API-based access
- Direct integration with Hugging Face Hub

Alternative platforms (AWS SageMaker, self-hosted EC2, Modal, vLLM) were considered but not selected due to higher setup complexity or infrastructure management overhead.


### Endpoint Characteristics

* Fully managed infrastructure
* Publicly accessible inference endpoint
* Supports programmatic API access

---  

## Repository Structure    


```bash
.
├── deployment/
│ └── hf_endpoint_config.md
├── logs/
│   └── demo_summaries.csv  Example Prompts & Outputs
├── UI/
│   └── app.py
├── notebooks/
│   ├── inference_demo.ipynb
│   ├── performance_and_cost_evaluation_and_wandb.ipynb
├── src/
│   └── inference.py
├── wandb_graphs/
│   ├── Latency Plot.png
│   └── Error Plot.png
├── docs/
│ ├── cost_estimate.xlsx
│ └── cost_estimate.pdf
├── requirements.txt       #Space dependencies&Gradio UI for HF Space
├── .env_example
└── README.md

```

---

##  Environment Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux  
```  

### Install Dependencies

```bash  
pip install -r requirements.txt
```  

### Configure Environment Variables  

Create a `.env file`. See `.env.example' 

```bash
HF_TOKEN=your_huggingface_api_key
MODEL_URL=https://your-inference-endpoint-url
WANDB_API_KEY=your_wandb_api_key
```  
These credentials are required for:  
- Secure API access  
- Monitoring integration  
- Endpoint communication  


## Deployment Instructions  

### Step 1: Upload Model to Hugging Face Hub

Push merged fine-tuned model to your private repository.

### Step 2: Create Inference Endpoint

1. Navigate to Hugging Face → Inference Endpoints    
2. Select your model    
3. Choose CPU/GPU instance  
4. Enable autoscaling (1–3 replicas)       
5. Deploy endpoint    
6. Copy endpoint URL    

### Step 3: Set Environment Variables

Add endpoint URL and API token to .env.

### Step 4: Test Deployment  

```bash  
python src/inference.py
```
Or use `notebooks/inference_demo.ipynb`

> (TO BE ENCLOSED?) Inference Access : Inference is demonstrated using a Python client that sends requests directly to the deployed Hugging Face endpoint. The endpoint accepts raw dialogue text and returns a concise summary.
Running Inference:  Open `notebooks/inference_demo.ipynb`, Execute cells to send requests to the deployed endpoint  

## Client Code

Client implementation is located in:

```bash  
python src/inference.py
```  
The client:  
- Sends POST requests to the inference endpoint  
- Accepts raw dialogue text  
- Returns generated summaries  
- Logs latency and error metrics  
- Handles common HTTP errors    

### Example Request  

```bash  
dialogue_single = """Olivia: Who are you voting for in this election?
Oliver: Liberals as always.
Olivia: Me too!!
Oliver: Great"""
summary_single = client.summarize(dialogue_single)
print("Single Dialogue Summary:\n", summary_single)
```  

### Example Input  

```bash  
"""Olivia: Who are you voting for in this election?
Oliver: Liberals as always.
Olivia: Me too!!
Oliver: Great"""  
```  

### Expected Output  

```bash  
Oliver is voting for Liberals this election.
```

Additional test examples are available in `src/demo_summaries.csv`

---  

## Performance & Cost Evaluation  

A reproducible evaluation was conducted using **10 dialogue samples** to measure real-world inference behavior.

Performance analysis (testing) can be reproduced by running:

* `notebooks/performance_analysis.ipynb`

This notebook:

* Sends inference requests
* Measures latency
* Counts tokens using the Hugging Face tokenizer
* Estimates cost based on endpoint runtime


Run all cells to:

  * Collect latency metrics
  * Log results to W&B  IMP
  * Estimate cost and reliability


### Metrics Collected

* **Latency per request (ms)**
* **Response reliability**
* **Token usage (input + output)**
* **Estimated cost per 1K tokens (derived from time-based billing)**    

### Observed Results Summary

| Metric                             | Value                              |
| ---------------------------------- | ---------------------------------- |
| Average Latency                    | ~3,600 ms                          |
| Response Reliability               | 100%                               |
| Endpoint Cost                      | $0.27 / hour                       |
| Total Estimated Cost (10 requests) | ~$0.0027                           |
| Cost per 1K Tokens @                | ~$0.0015                          |

> @ Cost per 1K Tokens = Derived from runtime + token usage  

Latency values varied across requests, which is expected for managed cloud inference behavior (cold starts, shared GPU utilization).

---

## 3. Monitoring & Observability

### Monitoring Tool

* **Tool Used:** Weights & Biases (W&B)

### Logged Metrics

* `latency_ms` — per-request inference latency
* `error_flag` — binary flag indicating request failure
* `token usage` - derived  

Metrics are logged per request for real-time visibility. See `performance_and_cost_evaluation_and_wandb.ipynb`.

### Monitoring Visualizations

Screenshots of the W&B dashboard are included in:

```
wandb_graphs/
├── Latency Plot.png
└── Error Plot.png
```

These plots show latency trends over successive inference requests and confirm stable endpoint behavior.


### Observability Setup

Each inference request logs metrics to W&B in real time, enabling per-request visibility into model performance and reliability.

### Observed Behavior

* Latency ranged between ~2.9s and ~5.3s per request
* No errors were recorded across all test requests (0% error rate)
* One latency spike was observed, consistent with expected cloud inference variability

---

## UI Demo

A Gradio-based UI is available in:

```bash  
UI/app.py
```  

This allows interactive summarization via Hugging Face Spaces.  

The Hugging Face Spaces demo UI is available at the following link: [Link](https://huggingface.co/spaces/Chibuu/llm-inference-demo). Please note that the current configuration connects to a private inference endpoint and therefore requires owner-level authentication.  

---  

## Cost Model

A dynamic cost model with adjustable parameters is provided:

- **Editable Spreadsheet**: `cost_model/cost_estimate.xlsx`
- **Viewable PDF**: `cost_model/cost_estimate.pdf`

The detailed and dynamically calculated deployment cost model, provided in `cost_estimate.xlsx`, includes infrastructure cost breakdowns (GPU, storage, network, monitoring), throughput assumptions, and cost-per-1,000 request calculations based on a conservative utilization model. All formulas are editable, allowing adjustment of GPU pricing, request volume, and optimization strategies to simulate different production scenarios.


## Security Considerations

- API authentication via Hugging Face tokens  
- Secure HTTPS communication  
- No raw dialogue persistence       
- Rate limiting via endpoint configuration    
- Environment variables for secret storage    

---

## Conclusion  

This project showcases a practical, production-oriented LLM deployment workflow:    

* Fine-tuned model deployed to a cloud inference service
* Realistic performance and cost evaluation
* Integrated monitoring with real-time observability
* Deployment reproducibility  


The system is deployable, testable, and observable, fully satisfying the GitHub repository requirements for [LLM Deployment&Engineering Ready Tensor Cerification](https://app.readytensor.ai/lessons/llmed-program-module-2-project-deploy-and-monitor-your-fine-tuned-llm-schoFmT5).    

