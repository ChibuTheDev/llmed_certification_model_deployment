
import argparse
from huggingface_hub import create_inference_endpoint

def main():
    parser = argparse.ArgumentParser(description="Create Hugging Face Inference Endpoint")

    # ---- Configurable parameters ----
    parser.add_argument("--name", required=True, help="Endpoint name")
    #parser.add_argument("--repository", required=True, help="Model repository")
    parser.add_argument("--repository", default="ReadyTensorCertification/bart-highlightsum-merged")
    parser.add_argument("--framework", default="pytorch")
    parser.add_argument("--task", default="summarization")
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--vendor", default="aws")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--type", default="protected", choices=["public", "protected"])

    args = parser.parse_args()

    # ---- Create endpoint ----
    endpoint = create_inference_endpoint(
        name=args.name,
        repository=args.repository,
        framework=args.framework,
        task=args.task,
        accelerator=args.accelerator,
        vendor=args.vendor,
        region=args.region,
        type=args.type,
    )

    # ---- Output PARAMETERS (not fixed values) ----
    print("\n✅ Endpoint creation requested with parameters:\n")

    print(f"name: {args.name}")
    print(f"repository: {args.repository}")
    print(f"framework: {args.framework}")
    print(f"task: {args.task}")
    print(f"accelerator: {args.accelerator}")
    print(f"vendor: {args.vendor}")
    print(f"region: {args.region}")
    print(f"type: {args.type}")

    print("\n⏳ Status:", endpoint.status)
    print("🔗 Endpoint URL (when ready):", endpoint.url)


if __name__ == "__main__":
    main()

