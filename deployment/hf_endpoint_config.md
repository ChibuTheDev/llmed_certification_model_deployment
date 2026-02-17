model_id: chibu/bart-dialogue-summarizer

deployment:
  provider: huggingface_inference_endpoints
  cloud_provider: aws
  region: us-east-1
  availability_zone: N. Virginia
  hardware:
    instance_family: intel_sapphire_rapids
    vcpus: 8
    memory_gb: 16
    accelerator: none
  pricing:
    hourly_cost_per_replica_usd: 0.27
    billing_unit: per_running_replica

scaling:
  strategy: auto
  metric: requests_per_second
  min_replicas: 1
  max_replicas: 3
