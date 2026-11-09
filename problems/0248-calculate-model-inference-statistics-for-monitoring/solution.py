import torch

def calculate_inference_stats(latencies_ms: torch.Tensor) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: torch.Tensor of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    if latencies_ms.numel() == 0:
        return {
           
        }

    # Ensure tensor is 1D float on CPU for numerical stability & consistent quantile computation
    latencies = latencies_ms.flatten().float()
    
    # 1. Average Latency (Mean)
    avg_latency = torch.mean(latencies).item()
    
    # 2. Throughput: total_requests / total_time_in_seconds
    # total_time_sec = sum(latencies_ms) / 1000
    # throughput = N / (sum(latencies_ms) / 1000) = (1000 * N) / sum(latencies_ms) = 1000 / avg_latency_ms
    total_latency_ms = torch.sum(latencies).item()
    throughput = (len(latencies) / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

    # 3. Percentiles (p50, p95, p99) via quantile
    # q in [0, 1] maps to p50 (0.50), p95 (0.95), p99 (0.99)
    quantiles = torch.tensor([0.50, 0.95, 0.99], device=latencies.device)
    p50, p95, p99 = torch.quantile(latencies, quantiles).tolist()

    return {
        'throughput_per_sec': round(throughput, 2),
        'avg_latency_ms': round(avg_latency, 2),
        'p50_ms': round(p50, 2),
        'p95_ms': round(p95, 2),
        'p99_ms': round(p99, 2),
    }