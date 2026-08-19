import torch

def calculate_sla_metrics(
    latencies: torch.Tensor,
    statuses: torch.Tensor,
    latency_sla_ms: float = 100.0
) -> dict:

    latencies = torch.as_tensor(latencies, dtype=torch.float32)
    statuses = torch.as_tensor(statuses, dtype=torch.float32)
    if len(latencies)==0 or len(statuses)==0:
      return {}

    success_mask = statuses == 0

    sla_mask = (
        (latencies <= latency_sla_ms) &
        success_mask
    )

    latency_sla_compliance = (
        torch.sum(sla_mask) /
        (torch.sum(success_mask) + 1e-7)
    )

    error_mask = (statuses == 1) | (statuses == 2)

    error_rate = (
        torch.sum(error_mask) /
        (len(statuses) + 1e-7)
    )

    overall_sla_compliance = (
        torch.sum(sla_mask) /
        (len(latencies) + 1e-7)
    )

    return {
        "latency_sla_compliance": round(
            latency_sla_compliance.item() * 100, 2
        ),
        "error_rate": round(
            error_rate.item() * 100, 2
        ),
        "overall_sla_compliance": round(
            overall_sla_compliance.item() * 100, 2
        ),
    }