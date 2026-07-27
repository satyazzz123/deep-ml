import torch

def compute_efficiency(n_experts: int, k_active: int, d_in: int, d_out: int) -> torch.Tensor:
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs as a torch.Tensor
    """
    Flops_dense_layer=n_experts*d_in*d_out
    Flops_moe_active_layer=k_active*d_in*d_out
    return torch.tensor(((Flops_dense_layer-Flops_moe_active_layer)/Flops_dense_layer)*100)
    