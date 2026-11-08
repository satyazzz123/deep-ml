import torch

import torch.nn.functional as F
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    s_scaled = student_logits / temperature
    t_scaled = teacher_logits.detach() / temperature

    s_log_probs = F.log_softmax(s_scaled, dim=-1)
    t_log_probs = F.log_softmax(t_scaled, dim=-1)
    t_probs = torch.exp(t_log_probs)

    # KL(P || Q) = sum(P * (log(P) - log(Q))) along class dimension
    kl_per_sample = torch.sum(t_probs * (t_log_probs - s_log_probs), dim=-1)

    # Average over batch dimension and scale by T^2
    return kl_per_sample.mean() * (temperature ** 2)