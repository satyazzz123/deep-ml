import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    logits=torch.as_tensor(logits,dtype=torch.float32)
    targets=torch.as_tensor(targets,dtype=torch.int32)
    max_element=torch.max(logits,dim=-1,keepdim=True).values
    normalied_logits=logits-max_element
    e_x=torch.exp(normalied_logits)
    e_x_sum=torch.log(torch.sum(e_x,dim=-1,keepdims=True))
    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    )


    loss=max_element+e_x_sum-target_logits
    return loss.mean()

