import torch
import torch.nn.functional as F
def moe(x: torch.Tensor, We: torch.Tensor, Wg: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    x=torch.tensor(x,dtype=torch.float32)
    We=torch.tensor(We,dtype=torch.float32)
    Wg=torch.tensor(Wg,dtype=torch.float32)
    logits=x@Wg
    values,_=torch.max(logits,dim=-1)
    logits=logits-values.reshape(-1,logits.shape[1],1)
    exp_vals=torch.exp(logits)
    total_exp=torch.sum(exp_vals,dim=-1,keepdim=True)
    softmaxed_value=exp_vals/total_exp


    _,indices=torch.topk(softmaxed_value,top_k,dim=-1)
    mask= torch.full(softmaxed_value.shape, float(0))
    mask.scatter_(dim=-1,index=indices,value=float(1))
    top_k=mask*softmaxed_value
    top_k=top_k/torch.sum(top_k,dim=-1,keepdim=True)
    x_temp=x.unsqueeze(2)
    # print(x_temp.shape)
    experts_score=x.unsqueeze(1)@We
    experts_score=experts_score.permute(0,2,1,3)
    weighted_aggregation=top_k.unsqueeze(2)@experts_score
    return weighted_aggregation.squeeze(2)

