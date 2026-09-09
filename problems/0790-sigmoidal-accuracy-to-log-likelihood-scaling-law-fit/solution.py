import torch
def fit_sigmoid_scaling(nll_list, acc_list, predict_nll):
    """
    Fit acc = 1 / (1 + exp(a*nll + b)) via least squares in logit space and
    predict accuracy at predict_nll.

    Returns:
        [a, b, predicted_acc] as a list of floats.
    """
    nll_list=torch.as_tensor(nll_list,dtype=torch.float32)
    acc_list=torch.as_tensor(acc_list,dtype=torch.float32)
    inverse_logit=torch.log((1-acc_list)/(acc_list))
    ones=torch.ones(len(nll_list),1)
    X=torch.concat((nll_list.reshape(-1,1),ones),dim=-1)
    theta=torch.inverse(X.T@X) @ X.T@inverse_logit
    pred=1/(1+torch.exp(predict_nll*theta[0]+theta[1]))
    return (theta[0].item(),theta[1].item(),pred.item())



    