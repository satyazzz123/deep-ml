import torch
import torch

def confusion_matrix_fun(data: list) -> torch.Tensor:
    """
    Generate a 2x2 confusion matrix for binary classification.

    Args:
        data: A list of [y_true, y_pred] pairs for binary labels (0 or 1)

    Returns:
        A 2x2 torch.Tensor confusion matrix arranged as [[TP, FN], [FP, TN]]
    """
    TP=0
    FP=0
    TN=0
    FN=0
    for i in data:
      if (i[0]+i[1]==2):
        TP+=1
      if (i[0]-i[1]==-1):
        FP+=1
      if (i[0]+i[1]==0):
        TN+=1
      if (i[1]-i[0]==-1):
        FN+=1
    return torch.tensor([[TP,FN],[FP,TN]])
  


def accuracy_fun(data:list):
    TP=0
    FP=0
    TN=0
    FN=0
    for i in data:
      if (i[0]+i[1]==2):
        TP+=1
      if (i[0]-i[1]==-1):
        FP+=1
      if (i[0]+i[1]==0):
        TN+=1
      if (i[1]-i[0]==-1):
        FN+=1
    return torch.tensor((TP+TN)/(TP+FP+TN+FN))
def f1_fun(data:list):
    TP=0
    FP=0
    TN=0
    FN=0
    for i in data:
      if (i[0]+i[1]==2):
        TP+=1
      if (i[0]-i[1]==-1):
        FP+=1
      if (i[0]+i[1]==0):
        TN+=1
      if (i[1]-i[0]==-1):
        FN+=1
    return torch.tensor((2*TP)/(2*TP+FN+FP))

def specificity_fun(data:list):
    TP=0
    FP=0
    TN=0
    FN=0
    for i in data:
      if (i[0]+i[1]==2):
        TP+=1
      if (i[0]-i[1]==-1):
        FP+=1
      if (i[0]+i[1]==0):
        TN+=1
      if (i[1]-i[0]==-1):
        FN+=1
    return torch.tensor((TN)/(TN+FP))


def NPV(data:list):
    TP=0
    FP=0
    TN=0
    FN=0
    for i in data:
      if (i[0]+i[1]==2):
        TP+=1
      if (i[0]-i[1]==-1):
        FP+=1
      if (i[0]+i[1]==0):
        TN+=1
      if (i[1]-i[0]==-1):
        FN+=1
    return torch.tensor((TN)/(TN+FN))

def performance_metrics(actual: torch.Tensor, predicted: torch.Tensor) -> tuple:
    """
    Compute performance metrics for binary classification using PyTorch.

    Args:
        actual: torch.Tensor of actual binary labels (0 or 1)
        predicted: torch.Tensor of predicted binary labels (0 or 1)

    Returns:
        tuple: (confusion_matrix Tensor, accuracy float, f1 float, specificity float, negative_predictive_value float)
    """
    # Implement your solution here using PyTorch built-ins
    combined_list=[]
    actual=torch.as_tensor(actual,dtype=torch.float32)
    predicted=torch.as_tensor(predicted,dtype=torch.float32)
    for i in range(len(actual)):
      combined_list.append([actual[i].item(),predicted[i].item()])
    confusion_matrix=confusion_matrix_fun(combined_list)
    accuracy=accuracy_fun(combined_list)
    f1=f1_fun(combined_list)
    specificity=specificity_fun(combined_list)
    negative_predictive=NPV(combined_list)  
    return confusion_matrix,round(accuracy.item(), 3),round(f1.item(), 3),round(specificity.item(), 3),round(negative_predictive.item(), 3)
    