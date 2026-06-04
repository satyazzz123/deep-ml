import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

class CustomNumbersDataset(Dataset):
    def __init__(self,x,y):
        # Generate dummy data: numbers 0 to 99
        self.data = x
        self.labels=y

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Return one sample at the given index
        return self.data[idx], self.labels[idx]
def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, 
                    learning_rate: float, n_epochs: int, 
                    batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights as a tensor
    """
    # Your implementation here
    X=torch.as_tensor(X,dtype=torch.float32)
    W=torch.zeros(X.shape[-1],dtype=torch.float32,requires_grad=True)
    y=torch.as_tensor(y,dtype=torch.float32)
    criterion = nn.MSELoss()
    
    if method=='batch':
      for _ in range(n_epochs):
    

        y_pred=X@W
        loss=criterion(y_pred,y)
        loss.backward()

        with torch.no_grad():
          W-=learning_rate*W.grad
          W.grad.zero_()
      return W.detach()

    if method=='stochastic':
      for _ in range(n_epochs):
        for i,j in zip(X,y):
          y_pred=torch.dot(i,W)
          loss=criterion(y_pred,j)
          loss.backward()
          with torch.no_grad():
            W-=learning_rate*W.grad
            W.grad.zero_()
      return W.detach()

    if method=="mini_batch":
      mydataset=CustomNumbersDataset(X,y)
      
      mydataloader=DataLoader(mydataset,batch_size,shuffle=False)

      for _ in range(n_epochs):
        for batch,y in mydataloader:
          # print(batch)
          y_pred=batch@W
          loss=criterion(y_pred,y)
          loss.backward()
          with torch.no_grad():
            W-=learning_rate*W.grad
            W.grad.zero_()
      return W.detach()
          
          








