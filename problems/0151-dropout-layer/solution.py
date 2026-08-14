import torch

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.

        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        self.p=p
        self.mask=None
        torch.manual_seed(42)


    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass of the dropout layer.

        Generate a new mask on each training forward pass and store it in self.mask.
        """

        x=torch.as_tensor(x,dtype=torch.float32)
        
        if training:
          self.mask = torch.bernoulli(torch.full((x.shape), 1-self.p))
          dropped_out_x=(x*self.mask)/(1-self.p)
          

        else:
          self.mask=torch.ones(x.shape)
          dropped_out_x=x
        return dropped_out_x
        
        

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass of the dropout layer.

        Use the stored self.mask from the most recent forward pass.
        """
        # Your code here
        grad=torch.as_tensor(grad,dtype=torch.float32)
        dropped_out_grad=(grad*self.mask)/(1-self.p)
        return dropped_out_grad