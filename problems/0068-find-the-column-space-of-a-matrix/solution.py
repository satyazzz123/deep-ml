import torch

def matrix_image(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the column space (image) of a matrix A.
    Returns the basis vectors (original columns) that span the column space.

    Args:
        A: Input matrix as a torch.Tensor

    Returns:
        torch.Tensor: Matrix whose columns form a basis for the column space of A
    """
    # Write your code here
    
    A=torch.as_tensor(A,dtype=torch.float32)
    og_tensor=A.detach().clone()
    for i in range(len(A)-1):
      
      pivot_row=i
      if A[pivot_row][pivot_row]==0:
        for j in range(i+1,len(A)):
          if A[j][i]!=0:
            pivot_row=j
            break
        A[[i,pivot_row]]=A[[pivot_row,i]]
      
      pivot=A[i][i]

      A[i]=A[i]/pivot

      for j in range(i+1,len(A)):
        factor=A[j][i]
        A[j]=A[j]-A[i]*factor
    
 




    col_list=[]
    for i in range(len(A)):
      if A[i][i]!=0:
        col_list.append(i)



    print(col_list)
    og_tensor=og_tensor.T
    final_list=[]
    for i in col_list:
      final_list.append(og_tensor[i].tolist())
    return torch.tensor(final_list).T
