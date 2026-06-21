import torch

def compressed_col_sparse_matrix(dense_matrix: torch.Tensor):
    """
    Convert a dense matrix tensor into its Compressed Column Sparse (CSC) representation
    using PyTorch's built-in sparse CSC tensor support.

    :param dense_matrix: 2D torch.Tensor representing the dense matrix
    :return: Tuple of (values, row_indices, col_pointer) as torch.Tensors
    """
    dense_matrix=torch.as_tensor(dense_matrix,dtype=torch.float32)
    value_tensor=[]
    for j in range(len(dense_matrix[0])):
      for i in range(len(dense_matrix)):
        if dense_matrix[i][j]!=0:
          value_tensor.append(dense_matrix[i][j].item())

    row_idx=[]
    for j in range(len(dense_matrix[0])):
      for i in range(len(dense_matrix)):
        if dense_matrix[i][j]!=0:
          row_idx.append(i)
    
    col_ptr=[]
    for j in range(len(dense_matrix[0])):
      value=0
      for i in range(len(dense_matrix)):
        if dense_matrix[i][j]!=0:
          value+=1
      col_ptr.append(value)
    col_ptr=[0]+col_ptr
    for i in range(1,len(col_ptr)):
      col_ptr[i]=col_ptr[i]+col_ptr[i-1]
      

 

    return torch.tensor(value_tensor),torch.tensor(row_idx),torch.tensor(col_ptr)
