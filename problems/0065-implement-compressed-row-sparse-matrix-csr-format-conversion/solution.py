import torch

def compressed_row_sparse_matrix(dense_matrix) -> tuple:
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation
    using PyTorch's built-in sparse CSR tensor support.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values tensor, column indices tensor, row pointer tensor)
    """
    # dense_matrix=torch.as_tensor(dense_matrix,dtype=torch.float32)
    values_array=[]
    for d in dense_matrix:
      for v in d:
        if v!=0:
          values_array.append(v)
    column_indices=[]
    for d in dense_matrix:
      for i in range(len(d)):
        if d[i]!=0:
          column_indices.append(i)
    row_pointer=[]
    
    for d in dense_matrix:
      val=0
      for i in d:
        if i!=0:
          val+=1
      row_pointer.append(val)
    row_pointer=[0]+row_pointer
    for i in range(1,len(row_pointer)):
      row_pointer[i]=row_pointer[i-1]+row_pointer[i]



    return (values_array,column_indices,row_pointer)