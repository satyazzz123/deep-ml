import torch

def transpose_matrix(a) -> torch.Tensor:

    a_t = torch.as_tensor(a)
    transposed_matrix=[]
    for i in range(len(a_t[0])):
        new_row=[]
        for j in range(len(a_t)):
            new_row.append(a_t[j][i])
        transposed_matrix.append(new_row)
    return torch.tensor(transposed_matrix)

    