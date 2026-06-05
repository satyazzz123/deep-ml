import torch
import torch.nn.functional as F

def add_padding(x):
    a,b=x.shape
    result=x if isinstance(x,list) else x.tolist()

    for i in range(len(result)):
        result[i].insert(0,0)
        result[i].append(0)

    result=torch.as_tensor(result,dtype=torch.float32)
    result=result.T
    result=result.tolist()

    for i in range(len(result)):
        result[i].insert(0,0)
        result[i].append(0)

    return torch.tensor(result,dtype=torch.float32).T


def simple_conv2d(input_matrix, kernel, padding, stride):

    x=torch.as_tensor(input_matrix,dtype=torch.float32)
    n=x.shape[0]

    kernel=torch.as_tensor(kernel,dtype=torch.float32)

    for _ in range(padding):
        x=add_padding(x)

    kernel_size=kernel.shape[0]

    output_shape=((n+2*padding-kernel_size)//stride)+1

    conv=[]

    for i in range(output_shape):
        for j in range(output_shape):

            minor = (
                x[
                    i*stride:i*stride+kernel_size,
                    j*stride:j*stride+kernel_size
                ]
                * kernel
            )

            conv.append(torch.sum(minor))

    conv=torch.tensor(conv,dtype=torch.float32)

    conv=torch.reshape(conv,(output_shape,output_shape))

    return conv