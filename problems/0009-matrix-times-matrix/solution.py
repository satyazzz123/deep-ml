import torch
import itertools

# --- 1. YOUR BROADCASTING FUNCTION ---
def broadcasting(a, b):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)

    shape_a = list(a.shape)
    shape_b = list(b.shape)
    
    updated_shape_a = [] 
    updated_shape_b = [] 
    
    if len(shape_a) > len(shape_b):
        diff = len(shape_a) - len(shape_b)
        updated_shape_b = [1] * diff + shape_b 
        
    if len(shape_b) > len(shape_a):
        diff = len(shape_b) - len(shape_a)
        updated_shape_a = [1] * diff + shape_a 
    
    if updated_shape_a == [] and updated_shape_b == []:
        if shape_a[-1] != shape_b[-2]:
            return torch.tensor(-1)
        else:
            for i in range(len(shape_b)-2):
                if shape_b[i] != shape_a[i] and shape_b[i] != 1 and shape_a[i] != 1:
                    return torch.tensor(-1)

                if shape_a[i] == 1 or shape_b[i] == 1:
                    if shape_a[i] >= shape_b[i]:
                        shape_b[i] = shape_a[i]
                    else:
                        shape_a[i] = shape_b[i]
            return shape_a, shape_b 

    if updated_shape_b != []: 
        if shape_a[-1] != updated_shape_b[-2]:
            return torch.tensor(-1)
        else:
            for i in range(len(shape_a)-2): 
                if updated_shape_b[i] != shape_a[i] and updated_shape_b[i] != 1 and shape_a[i] != 1:
                    return torch.tensor(-1)

                if shape_a[i] == 1 or updated_shape_b[i] == 1:
                    if shape_a[i] >= updated_shape_b[i]:
                        updated_shape_b[i] = shape_a[i]
                    else:
                        shape_a[i] = updated_shape_b[i]
            return shape_a, updated_shape_b 

    if updated_shape_a != []: 
        if updated_shape_a[-1] != shape_b[-2]:
            return torch.tensor(-1)
        else:
            for i in range(len(shape_b)-2): 
                if shape_b[i] != updated_shape_a[i] and shape_b[i] != 1 and updated_shape_a[i] != 1:
                    return torch.tensor(-1)

                if updated_shape_a[i] == 1 or shape_b[i] == 1:
                    if updated_shape_a[i] >= shape_b[i]:
                        shape_b[i] = updated_shape_a[i]
                    else:
                        updated_shape_a[i] = shape_b[i]
            return updated_shape_a, shape_b 


# --- 2. THE MULTIPLICATION LOGIC ---
def matrixmul(a, b):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)

    # 1. Check compatibility
    result_shapes = broadcasting(a, b)
    if result_shapes == -1:
         return torch.tensor(-1)
    x, y = result_shapes

    # 2. Pad dimensions of 'a' and 'b' so they have the same number of dims
    # (This prevents indexing errors in our loops)
    while a.dim() < b.dim(): a = a.unsqueeze(0)
    while b.dim() < a.dim(): b = b.unsqueeze(0)

    # 3. Determine the final output shape
    m = x[-2] # Rows of final matrix
    n = x[-1] # Shared inner dimension
    p = y[-1] # Columns of final matrix
    
    # Calculate target batch dimensions
    batch_dims = [max(x[i], y[i]) for i in range(len(x) - 2)]
    out_shape = batch_dims + [m, p]
    result = torch.zeros(out_shape)

    # 4. Generate dynamic outer loops for batch dimensions
    batch_ranges = [range(d) for d in batch_dims]
    
    for batch_idx in itertools.product(*batch_ranges):
        
        # Broadcasting Magic: Modulo operator handles dimension size 1 vs N
        idx_a = tuple(batch_idx[i] % a.shape[i] for i in range(len(batch_dims)))
        idx_b = tuple(batch_idx[i] % b.shape[i] for i in range(len(batch_dims)))
        
        # Core 2D Matrix Multiplication
        for i in range(m):         
            for j in range(p):     
                dot_sum = 0
                for k in range(n): 
                    val_a = a[idx_a + (i, k)]
                    val_b = b[idx_b + (k, j)]
                    dot_sum += val_a * val_b
                    
                result[batch_idx + (i, j)] = dot_sum
                
    return result