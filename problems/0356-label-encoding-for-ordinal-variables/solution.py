import torch

def label_encode_ordinal(values: list, order: list) -> torch.Tensor:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        torch.Tensor of integers (dtype=torch.long) representing the encoded values,
        with -1 for any value not found in order
    """
    unique_values=list(set(values))
    missing_vals=[]
    if set(values)!=set(order):
      missing_vals=list(set(values)-set(order))
    # ordered_list=[i for i in range(len(order))]
    ordered_set_with_values={}
    for i in range(len(order)):
      ordered_set_with_values[order[i]]=i
    print(missing_vals)
    if len(missing_vals)!=0:
       for i in range(len(missing_vals)):
        ordered_set_with_values[missing_vals[i]]=-1
    ordinal_list=[]
    for i in values:
      ordinal_list.append(ordered_set_with_values[i])

    return torch.tensor(ordinal_list)

