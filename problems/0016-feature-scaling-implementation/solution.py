import torch
def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:

    data_t = torch.as_tensor(data, dtype=torch.float)
    mean=[]
    std_deviation=[]
    ## code for standardization
    for cols in range(len(data_t[0])):
        x=data_t[:,cols]
        x_mean=x.mean()
        x_std=x.std(correction=0)
        mean.append(x_mean)
        std_deviation.append(x_std)
    mean=torch.tensor(mean)
    std_deviation=torch.tensor(std_deviation)
    standardised_values=(data_t-mean)/(std_deviation+1e-8)
    x_max=[]
    x_min=[]
    ## code for min-max normalization
    for cols in range(len(data_t[0])): 

          x=data_t[:,cols]

          x_min.append(x.min())

          x_max.append(x.max())

    x_min=torch.tensor(x_min)

    x_max=torch.tensor(x_max)

    numerator=data_t-x_min

    denominator=x_max-x_min +1e-8

    normalized_values=numerator/denominator 
    return(torch.round(standardised_values,decimals=4),torch.round(normalized_values,decimals=4))
              
    
