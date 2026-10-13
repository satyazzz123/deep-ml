import torch

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> torch.Tensor:
    num_of_variables = len(x)
    num_of_func = len(f(x))

    jacobian = torch.zeros((num_of_func, num_of_variables))

    for i in range(num_of_variables):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        func_plus = f(x_plus)
        func_minus = f(x_minus)

        for j in range(num_of_func):
            jacobian[j, i] = (
                func_plus[j] - func_minus[j]
            ) / (2 * h)

    return jacobian