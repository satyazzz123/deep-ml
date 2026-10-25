
import math

def auto_mlp_from_spaces(
    obs_space: dict,
    act_space: dict,
    hidden_layers: list
) -> dict:

    # 1. Calculate input dimension
    space = obs_space["type"]

    if space == "Box":
        input_dim = math.prod(obs_space["shape"])
    elif space == "Discrete":
        input_dim = obs_space["n"]
    elif space == "MultiBinary":
        input_dim = math.prod(
            obs_space.get("shape", (obs_space["n"],))
        )
    else:
        raise ValueError(f"Unsupported observation space: {space}")

    # 2. Calculate output dimension
    action = act_space["type"]

    if action == "Box":
        output_dim = math.prod(act_space["shape"])
    elif action == "Discrete":
        output_dim = act_space["n"]
    elif action == "MultiBinary":
        output_dim = math.prod(
            act_space.get("shape", (act_space["n"],))
        )
    else:
        raise ValueError(f"Unsupported action space: {action}")

    # 3. Construct the full architecture
    dimensions = [input_dim] + hidden_layers + [output_dim]

    # 4. Calculate layer shapes and parameter count
    layer_shapes = []
    total_params = 0

    for i in range(1, len(dimensions)):
        input_width = dimensions[i - 1]
        output_width = dimensions[i]

        weight_shape = (input_width, output_width)
        bias_shape = (output_width,)

        layer_shapes.append((weight_shape, bias_shape))

        total_params += (
            math.prod(weight_shape) + math.prod(bias_shape)
        )

    # 5. Return all required results
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "layer_shapes": layer_shapes,
        "total_params": total_params,
    }
