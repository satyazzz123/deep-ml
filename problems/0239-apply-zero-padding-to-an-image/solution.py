import torch

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    
    Args:
        img: 2D list or torch.Tensor of pixel values
        pad_width: integer number of pixels to pad on each side
    
    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """
    # 1. Validate pad_width
    if not isinstance(pad_width, int) or pad_width < 0:
        return -1

    # 2. Safely parse and validate img dimensions
    try:
        # Convert to tensor; integer dtype preserves pixel integer values
        tensor_img = torch.as_tensor(img, dtype=torch.int64)
    except Exception:
        return -1

    if tensor_img.ndim != 2:
        return -1

    h, w = tensor_img.shape

    # Handle zero padding edge-case
    if pad_width == 0:
        return tensor_img.tolist()

    # 3. Create padded tensor of zeros
    padded_shape = (h + 2 * pad_width, w + 2 * pad_width)
    padded_tensor = torch.zeros(padded_shape, dtype=torch.int64)

    # 4. Copy original image into inner window
    padded_tensor[pad_width : pad_width + h, pad_width : pad_width + w] = tensor_img

    # 5. Return as 2D list of integers
    return padded_tensor.tolist()