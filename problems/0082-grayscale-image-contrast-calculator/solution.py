import torch

def calculate_contrast(img: torch.Tensor) -> float:
    """
    Calculate the contrast of a grayscale image.
    Args:
        img (torch.Tensor): 2D tensor representing a grayscale image with pixel values between 0 and 255.
    Returns:
        float: Contrast value rounded to 3 decimal places.
    """
    # Your code here
    img=torch.as_tensor(img,dtype=torch.float32)
    max=torch.max(img)
    min=torch.min(img)
    return (max-min).item()