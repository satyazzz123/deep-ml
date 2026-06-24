import torch
def safe_tensor_creation(data):
    try:
        # This will fail if data is incompatible (e.g., a list of strings)
        return torch.as_tensor(data)
    except (RuntimeError, TypeError, ValueError) as e:
        # print(f"Error encountered: {e}")
        return -1

def calculate_brightness(img) -> float:
    """
    Calculate the average brightness of a grayscale image using PyTorch.

    Args:
        img: A 2D list where each element represents a pixel value between 0-255.

    Returns:
        The average brightness rounded to two decimal places, or -1 for invalid inputs.
    """
    # Write your code here
    img=safe_tensor_creation(img)
    if isinstance(img,int):
      return -1
    if img.numel() == 0:
      return -1
    min_val, max_val = 0.0, 255.0


    is_in_range = torch.all((img >= min_val) & (img <= max_val)).item()
    if is_in_range:
      
      avg=torch.sum(torch.sum(img,dim=1))/(img.shape[0]*img.shape[1])
    
      return avg.item()
    return -1
