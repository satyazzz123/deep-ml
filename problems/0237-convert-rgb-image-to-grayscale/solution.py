import torch

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or torch.Tensor of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Write your code here
    image=torch.as_tensor(image,dtype=torch.float32)
    # H,W,channel=image.shape
    val=image.shape
    # print(len(val))

    if torch.max(image).item()>255 or torch.min(image).item()<0 or len(val)<3:
      return -1
    
    gray_scale=torch.tensor([0.299,0.587,0.114])
    gray_image=image@gray_scale
    return torch.round(gray_image.squeeze(dim=-1)).tolist()
