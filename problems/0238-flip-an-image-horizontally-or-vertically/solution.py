import torch

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: 2D or 3D list/array/tensor representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'
    
    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    # Your code here
    image=torch.as_tensor(image,dtype=torch.float32)
    if image.ndim not in (2, 3):
      return -1
    if direction not in ("horizontal", "vertical"):
      return -1
    if direction=="horizontal":
      if len(image.shape)==2:
        return torch.flip(image,dims=[-1]).tolist()
      return torch.flip(image,dims=[1]).tolist()

    else:
      if len(image.shape)==2:
        return torch.flip(image,dims=[0]).tolist()
      return torch.flip(image,dims=[0]).tolist()



    