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
    # Write your code here
    img=torch.as_tensor(img,dtype=torch.float32)
    if img.ndim!=2 or pad_width<0 :
      return -1
    padding_shape=(img.shape[0]+pad_width*2,img.shape[1]+pad_width*2)
    padded_tensor=torch.zeros(padding_shape)
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        padded_tensor[i+pad_width,j+pad_width]=img[i,j]

    return padded_tensor.tolist()
