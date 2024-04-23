from PIL import Image
import torch 
import numpy as np

def save_any_image(path, image):
    # type
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
    
    # shape
    if image.ndim == 4:
        image = image[0]
        print("Warning: image has batch dimension, only saving first image")
    if image.shape[0] == 1:
        image = image[0]
    elif image.shape[0] in [3, 4]:
        image = image.transpose(1, 2, 0)
        print("Warning: was your image height size 3 or 4? If yes, the result is not as expected")
        print("Warning: was your batch size 3 or 4? If yes, the result is not as expected")
    
    # range, dtype
    assert image.min() >= 0, f"Image has invalid range: {image.min()} - {image.max()}"
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif np.unique(image).size == 2 and image.max() == 1 and image.min() == 0:
        image = image.astype(np.uint8) * 255
    elif image.max() <= 1:
        assert image.dtype != np.uint8, "Image is already in [0, 1] range but has uint8 dtype"
        image = (image * 255).astype(np.uint8)
    elif image.max() <= 255:
        assert image.dtype == np.uint8, f"Image is in [0, 255] range but have {image.dtype} dtype"
    else:
        raise ValueError(f"Image has invalid range: {image.min()} - {image.max()}")
    
    # save
    image = Image.fromarray(image)
    if '.' not in path:
        path = path + '.png'
    image.save(path)
    print(f"Image saved to {path}")

    return