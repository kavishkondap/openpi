import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

def crop(images: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """
    Crops a batch of images to a target aspect ratio using only NumPy.
    
    Args:
        images: [..., H, W, C] array
        aspect_ratio: desired width / height
    
    Returns:
        Cropped images as a NumPy array with shape [..., new_h, new_w, C]
    """
    # Flatten batch dimensions
    orig_shape = images.shape
    flat_images = images.reshape(-1, *orig_shape[-3:])
    batch_size = flat_images.shape[0]
    
    cropped = []
    for img in flat_images:
        h, w, c = img.shape
        cur_ratio = w / h
        if cur_ratio > aspect_ratio:
            # Too wide → crop width
            new_w = int(h * aspect_ratio)
            left = (w - new_w) // 2
            cropped_img = img[:, left:left+new_w, :]
        elif cur_ratio < aspect_ratio:
            # Too tall → crop height
            new_h = int(w / aspect_ratio)
            top = (h - new_h) // 2
            cropped_img = img[top:top+new_h, :, :]
        else:
            cropped_img = img  # no crop
        cropped.append(cropped_img)
    
    # Stack and reshape
    cropped = np.stack(cropped, axis=0)
    return cropped.reshape(*orig_shape[:-3], *cropped.shape[1:])