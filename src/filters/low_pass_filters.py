import numpy as np
from PIL import Image

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                     np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    image_array = np.array(image)
    kernel = gaussian_kernel(kernel_size, sigma)
    height, width = image_array.shape[0], image_array.shape[1]
    pad_width = kernel_size // 2
    output_image = np.zeros_like(image_array)
    
    for i in range(pad_width, height - pad_width):
        for j in range(pad_width, width - pad_width):
            region = image_array[i - pad_width:i + pad_width + 1, j - pad_width:j + pad_width + 1]
            for k in range(3):
                output_image[i, j, k] = np.sum(region[:, :, k] * kernel)
    
    return Image.fromarray(np.uint8(output_image))

def apply_media_blur(image, kernel_size=3):
    image_array = np.array(image)
    height, width = image_array.shape[0], image_array.shape[1]
    pad_width = kernel_size // 2
    output_image = np.zeros_like(image_array)
    
    for i in range(pad_width, height - pad_width):
        for j in range(pad_width, width - pad_width):
            region = image_array[i - pad_width:i + pad_width + 1, j - pad_width:j + pad_width + 1]
            for k in range(3):
                output_image[i, j, k] = np.mean(region[:, :, k])
    
    return Image.fromarray(np.uint8(output_image))
