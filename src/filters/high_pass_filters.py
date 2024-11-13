from PIL import Image
import numpy as np

import numpy as np
from PIL import Image

def laplacian_filter(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert('L')
    image = np.array(image)


    kernel = np.array([[0,  0,  1,  0,  0],
                       [0,  1,  2,  1,  0],
                       [1,  2, -16,  2,  1],
                       [0,  1,  2,  1,  0],
                       [0,  0,  1,  0,  0]])

    height, width = image.shape
    result_image = np.zeros_like(image)

    pad = 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            neighbors = padded_image[i-2:i+3, j-2:j+3]
            filtered_value = np.sum(neighbors * kernel)
            result_image[i - pad, j - pad] = np.clip(filtered_value, 0, 255)

    return Image.fromarray(result_image.astype(np.uint8))



def sobel_filter(image):
    image = image.convert('L')
    image = np.array(image)

    kernel_horizontal = np.array([[-1, -1, 0, 1, 1],
                                  [-2, -2, 0, 2, 2],
                                  [-4, -4, 0, 4, 4],
                                  [-2, -2, 0, 2, 2],
                                  [-1, -1, 0, 1, 1]])

    kernel_vertical = np.array([[-1, -2, -4, -2, -1],
                                [-1, -2, -4, -2, -1],
                                [0, 0, 0, 0, 0],
                                [1, 2, 4, 2, 1],
                                [1, 2, 4, 2, 1]])

    height, width = image.shape

    sobel_x_image = np.zeros_like(image, dtype=np.float32)
    sobel_y_image = np.zeros_like(image, dtype=np.float32)

    for i in range(2, height - 2):
        for j in range(2, width - 2):
            neighbors = image[i-2:i+3, j-2:j+3]

            gx = np.sum(neighbors * kernel_horizontal)
            sobel_x_image[i, j] = gx

            gy = np.sum(neighbors * kernel_vertical)
            sobel_y_image[i, j] = gy

    result_image = np.sqrt(sobel_x_image**2 + sobel_y_image**2)

    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    return Image.fromarray(result_image)
