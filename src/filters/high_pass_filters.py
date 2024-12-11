import numpy as np
from PIL import Image

def laplacian_filter(image, kernel_size):   #kernel_size sendo passado commo parametro podendo ser 3, 5 ou 7
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert('L')
    image = np.array(image)
    
    match kernel_size:

        case 3:
            kernel = np.array([[ 0,  0, -1],
                                [ 0, -1, -2],
                                [-1, -2,  4]])
        case 5:
            kernel = np.array([[ 0,   0,   1,   0,   0],
                                [ 0,   1,   2,   1,   0],
                                [ 1,   2,  -16,   2,   1],
                                [ 0,   1,   2,   1,   0],
                                [ 0,   0,   1,   0,   0]])
        case 7:
            kernel = np.array([[ 0,   0,   0,   0,   0,   0,   0],
                                [ 0,   0,   0,   0,   0,   0,   0],
                                [ 0,   0,   0,   0,   0,   0,   0],
                                [ 0,   0,   0,  -1,  -1,   0,   0],
                                [ 0,   0,   0,  -1,  24,  -1,   0],
                                [ 0,   0,   0,   0,   0,   0,   0],
                                [ 0,   0,   0,   0,   0,   0,   0]])
        case _:
            raise ValueError(f"Unsupported kernel size: {kernel_size}")


    height, width = image.shape
    result_image = np.zeros_like(image)

    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            neighbors = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            filtered_value = np.sum(neighbors * kernel)
            result_image[i - pad, j - pad] = np.clip(filtered_value, 0, 255)

    return Image.fromarray(result_image.astype(np.uint8))



def sobel_filter(image, scaling_factor):    #passando um fator de escala como parâmetro [sugestão 0.5, 1.0 e 2.0 ou seja: atenuada, normal e aumentada]
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
            sobel_x_image[i, j] = gx * scaling_factor

            gy = np.sum(neighbors * kernel_vertical)
            sobel_y_image[i, j] = gy * scaling_factor

    result_image = np.sqrt(sobel_x_image**2 + sobel_y_image**2)

    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    return Image.fromarray(result_image)
