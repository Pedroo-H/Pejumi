import numpy as np
from segmentation.global_threshold import apply_global_threshold


def apply_erosion(image, kernel_size):

    thresholded_image = apply_global_threshold(image, tolerance=0.5)
    binary_image = (thresholded_image > 0).astype(np.uint8) * 255

    match kernel_size:
        case 3:
            kernel_cruz = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        case 5:
            kernel_cruz = np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ],
                dtype=np.uint8,
            )
        case 7:
            kernel_cruz = np.array(
                [
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
        case _:
            raise ValueError("kernel_size deve ser 3, 5 ou 7.")

    eroded_image = np.zeros_like(binary_image)

    pad = kernel_size // 2
    padded_image = np.pad(binary_image, pad, mode="constant", constant_values=0)

    for i in range(pad, binary_image.shape[0] + pad):
        for j in range(pad, binary_image.shape[1] + pad):
            region = padded_image[i - pad : i + pad + 1, j - pad : j + pad + 1]
            if np.all(
                region[kernel_cruz == 1] == 255
            ):  # Verifica se todos os valores da região são 255
                eroded_image[i - pad, j - pad] = 255
            else:
                eroded_image[i - pad, j - pad] = 0

    return eroded_image
