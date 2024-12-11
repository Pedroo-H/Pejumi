from .erosion import apply_erosion
from .dilation import apply_dilation


def apply_opening(image, kernel_size):  # definimos o kernel_size como 3, 5 ou 7

    eroded_image = apply_erosion(image)
    opening_image = apply_dilation(eroded_image)

    return opening_image
