from .erosion import apply_erosion
from .dilation import apply_dilation

def apply_closing(image, kernel_size):  #definimos o kernel_size como 3, 5 ou 7

    dilated_image = apply_dilation(image, kernel_size)             
    closing_image = apply_erosion(dilated_image, kernel_size)   

    return closing_image