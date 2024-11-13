from .erosion import apply_erosion
from .dilation import apply_dilation

def apply_closing(image):

    dilated_image = apply_dilation(image)             
    closing_image = apply_erosion(dilated_image)   

    return closing_image