from .erosion import apply_erosion
from .dilation import apply_dilation

def apply_opening(image):

    eroded_image = apply_erosion(image)            
    opening_image = apply_dilation(eroded_image)    

    return opening_image
