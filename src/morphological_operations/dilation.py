import numpy as np
from segmentation.global_threshold import apply_global_threshold

def apply_dilation(image):

    thresholded_image = apply_global_threshold(image)                                  

    binary_image = (thresholded_image > 0).astype(np.uint8) * 255      
    kernel_cruz = np.array([[0, 1, 0],                     
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

    dilated_image = np.zeros_like(binary_image)              

    for i in range(1, binary_image.shape[0] - 1):           
        for j in range(1, binary_image.shape[1] - 1):
            region = binary_image[i-1:i+2, j-1:j+2]        
            if np.any(region[kernel_cruz == 1] == 255): 
                dilated_image[i, j] = 255                            
            else:
                dilated_image[i, j] = 0
    
    return dilated_image