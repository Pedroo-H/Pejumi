import numpy as np

def apply_global_threshold(image):


    if len(image.shape) == 3:  
        image_array = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    else:
        image_array = image

    threshold_value = np.mean(image_array)
    thresholded_image = np.zeros_like(image_array)      

    while True:
        G1 = image_array > threshold_value     
        G2 = image_array <= threshold_value     

        m1 = np.mean(image_array[G1]) if np.any(G1) else 0 
        m2 = np.mean(image_array[G2]) if np.any(G2) else 0 
  
        new_threshold_value = (m1 + m2) / 2
        
        if abs(new_threshold_value - threshold_value) <= 0.1: 

            break
        
        threshold_value = new_threshold_value
    
    for i in range(image.shape[0]):  
        for j in range(image.shape[1]):  
            if image_array[i, j] >= threshold_value:
                thresholded_image[i, j] = 255           
            else:
                thresholded_image[i, j] = 0  

    return thresholded_image