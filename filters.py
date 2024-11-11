import cv2
import numpy as np

def apply_gaussian_blur(image):
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]]) / 256.0  # Kernel Gaussiano
    return cv2.filter2D(image, -1, kernel)

def apply_media_blur(image):
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

def apply_canny_edges(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_sobel_filter(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.sqrt(cv2.addWeighted(sobel_x ** 2, 1, sobel_y ** 2, 1, 0))
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2RGB)

def apply_global_threshold(image):


    if len(image.shape) == 3:  
        image_array = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    else:
        image_array = image

    threshold_value = np.mean(image_array)
    thresholded_image = np.zeros_like(image_array)      

    while True:
        G1 = image_array > threshold_value      #matriz booleana (pixels > threshold)
        G2 = image_array <= threshold_value     #matriz booleana (pixels <= threshold)

        m1 = np.mean(image_array[G1]) if np.any(G1) else 0  #filtra valores da imagem onde o pixel > threshold e calcula a medica
        m2 = np.mean(image_array[G2]) if np.any(G2) else 0  #filtra valores da imagem onde o pixel <= threshold e calcula a media
  
        new_threshold_value = (m1 + m2) / 2     #novo limiar é a média dos dois grupos
        
        if abs(new_threshold_value - threshold_value) <= 0.1: #critério de parada do cálculo do threshold

            break
        
        threshold_value = new_threshold_value
        print("oi")
    
    for i in range(image.shape[0]):  
        for j in range(image.shape[1]):  
            if image_array[i, j] >= threshold_value:
                thresholded_image[i, j] = 255           
            else:
                thresholded_image[i, j] = 0  

    return thresholded_image

def apply_otsu_threshold(image):


    if len(image.shape) == 3:  
        image_array = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    else:
        image_array = image

    threshold_value = np.mean(image_array)
    thresholded_image = np.zeros_like(image_array)      

    while True:
        G1 = image_array > threshold_value      #matriz booleana (pixels > threshold)
        G2 = image_array <= threshold_value     #matriz booleana (pixels <= threshold)

        m1 = np.mean(image_array[G1]) if np.any(G1) else 0  #filtra valores da imagem onde o pixel > threshold e calcula a medica
        m2 = np.mean(image_array[G2]) if np.any(G2) else 0  #filtra valores da imagem onde o pixel <= threshold e calcula a media
  
        new_threshold_value = (m1 + m2) / 2     #novo limiar é a média dos dois grupos
        
        if abs(new_threshold_value - threshold_value) <= 0.1: #critério de parada do cálculo do threshold

            break
        
        threshold_value = new_threshold_value
        print("oi")
    
    for i in range(image.shape[0]):  
        for j in range(image.shape[1]):  
            if image_array[i, j] >= threshold_value:
                thresholded_image[i, j] = 255           
            else:
                thresholded_image[i, j] = 0  

    return thresholded_image
