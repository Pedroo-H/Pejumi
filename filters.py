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

    thresholded_image = np.zeros_like(image_array)   
    hist, bins = np.histogram(image_array, bins=256, range=(0, 256))  #histograma
    total_pixels = image_array.size

    sum_total = np.sum(np.arange(256) * hist)   #soma do produto intensidade * qnt pixels
    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold_value = 0

    for t in range(256):
        weight_background += hist[t]               # Peso do background
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background  # Peso do foreground
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        
        mean_background = sum_background / weight_background                    #media dos backgrounds
        mean_foreground = (sum_total - sum_background) / weight_foreground      #media dos foregrounds

        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2     #variancia entre classes

        if variance_between > max_variance:
            max_variance = variance_between     # Atualizar o threshold se a variância for a máxima encontrada
            threshold_value = t

    for i in range(image.shape[0]):  
        for j in range(image.shape[1]):  
            if image_array[i, j] >= threshold_value:
                thresholded_image[i, j] = 255           
            else:
                thresholded_image[i, j] = 0  

    return thresholded_image

def apply_erosion(image):

    thresholded_image = apply_global_threshold(image)                   #aplicando thresholding inicialmente                  

    binary_image = (thresholded_image > 0).astype(np.uint8) * 255      #converte imagem para binária (0 ou 255)

    kernel_cruz = np.array([[0, 1, 0],                      #matriz 3x3 de erosao em formato de cruz
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

    eroded_image = np.zeros_like(binary_image)              #matriz de saída

    for i in range(1, binary_image.shape[0] - 1):           #o -1 exclui o pixel da borda
        for j in range(1, binary_image.shape[1] - 1):
            region = binary_image[i-1:i+2, j-1:j+2]         #pega a regiao da imagem onde o kernel se sobrepoe 
            if np.all(region[kernel_cruz == 1] == 255):     #kernel_cruz é uma mascara booleana: checa se o pixel de valor 1 no kernel tem valor 255 na regiao selecionada
                eroded_image[i, j] = 255                            
            else:
                eroded_image[i, j] = 0
    
    return eroded_image

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
            if np.any(region[kernel_cruz == 1] == 255):         #aqui usamos o ANY (pelo menos um dos pixels é branco (255))
                dilated_image[i, j] = 255                            
            else:
                dilated_image[i, j] = 0
    
    return dilated_image