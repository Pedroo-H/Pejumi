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