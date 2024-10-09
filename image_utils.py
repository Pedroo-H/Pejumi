import cv2
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk

def load_image(max_width=800, max_height=800):
    path = filedialog.askopenfilename()
    if path:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        scale = min(max_width / width, max_height / height)
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size)
        return img
    return None

def convert_to_imageTk(image):
    return ImageTk.PhotoImage(image=Image.fromarray(image))
