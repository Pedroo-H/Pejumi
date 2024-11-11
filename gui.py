import tkinter as tk
from image_utils import load_image, convert_to_imageTk
from filters import (
    apply_gaussian_blur,
    apply_media_blur,
    apply_canny_edges,
    apply_sobel_filter,
    apply_global_threshold,
)

def create_gui():
    root = tk.Tk()
    root.title("Pejumi - O seu editor mais confiável (ou não)")

    image_frame = tk.Frame(root)
    image_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    global panel_original
    panel_original = tk.Label(image_frame)
    panel_original.pack(side="left", padx=10, pady=10)

    global panel_filtered
    panel_filtered = tk.Label(image_frame)
    panel_filtered.pack(side="right", padx=10, pady=10)


    btn_load = tk.Button(root, text="Carregar Imagem", command=load_and_display_image)
    btn_load.pack(side="top", fill="both", expand="yes", padx=10, pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    btn_gaussian = tk.Button(button_frame, text="Desfoque Gaussiano",
                              command=lambda: apply_filter(apply_gaussian_blur))
    btn_gaussian.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    btn_media = tk.Button(button_frame, text="Desfoque de Média",
                          command=lambda: apply_filter(apply_media_blur))
    btn_media.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    btn_canny = tk.Button(button_frame, text="Detecção de Bordas (Canny)",
                           command=lambda: apply_filter(apply_canny_edges))
    btn_canny.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    btn_sobel = tk.Button(button_frame, text="Filtro de Sobel",
                          command=lambda: apply_filter(apply_sobel_filter))
    btn_sobel.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    btn_global_thresholding = tk.Button(button_frame, text="Thresholding Global",
                          command=lambda: apply_filter(apply_global_threshold))
    btn_global_thresholding.pack(side="left", fill="both", expand=True, padx=5, pady=5)


    root.mainloop()


def load_and_display_image():
    global img, img_tk, panel_original
    img = load_image()
    if img is not None:
        img_tk = convert_to_imageTk(img)
        panel_original.config(image=img_tk)

def apply_filter(filter_function):
    global img, panel_filtered
    if img is not None:
        filtered_img = filter_function(img)
        img_filtered_tk = convert_to_imageTk(filtered_img)
        panel_filtered.config(image=img_filtered_tk)
        panel_filtered.image = img_filtered_tk 
