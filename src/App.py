import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from filters.high_pass_filters import *
from filters.low_pass_filters import apply_gaussian_blur, apply_media_blur
from morphological_operations import dilation, closing, erosion, opening
from segmentation import global_threshold, otsu_threshold


class Menu(ctk.CTkFrame):
    def __init__(self, master, load_image, apply_filter, morphological_operations, segmentation):
        super().__init__(master)

        self.load_image = load_image
        self.apply_filter = apply_filter
        self.segmentation = segmentation
        self.morphological_operations = morphological_operations

        self.load_button = ctk.CTkButton(self, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=10, pady=5)

        self.filter_button = ctk.CTkButton(self, text="Apply filter", command=self.apply_filter)
        self.filter_button.grid(row=0, column=1, padx=10, pady=5)

        self.morph_button = ctk.CTkButton(self, text="Morphological Operations", command=self.morphological_operations)
        self.morph_button.grid(row=0, column=2, padx=10, pady=5)

        self.segment_button = ctk.CTkButton(self, text="Image Segmentation", command=self.segmentation)
        self.segment_button.grid(row=0, column=3, padx=10, pady=5)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode('dark')
        self.geometry('1500x900')
        self.title('ProcImage')
        self.minsize(1000, 600)

        self.image = None
        self.filtered_image = None
        self.current_bottom_frame = []

        self.menu_container = Menu(
            master=self,
            load_image=self.load_image,
            apply_filter=self.apply_filter_bottom,
            morphological_operations=self.apply_morphological_operations,
            segmentation=self.apply_segmentation
        )
        self.menu_container.pack(padx=10, pady=10, fill="x", expand=False)

        self.images_container = ctk.CTkFrame(self)
        self.images_container.pack(padx=10, pady=10, fill="both", expand=True)

        self.original_canvas = ctk.CTkCanvas(self.images_container, bg="#404040", bd=5, highlightthickness=0)
        self.original_canvas.pack(side="left", fill="both", expand=True, padx=15, pady=15)

        self.filtered_canvas = ctk.CTkCanvas(self.images_container, bg="#404040", bd=5, highlightthickness=0)
        self.filtered_canvas.pack(side="right", fill="both", expand=True, padx=15, pady=15)


    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png"), ("Images", "*.jpg"), ("Images", "*.jpeg"), ("Images", "*.bmp"), ("Images", "*.gif")])
        if file_path:
            img = Image.open(file_path)
            self.image = img.convert("RGB")
            image_tk = ImageTk.PhotoImage(self.image)
            self.original_canvas.create_image(0, 0, image=image_tk, anchor="nw")
            self.original_canvas.image = image_tk


    def apply_filter_bottom(self):
        self._clear_previous_bottom_frame()

        self.bottom_container_1 = ctk.CTkFrame(self)
        self.bottom_container_1.pack(side='left', padx=10, pady=10, fill="x", expand=True)

        self.high_pass_label = ctk.CTkLabel(self.bottom_container_1, text="High-pass Filters", fg_color="transparent")
        self.high_pass_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky='n')

        self.laplaciano_button = ctk.CTkButton(self.bottom_container_1, text="Laplacian", command=lambda: self.apply_option('laplacian'))
        self.laplaciano_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.sobel_button = ctk.CTkButton(self.bottom_container_1, text="Sobel", command=lambda: self.apply_option('sobel'))
        self.sobel_button.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        self.slider_high_pass = ctk.CTkSlider(self.bottom_container_1, from_=0, to=10, number_of_steps=10)
        self.slider_high_pass.set(5)
        self.slider_high_pass.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.bottom_container_1.grid_columnconfigure(0, weight=1)
        self.bottom_container_1.grid_columnconfigure(1, weight=1)

        self.bottom_container_2 = ctk.CTkFrame(self)
        self.bottom_container_2.pack(side='left', padx=10, pady=10, fill="x", expand=True)

        self.low_pass_label = ctk.CTkLabel(self.bottom_container_2, text="Low-pass Filters", fg_color="transparent")
        self.low_pass_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky='n')

        self.gaussian_blur_button = ctk.CTkButton(self.bottom_container_2, text="Gaussian blur", command=lambda: self.apply_option('gaussian'))
        self.gaussian_blur_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.media_blur_button = ctk.CTkButton(self.bottom_container_2, text="Media blur", command=lambda: self.apply_option('media'))
        self.media_blur_button.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        self.slider_low_pass = ctk.CTkSlider(self.bottom_container_2, from_=0, to=10, number_of_steps=10)
        self.slider_low_pass.set(5)
        self.slider_low_pass.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.bottom_container_2.grid_columnconfigure(0, weight=1)
        self.bottom_container_2.grid_columnconfigure(1, weight=1)

        self.current_bottom_frame = [self.bottom_container_1, self.bottom_container_2]


    def apply_morphological_operations(self):
        self._clear_previous_bottom_frame()

        self.bottom_container = ctk.CTkFrame(self)
        self.bottom_container.pack(side='left', padx=10, pady=10, fill="x", expand=True)

        self.bottom_container.grid_rowconfigure(0, weight=1)
        self.bottom_container.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="group")

        self.closing_button = ctk.CTkButton(self.bottom_container, text="Closing", command=lambda: self.apply_option('closing'))
        self.closing_button.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

        self.dilation_button = ctk.CTkButton(self.bottom_container, text="Dilation", command=lambda: self.apply_option('dilation'))
        self.dilation_button.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

        self.erosion_button = ctk.CTkButton(self.bottom_container, text="Erosion", command=lambda: self.apply_option('erosion'))
        self.erosion_button.grid(row=0, column=2, padx=10, pady=5, sticky='ew')

        self.opening_button = ctk.CTkButton(self.bottom_container, text="Opening", command=lambda: self.apply_option('opening'))
        self.opening_button.grid(row=0, column=3, padx=10, pady=5, sticky='ew')

        self.slider_morph = ctk.CTkSlider(self.bottom_container, from_=0, to=10, number_of_steps=10)
        self.slider_morph.set(5)
        self.slider_morph.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        self.bottom_container.grid_columnconfigure(0, weight=1)
        self.bottom_container.grid_columnconfigure(1, weight=1)
        self.bottom_container.grid_columnconfigure(2, weight=1)
        self.bottom_container.grid_columnconfigure(3, weight=1)

        self.current_bottom_frame = [self.bottom_container]


    def apply_segmentation(self):
        self._clear_previous_bottom_frame()

        self.bottom_container = ctk.CTkFrame(self)
        self.bottom_container.pack(side='left', padx=10, pady=10, fill="x", expand=True)

        self.bottom_container.grid_rowconfigure(0, weight=1)
        self.bottom_container.grid_columnconfigure((0, 1), weight=1, uniform="group")

        self.global_threshold_button = ctk.CTkButton(self.bottom_container, text="Global threshold", command=lambda: self.apply_option('global'))
        self.global_threshold_button.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

        self.otsu_threshold_button = ctk.CTkButton(self.bottom_container, text="Otsu threshold", command=lambda: self.apply_option('otsu'))
        self.otsu_threshold_button.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

        self.slider_segmentation = ctk.CTkSlider(self.bottom_container, from_=0, to=10, number_of_steps=10)
        self.slider_segmentation.set(5)
        self.slider_segmentation.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        self.current_bottom_frame = [self.bottom_container]


    def _clear_previous_bottom_frame(self):
        for frame in self.current_bottom_frame:
            frame.pack_forget()


    def apply_option(self, option):
        match option:
            case 'laplacian':
                filtered_image = laplacian_filter(self.image)
            case 'sobel':
                filtered_image = sobel_filter(self.image)
            case 'gaussian':
                filtered_image = apply_gaussian_blur(self.image)
            case 'media':
                filtered_image = apply_media_blur(self.image)
            case 'closing':
                image_array = np.array(self.image)
                filtered_image = closing.apply_closing(image_array)
            case 'dilation':
                image_array = np.array(self.image)
                filtered_image = dilation.apply_dilation(image_array)
            case 'erosion':
                image_array = np.array(self.image)
                filtered_image = erosion.apply_erosion(image_array)
            case 'opening':
                image_array = np.array(self.image)
                filtered_image = opening.apply_dilation(image_array)
            case 'global':
                image_array = np.array(self.image)
                filtered_image = global_threshold.apply_global_threshold(image_array)
            case 'otsu':
                image_array = np.array(self.image)
                filtered_image = otsu_threshold.apply_otsu_threshold(image_array)
            case _:
                pass

        if isinstance(filtered_image, np.ndarray):
            filtered_image = Image.fromarray(filtered_image)

        self.filtered_image = ImageTk.PhotoImage(filtered_image)
        self.filtered_canvas.create_image(0, 0, image=self.filtered_image, anchor="nw")


if __name__ == "__main__":
    app = App()
    app.mainloop()
