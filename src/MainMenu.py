import customtkinter as ctk
from PIL import Image, ImageOps


class Menu(ctk.CTkFrame):
    def __init__(
        self, master, load_image, apply_filter, morphological_operations, segmentation
    ):
        super().__init__(master)

        # functions config
        self.load_image = load_image
        self.apply_filter = apply_filter
        self.segmentation = segmentation
        self.morphological_operations = morphological_operations

        # icon import and adaptation
        icon = Image.open("icons/load.png")
        r, g, b, a = icon.split()
        r = ImageOps.invert(r)
        g = ImageOps.invert(g)
        b = ImageOps.invert(b)
        inverted_icon = Image.merge("RGBA", (r, g, b, a))
        resized_image = inverted_icon.resize((22, 22))
        tk_icon = ctk.CTkImage(resized_image)

        # load button
        self.load_button = ctk.CTkButton(
            self, image=tk_icon, text="", width=24, command=self.load_image
        )
        self.load_button.grid(row=0, column=0, padx=5, pady=5)
        # filter button
        self.filter_button = ctk.CTkButton(
            self, text="Apply filter", command=self.apply_filter
        )
        self.filter_button.grid(row=0, column=1, padx=5, pady=5)
        # morphological operations button
        self.morph_button = ctk.CTkButton(
            self, text="Morphological Operations", command=self.morphological_operations
        )
        self.morph_button.grid(row=0, column=2, padx=5, pady=5)
        # segmentation button
        self.segment_button = ctk.CTkButton(
            self, text="Image Segmentation", command=self.segmentation
        )
        self.segment_button.grid(row=0, column=3, padx=5, pady=5)
