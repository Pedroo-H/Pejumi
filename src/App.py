import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
from filters.high_pass_filters import sobel_filter, laplacian_filter
from filters.low_pass_filters import apply_gaussian_blur, apply_media_blur
from morphological_operations import dilation, closing, erosion, opening
from segmentation import global_threshold, otsu_threshold
from MainMenu import Menu


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # window config
        ctk.set_appearance_mode("dark")
        self.geometry("1200x900")
        self.title("ProcImage")
        self.minsize(1000, 600)
        # variables
        self.original_image = None
        self.image_states = []
        self.current_bottom_frame = []
        # main menu config
        self._setup_menu()
        # img container and canvas config
        self.image_container = ctk.CTkFrame(self)
        self.image_container.pack(
            side="left", padx=10, pady=10, fill="both", expand=True
        )
        self.image_canvas = ctk.CTkCanvas(
            self.image_container, bg="#404040", bd=5, highlightthickness=0
        )
        self.image_canvas.pack(fill="both", expand=True, padx=5, pady=5)

    def _setup_menu(self):
        self.main_menu = Menu(
            master=self,
            load_image=self.load_image,
            apply_filter=self.apply_filter_bottom,
            morphological_operations=self.apply_morphological_operations,
            segmentation=self.apply_segmentation,
        )
        self.main_menu.pack(side="top", padx=10, pady=10, fill="x", expand=False)

    def _setup_img_operations_menu(self):
        self.image_menu = ctk.CTkFrame(self.image_container)
        self.image_menu.pack(padx=5, pady=5)

        icon = Image.open("icons/undo.png")
        r, g, b, a = icon.split()
        r = ImageOps.invert(r)
        g = ImageOps.invert(g)
        b = ImageOps.invert(b)
        inverted_icon = Image.merge("RGBA", (r, g, b, a))
        resized_image = inverted_icon.resize((22, 22))
        tk_icon = ctk.CTkImage(resized_image)

        self.state_back = ctk.CTkButton(
            self.image_menu, image=tk_icon, text="", width=24, command=self.undo
        )
        self.state_back.grid(row=0, column=0, padx=5, pady=5)

        self.back_to_original = ctk.CTkButton(self.image_menu, text="Original")
        self.back_to_original.grid(row=0, column=1, padx=5, pady=5)
        self.back_to_original.bind(
            "<ButtonPress-1>", lambda event: self.show_original_image()
        )
        self.back_to_original.bind(
            "<ButtonRelease-1>", lambda event: self.restore_last_image()
        )

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;")]
        )
        if file_path:
            if hasattr(self, "image_menu"):
                self.image_menu.pack_forget()
                del self.image_menu
            img = Image.open(file_path)
            image_tk = ImageTk.PhotoImage(img.convert("RGB"))
            self.original_image = (img.convert("RGB"), image_tk)
            self.image_states = []
            self.image_states.append(self.original_image)
            self.image_canvas.create_image(
                0, 0, image=self.image_states[-1][1], anchor="nw"
            )
            self._setup_img_operations_menu()

    def show_original_image(self):
        self.image_canvas.create_image(0, 0, image=self.original_image[1], anchor="nw")

    def restore_last_image(self):
        self.image_canvas.create_image(
            0, 0, image=self.image_states[-1][1], anchor="nw"
        )

    def undo(self):
        if self.image_states and self.image_states[-1] != self.original_image:
            self.image_states.pop()
            self.image_canvas.create_image(
                0, 0, image=self.image_states[-1][1], anchor="nw"
            )

    def apply_filter_bottom(self):
        self._clear_previous_bottom_frame()

        # high pass filters container
        self.high_pass_container = ctk.CTkFrame(self)
        self.high_pass_container.pack(
            side="bottom", padx=10, pady=10, fill="both", expand=True
        )
        # label
        self.high_pass_label = ctk.CTkLabel(
            self.high_pass_container, text="High-pass Filters", fg_color="transparent"
        )
        self.high_pass_label.grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="n"
        )
        # laplacian button
        self.laplaciano_button = ctk.CTkButton(
            self.high_pass_container,
            text="Laplacian",
            command=lambda: self.apply_option("laplacian"),
        )
        self.laplaciano_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        # sobel button
        self.sobel_button = ctk.CTkButton(
            self.high_pass_container,
            text="Sobel",
            command=lambda: self.apply_option("sobel"),
        )
        self.sobel_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        # slider
        self.slider_high_pass = ctk.CTkSlider(
            self.high_pass_container, from_=3, to=7, number_of_steps=2
        )
        self.slider_high_pass.set(5)
        self.slider_high_pass.grid(
            row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )

        self.high_pass_container.grid_columnconfigure(0, weight=2)

        # low pass filters container
        self.low_pass_container = ctk.CTkFrame(self)
        self.low_pass_container.pack(
            side="bottom", padx=10, pady=10, fill="both", expand=True
        )
        # label
        self.low_pass_label = ctk.CTkLabel(
            self.low_pass_container, text="Low-pass Filters", fg_color="transparent"
        )
        self.low_pass_label.grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="n"
        )
        # gaussian blur button
        self.gaussian_blur_button = ctk.CTkButton(
            self.low_pass_container,
            text="Gaussian blur",
            command=lambda: self.apply_option("gaussian"),
        )
        self.gaussian_blur_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        # media blur button
        self.media_blur_button = ctk.CTkButton(
            self.low_pass_container,
            text="Media blur",
            command=lambda: self.apply_option("media"),
        )
        self.media_blur_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        # slider
        self.slider_low_pass = ctk.CTkSlider(
            self.low_pass_container, from_=3, to=7, number_of_steps=2
        )
        self.slider_low_pass.set(5)
        self.slider_low_pass.grid(
            row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )

        self.low_pass_container.grid_columnconfigure(0, weight=2)

        self.current_bottom_frame = [self.high_pass_container, self.low_pass_container]

    def apply_morphological_operations(self):
        self._clear_previous_bottom_frame()

        # morph operations container
        self.morph_container = ctk.CTkFrame(self)
        self.morph_container.pack(
            side="bottom", padx=10, pady=10, fill="both", expand=True
        )

        self.morph_container.grid_columnconfigure(0, weight=3)

        # label
        self.morph_label = ctk.CTkLabel(
            self.morph_container,
            text="Morphological Operations",
            fg_color="transparent",
        )
        self.morph_label.grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="n"
        )
        # closing button
        self.closing_button = ctk.CTkButton(
            self.morph_container,
            text="Closing",
            command=lambda: self.apply_option("closing"),
        )
        self.closing_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        # dilation button
        self.dilation_button = ctk.CTkButton(
            self.morph_container,
            text="Dilation",
            command=lambda: self.apply_option("dilation"),
        )
        self.dilation_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        # erosion button
        self.erosion_button = ctk.CTkButton(
            self.morph_container,
            text="Erosion",
            command=lambda: self.apply_option("erosion"),
        )
        self.erosion_button.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        # opening button
        self.opening_button = ctk.CTkButton(
            self.morph_container,
            text="Opening",
            command=lambda: self.apply_option("opening"),
        )
        self.opening_button.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        # slider
        self.slider_morph = ctk.CTkSlider(
            self.morph_container, from_=3, to=7, number_of_steps=2
        )
        self.slider_morph.set(5)
        self.slider_morph.grid(
            row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )

        self.current_bottom_frame = [self.morph_container]

    def apply_segmentation(self):
        self._clear_previous_bottom_frame()

        # segmentation_container
        self.segmentation_container = ctk.CTkFrame(self)
        self.segmentation_container.pack(
            side="bottom", padx=10, pady=10, fill="both", expand=True
        )

        self.segmentation_container.grid_columnconfigure(0, weight=3)

        # label
        self.segmentation_label = ctk.CTkLabel(
            self.segmentation_container, text="Segmentation", fg_color="transparent"
        )
        self.segmentation_label.grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="n"
        )
        # global button
        self.global_threshold_button = ctk.CTkButton(
            self.segmentation_container,
            text="Global threshold",
            command=lambda: self.apply_option("global"),
        )
        self.global_threshold_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        # otsu burron
        self.otsu_threshold_button = ctk.CTkButton(
            self.segmentation_container,
            text="Otsu threshold",
            command=lambda: self.apply_option("otsu"),
        )
        self.otsu_threshold_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        # slider
        self.slider_segmentation = ctk.CTkSlider(
            self.segmentation_container, from_=3, to=7, number_of_steps=2
        )
        self.slider_segmentation.set(5)
        self.slider_segmentation.grid(
            row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )

        self.current_bottom_frame = [self.segmentation_container]

    def _clear_previous_bottom_frame(self):
        for frame in self.current_bottom_frame:
            frame.pack_forget()

    # function that in fact runs the called functions
    def apply_option(self, option):
        match option:
            case "laplacian":
                kernel_size = self.slider_high_pass.get()
                filtered_image = laplacian_filter(
                    self.image_states[-1][0], int(kernel_size)
                )
            case "sobel":
                scaling_value = self.slider_high_pass.get()
                if scaling_value == 3.0:
                    scaling_value = 0.5
                if scaling_value == 5.0:
                    scaling_value == 1.0
                if scaling_value == 7.0:
                    scaling_value = 2.0
                filtered_image = sobel_filter(self.image_states[-1][0], scaling_value)
            case "gaussian":
                kernel_size = self.slider_low_pass.get()
                filtered_image = apply_gaussian_blur(
                    self.image_states[-1][0], int(kernel_size)
                )
            case "media":
                scaling_value = self.slider_low_pass.get()
                filtered_image = apply_media_blur(
                    self.image_states[-1][0], int(scaling_value)
                )
            case "closing":
                kernel_size = self.slider_morph.get()
                image_array = np.array(self.image_states[-1][0])
                filtered_image = closing.apply_closing(image_array, int(kernel_size))
            case "dilation":
                kernel_size = self.slider_morph.get()
                image_array = np.array(self.image_states[-1][0])
                filtered_image = dilation.apply_dilation(image_array, int(kernel_size))
            case "erosion":
                scaling_value = self.slider_morph.get()
                image_array = np.array(self.image_states[-1][0])
                filtered_image = erosion.apply_erosion(image_array, int(scaling_value))
            case "opening":
                scaling_value = self.slider_morph.get()
                image_array = np.array(self.image_states[-1][0])
                filtered_image = opening.apply_dilation(image_array, int(scaling_value))
            case "global":
                scaling_value = self.slider_segmentation.get()
                if scaling_value == 3.0:
                    scaling_value = 0.01
                if scaling_value == 5.0:
                    scaling_value == 0.5
                if scaling_value == 7.0:
                    scaling_value = 1.0
                image_array = np.array(self.image_states[-1][0])
                filtered_image = global_threshold.apply_global_threshold(
                    image_array, scaling_value
                )
            case "otsu":
                scaling_value = self.slider_segmentation.get()
                if scaling_value == 3.0:
                    scaling_value = 0.05
                if scaling_value == 5.0:
                    scaling_value = 1.0
                if scaling_value == 7.0:
                    scaling_value = 1.2
                image_array = np.array(self.image_states[-1][0])
                filtered_image = otsu_threshold.apply_otsu_threshold(
                    image_array, scaling_value
                )
            case _:
                pass

        if isinstance(filtered_image, np.ndarray):
            filtered_image = Image.fromarray(filtered_image)

        filtered_image = filtered_image.convert("RGB")
        filtered_image_tk = ImageTk.PhotoImage(filtered_image)
        self.image_states.append((filtered_image, filtered_image_tk))
        self.image_canvas.create_image(
            0, 0, image=self.image_states[-1][1], anchor="nw"
        )


if __name__ == "__main__":
    app = App()
    app.mainloop()
