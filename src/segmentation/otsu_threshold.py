import numpy as np


def apply_otsu_threshold(
    image, scaling_factor
):  # fator escalar como parametro da funcao

    if len(image.shape) == 3:
        image_array = (
            0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        )
    else:
        image_array = image

    thresholded_image = np.zeros_like(image_array)
    hist, bins = np.histogram(image_array, bins=256, range=(0, 256))  # histograma
    total_pixels = image_array.size

    sum_total = np.sum(
        np.arange(256) * hist
    )  # soma do produto intensidade * qnt pixels
    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold_value = 0

    for t in range(256):
        weight_background += hist[t]  # Peso do background
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background  # Peso do foreground
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]

        mean_background = sum_background / weight_background  # media dos backgrounds
        mean_foreground = (
            sum_total - sum_background
        ) / weight_foreground  # media dos foregrounds

        variance_between = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )  # variancia entre classes

        if variance_between > max_variance:
            max_variance = variance_between  # Atualizar o threshold se a variância for a máxima encontrada
            threshold_value = t

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            adjusted_threshold = (
                threshold_value * scaling_factor
            )  # fator multplicativo para ajustar sensibilidade [<1.0 = mais sensível;  >1.0 = menos sensível]
            if image_array[i, j] >= adjusted_threshold:
                thresholded_image[i, j] = 255
            else:
                thresholded_image[i, j] = 0

    return thresholded_image
