from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from code.utils import gaussian_kernel, get_convolved_image, argmax_2d
from code.utils import sobel_derivative_kernel
from copy import deepcopy


def derivatives(image, derivative_kernel=sobel_derivative_kernel(), gaussian_cutoff_frequency=3):
    dx, dy = derivative_kernel
    g = gaussian_kernel(
        shape=(gaussian_cutoff_frequency * 4 + 1, gaussian_cutoff_frequency * 4 + 1),
        cutoff_frequency=gaussian_cutoff_frequency)
    gx = get_convolved_image(image=g, kernel=dx)
    gy = get_convolved_image(image=g, kernel=dy)
    image_x = get_convolved_image(image=image, kernel=gx)
    image_y = get_convolved_image(image=image, kernel=gy)
    return image_x, image_y


def non_max_suppression_45_45(edge_strength, edge_direction):
    padded_image = np.pad(edge_strength, [(1,), (1,)])
    result = np.zeros_like(edge_strength)
    for i in range(1, edge_strength.shape[0] - 1):
        for j in range(1, edge_strength.shape[1] - 1):
            if (-22.5 <= edge_direction[i - 1, j - 1] <= 22.5) or (
                    157.5 < edge_direction[i - 1, j - 1] or edge_direction[i - 1, j - 1] < -157.5):
                side_1 = padded_image[i - 1, j]
                side_2 = padded_image[i + 1, j]
            elif (22.5 < edge_direction[i - 1, j - 1] <= 67.5) or (
                    -112.5 > edge_direction[i - 1, j - 1] >= -157.5):
                side_1 = padded_image[i - 1, j - 1]
                side_2 = padded_image[i + 1, j + 1]
            elif (67.5 < edge_direction[i - 1, j - 1] <= 112.5) or (
                    -67.5 > edge_direction[i - 1, j - 1] >= -112.5):
                side_1 = padded_image[i, j - 1]
                side_2 = padded_image[i, j + 1]
            elif (112.5 < edge_direction[i - 1, j - 1] <= 157.5) or (
                    -22.5 > edge_direction[i - 1, j - 1] >= -67.5):
                side_1 = padded_image[i - 1, j + 1]
                side_2 = padded_image[i + 1, j - 1]
            else:
                side_1 = None
                side_2 = None
            if (padded_image[i, j] > side_1) and (padded_image[i, j] > side_2):
                result[i, j] = padded_image[i, j]
    return result[1:-1, 1:-1]


def non_max_suppression_53_37(edge_strength, edge_direction):
    padded_image = np.pad(edge_strength, [(1,), (1,)])
    result = np.zeros_like(edge_strength)
    for i in range(1, edge_strength.shape[0] - 1):
        for j in range(1, edge_strength.shape[1] - 1):
            if (-26.5 <= edge_direction[i - 1, j - 1] <= 26.5) or (
                    153.5 < edge_direction[i - 1, j - 1] or edge_direction[i - 1, j - 1] < -153.5):
                side_1 = padded_image[i - 1, j]
                side_2 = padded_image[i + 1, j]
            elif (26.5 < edge_direction[i - 1, j - 1] <= 63.5) or (
                    -116.5 > edge_direction[i - 1, j - 1] >= -153.5):
                side_1 = padded_image[i - 1, j - 1]
                side_2 = padded_image[i + 1, j + 1]
            elif (63.5 < edge_direction[i - 1, j - 1] <= 116.5) or (
                    -63.5 > edge_direction[i - 1, j - 1] >= -116.5):
                side_1 = padded_image[i, j - 1]
                side_2 = padded_image[i, j + 1]
            elif (116.5 < edge_direction[i - 1, j - 1] <= 153.5) or (
                    -26.5 > edge_direction[i - 1, j - 1] >= -63.5):
                side_1 = padded_image[i - 1, j + 1]
                side_2 = padded_image[i + 1, j - 1]
            else:
                side_1 = None
                side_2 = None
            if (padded_image[i, j] > side_1) and (padded_image[i, j] > side_2):
                result[i, j] = padded_image[i, j]
    return result[1:-1, 1:-1]


def canny():
    bears = np.array(Image.open("../data/source/100075-original.jpg").convert('L'))
    # plt.axis('off')
    # plt.imshow(bears, cmap='gray')
    # plt.show()
    image_x, image_y = derivatives(bears)
    # plt.axis('off')
    # plt.imshow(x, cmap='gray')
    # plt.show()
    #
    # plt.axis('off')
    # plt.imshow(y, cmap='gray')
    # plt.show()
    edge_strength = np.sqrt(np.power(image_x, 2) + np.power(image_y, 2))
    edge_direction = np.degrees(np.arctan(image_x / image_y))
    plt.axis('off')
    es_slice = deepcopy(edge_strength[100:150, 100:150])
    ed_slice = deepcopy(edge_direction[100:150, 100:150])
    res = non_max_suppression_53_37(edge_strength=es_slice, edge_direction=ed_slice)
    plt.imshow(res, cmap='gray')
    plt.show()
    plt.axis('off')
    res = non_max_suppression_45_45(edge_strength=es_slice, edge_direction=ed_slice)
    plt.imshow(res, cmap='gray')
    plt.show()
    # plt.axis('off')
    # plt.imshow(edge_direction, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    canny()
