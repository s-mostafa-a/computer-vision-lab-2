from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from code.utils import gaussian_kernel, get_convolved_image, argmax_2d
from code.utils import sobel_derivative_kernel
from scipy.ndimage import label


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


def non_max_suppression(edge_strength, edge_direction, alpha):
    assert 0 < alpha < 90, "alpha is a degree between 0 and 90 (non-inclusive)"
    beta = 90 - alpha
    padded_image = np.pad(edge_strength, [(1,), (1,)])
    result = np.zeros_like(edge_strength)
    for i in range(1, edge_strength.shape[0] - 1):
        for j in range(1, edge_strength.shape[1] - 1):
            if (-alpha / 2 <= edge_direction[i - 1, j - 1] <= alpha / 2) or (
                    180 - alpha / 2 < edge_direction[i - 1, j - 1] or
                    edge_direction[i - 1, j - 1] < -180 + alpha / 2):
                side_1 = padded_image[i - 1, j]
                side_2 = padded_image[i + 1, j]
            elif (alpha / 2 < edge_direction[i - 1, j - 1] <= beta + alpha / 2) or (
                    -180 + beta + alpha / 2 > edge_direction[i - 1, j - 1] >= -180 + alpha / 2):
                side_1 = padded_image[i - 1, j - 1]
                side_2 = padded_image[i + 1, j + 1]
            elif (beta + alpha / 2 < edge_direction[i - 1, j - 1] <= beta + 3 * alpha / 2) or (
                    -180 + beta + 3 * alpha / 2 > edge_direction[
                i - 1, j - 1] >= -180 + beta + alpha / 2):
                side_1 = padded_image[i, j - 1]
                side_2 = padded_image[i, j + 1]
            elif (beta + 3 * alpha / 2 < edge_direction[
                i - 1, j - 1] <= 2 * beta + 3 * alpha / 2) or (
                    -180 + 2 * beta + 3 * alpha / 2 > edge_direction[
                i - 1, j - 1] >= -180 + beta + 3 * alpha / 2):
                side_1 = padded_image[i - 1, j + 1]
                side_2 = padded_image[i + 1, j - 1]
            else:
                side_1 = None
                side_2 = None
            if (padded_image[i, j] > side_1) and (padded_image[i, j] > side_2):
                result[i, j] = padded_image[i, j]
    return result[1:-1, 1:-1]


def threshold(image, low, high, strong=255, weak=100):
    result = np.zeros(image.shape)
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image < high) & (image >= low))
    result[strong_row, strong_col] = strong
    result[weak_row, weak_col] = weak
    return result


def hysteresis(image, strong=255):
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(image, structure=structure)
    for lbl in range(1, num_features + 1):
        row, col = np.where(labeled_array == lbl)
        if np.max(image[row, col]) != strong:
            image[row, col] = .0
        else:
            image[row, col] = strong
    return image


def canny():
    bears = np.array(Image.open("../data/source/100075-original.jpg").convert('L'))
    image_x, image_y = derivatives(bears)
    edge_strength = np.sqrt(np.power(image_x, 2) + np.power(image_y, 2))
    edge_direction = np.degrees(np.arctan(image_x / image_y))
    plt.hist(edge_strength.ravel(), bins=256)
    plt.show()
    res = non_max_suppression(edge_strength=edge_strength, edge_direction=edge_direction, alpha=53)
    res = threshold(res, 25, 50)
    res = hysteresis(res)
    plt.axis('off')
    plt.imshow(res, cmap='gray')
    plt.show()


if __name__ == '__main__':
    canny()
