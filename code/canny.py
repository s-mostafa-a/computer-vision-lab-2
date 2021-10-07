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
    result = np.zeros_like(padded_image)
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
            image[row, col] = 0.
        else:
            image[row, col] = 1.
    return image


def show_distribution_histogram(image):
    plt.hist(image.ravel(), bins=256)
    plt.show()


def canny(image, weak=128, strong=255, cutoff_frequency=3, alpha=53, low=25):
    high = 2 * low
    image_x, image_y = derivatives(image, gaussian_cutoff_frequency=cutoff_frequency)
    edge_strength = np.sqrt(np.power(image_x, 2) + np.power(image_y, 2))
    edge_direction = np.degrees(np.arctan(image_x / image_y))
    result = non_max_suppression(edge_strength=edge_strength, edge_direction=edge_direction,
                                 alpha=alpha)
    result = threshold(result, low=low, high=high, weak=weak, strong=strong)
    result = hysteresis(result, strong=strong)
    return result


def quality_assessment(detection, ground_truth):
    assert np.min(ground_truth) == 0 and np.max(ground_truth) == 1 and 0 <= np.min(
        detection) <= 1 and 0 <= np.max(detection) <= 1, f"""{np.min(ground_truth)}, {
    np.max(ground_truth)}, {np.min(detection)}, {np.max(detection)}"""
    ground_truth_inverse = 1 - ground_truth
    tpr = np.sum(detection * ground_truth) / np.sum(ground_truth)
    fpr = np.sum(detection * ground_truth_inverse) / np.sum(ground_truth_inverse)
    return tpr, fpr


def loop_for_training(image, ground_truth, save_plots=True):
    stn = 255
    wk = 128
    ground_truth = np.invert(ground_truth)
    ground_truth[ground_truth > wk] = stn
    ground_truth[ground_truth <= wk] = 0
    ground_truth = ground_truth / stn

    cut_offs = [17, 15, 13, 11, 9, 7, 5, 3, 1]
    alphas = [45, 53, 60]
    lows = [i for i in range(10, 42, 2)]
    for cut_off in cut_offs:
        for alpha in alphas:
            roc_ys = []
            roc_xs = []
            for low in lows:
                detection = canny(image=image, weak=wk, strong=stn, cutoff_frequency=cut_off,
                                  alpha=alpha, low=low)
                roc_y, roc_x = quality_assessment(detection=detection, ground_truth=ground_truth)
                roc_xs.append(roc_x)
                roc_ys.append(roc_y)
            plt.plot(roc_xs, roc_ys)
            plt.title(f'cut_off={cut_off}, alpha={alpha}')
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            if save_plots:
                plt.savefig(f"../data/result/plots/cutoff_{cut_off}__alpha_{alpha}.jpeg")
            plt.show()


if __name__ == '__main__':
    img = np.array(Image.open("../data/source/100075-original.jpg").convert('L'))
    gnd_trt = np.array(Image.open("../data/source/100075-reference.jpg").convert('L'))
    loop_for_training(image=img, ground_truth=gnd_trt)
