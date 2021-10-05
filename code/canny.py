from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from code.utils import gaussian_kernel, get_convolved_image
from code.utils import sobel_derivative_kernel


def derivatives(image, derivative_kernel=sobel_derivative_kernel(), gaussian_cutoff_frequency=5):
    dx, dy = derivative_kernel
    g = gaussian_kernel(
        shape=(gaussian_cutoff_frequency * 4 + 1, gaussian_cutoff_frequency * 4 + 1),
        cutoff_frequency=gaussian_cutoff_frequency)
    gx = get_convolved_image(image=g, kernel=dx)
    gy = get_convolved_image(image=g, kernel=dy)
    image_x = get_convolved_image(image=image, kernel=gx)
    image_y = get_convolved_image(image=image, kernel=gy)
    return image_x, image_y


def canny():
    cut_off = 5
    bears = np.array(Image.open("../data/source/100075-original.jpg").convert('L'))
    plt.axis('off')
    plt.imshow(bears, cmap='gray')
    plt.show()
    image_x, image_y = derivatives(bears)
    # plt.axis('off')
    # plt.imshow(x, cmap='gray')
    # plt.show()
    #
    # plt.axis('off')
    # plt.imshow(y, cmap='gray')
    # plt.show()
    edge_strength = np.sqrt(np.power(image_x, 2) + np.power(image_y, 2))
    edge_direction = np.arctan(image_x / image_y)
    plt.axis('off')
    plt.imshow(edge_strength, cmap='gray')
    plt.show()

    plt.axis('off')
    plt.imshow(edge_direction, cmap='gray')
    plt.show()


if __name__ == '__main__':
    canny()
