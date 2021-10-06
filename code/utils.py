import numpy as np


def _assertion_checks(two_d_image, kernel, mode):
    assert len(two_d_image.shape) == 2, "The image must be a 2d image!"
    assert len(kernel.shape) == 2, "The kernel must be 2d!"
    assert mode in ("valid", "same"), "The mode parameter can only be `valid` or `same`"
    assert kernel.shape[0] % 2 == 1 and kernel.shape[
        1] % 2 == 1, "kernel shape must be a tuple of odd numbers!"


def _get_paddings_and_new_image(two_d_image, kernel_shape, mode):
    padding_i = kernel_shape[0] // 2
    padding_j = kernel_shape[1] // 2
    if mode == "same":
        two_d_image = np.pad(two_d_image, [(padding_i,), (padding_j,)])
    return padding_i, padding_j, two_d_image


def _convolution(two_d_image: np.array, kernel: np.array, mode: str = 'same'):
    _assertion_checks(two_d_image=two_d_image, kernel=kernel, mode=mode)
    padding_i, padding_j, two_d_image = _get_paddings_and_new_image(two_d_image=two_d_image,
                                                                    kernel_shape=kernel.shape,
                                                                    mode=mode)
    shape = tuple([two_d_image.shape[0] - 2 * padding_i, two_d_image.shape[1] - 2 * padding_j] +
                  list(kernel.shape))
    multi_dim_image = np.empty(shape=shape, dtype=two_d_image.dtype)
    for i in range(padding_i, two_d_image.shape[0] - padding_i):
        for j in range(padding_j, two_d_image.shape[1] - padding_j):
            multi_dim_image[i - padding_i, j - padding_j] = two_d_image[
                                                            i - padding_i:i + padding_i + 1,
                                                            j - padding_j:j + padding_j + 1]

    expanded_kernel = np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)

    final_image = np.sum((multi_dim_image * expanded_kernel), axis=(2, 3))
    return final_image


def get_convolved_image(image, kernel):
    result_image = np.empty(shape=image.shape, dtype=int)
    if len(image.shape) == 3:
        for ch in range(3):
            result_image[:, :, ch] = _convolution(two_d_image=image[:, :, ch],
                                                  kernel=kernel)
    elif len(image.shape) == 2:
        result_image = _convolution(two_d_image=image, kernel=kernel, mode='same')
    else:
        raise Exception
    return result_image


def gaussian_kernel(shape=(5, 5), cutoff_frequency=1.):
    assert len(shape) == 2, "kernel must be 2d!"
    assert shape[0] % 2 == 1 and shape[1] % 2 == 1
    ax1 = np.linspace(-(shape[0] - 1) / 2., (shape[0] - 1) / 2., shape[0])
    ax2 = np.linspace(-(shape[1] - 1) / 2., (shape[1] - 1) / 2., shape[1])
    gauss1 = np.exp(-0.5 * np.square(ax1) / np.square(cutoff_frequency))
    gauss2 = np.exp(-0.5 * np.square(ax2) / np.square(cutoff_frequency))
    kernel = np.outer(gauss1, gauss2)
    return kernel / np.sum(kernel)


def box_kernel(shape=(3, 3)):
    kernel = np.ones(shape)
    return kernel / np.sum(kernel)


def sobel_derivative_kernel():
    dx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    dy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    return dx, dy


def prewitt_derivative_kernel():
    dx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    dy = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])
    return dx, dy


def roberts_derivative_kernel():
    dx = np.array([[0, 1],
                   [-1, 0]])
    dy = np.array([[1, 0],
                   [0, -1]])
    return dx, dy


def argmax_2d(img: np.array):
    max1 = np.max(img, axis=0)
    argmax1 = np.argmax(img, axis=0)
    argmax2 = np.argmax(max1, axis=0)
    argmax2d = (argmax1[argmax2], argmax2)
    return argmax2d
