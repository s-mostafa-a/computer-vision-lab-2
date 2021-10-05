import numpy as np


def _assertion_checks(two_d_image, flt, mode):
    assert len(two_d_image.shape) == 2, "The image must be a 2d image!"
    assert len(flt.shape) == 2, "The filter must be 2d!"
    assert mode in ("valid", "same"), "The mode parameter can only be `valid` or `same`"
    assert flt.shape[0] % 2 == 1 and flt.shape[
        1] % 2 == 1, "filter shape must be a tuple of odd numbers!"


def _get_paddings_and_new_image(two_d_image, filter_shape, mode):
    padding_i = filter_shape[0] // 2
    padding_j = filter_shape[1] // 2
    if mode == "same":
        two_d_image = np.pad(two_d_image, [(padding_i,), (padding_j,)])
    return padding_i, padding_j, two_d_image


def _convolution(two_d_image: np.array, flt: np.array, mode: str = 'same'):
    _assertion_checks(two_d_image=two_d_image, flt=flt, mode=mode)
    padding_i, padding_j, two_d_image = _get_paddings_and_new_image(two_d_image=two_d_image,
                                                                    filter_shape=flt.shape,
                                                                    mode=mode)
    shape = tuple([two_d_image.shape[0] - 2 * padding_i, two_d_image.shape[1] - 2 * padding_j] +
                  list(flt.shape))
    multi_dim_image = np.empty(shape=shape, dtype=two_d_image.dtype)
    for i in range(padding_i, two_d_image.shape[0] - padding_i):
        for j in range(padding_j, two_d_image.shape[1] - padding_j):
            multi_dim_image[i - padding_i, j - padding_j] = two_d_image[
                                                            i - padding_i:i + padding_i + 1,
                                                            j - padding_j:j + padding_j + 1]

    expanded_filter = np.expand_dims(np.expand_dims(flt, axis=0), axis=0)

    final_image = np.sum((multi_dim_image * expanded_filter), axis=(2, 3))
    return final_image


def get_convolved_image(image, flt):
    result_image = np.empty(shape=image.shape, dtype=int)
    if len(image.shape) == 3:
        for ch in range(3):
            result_image[:, :, ch] = _convolution(two_d_image=image[:, :, ch],
                                                  flt=flt)
    elif len(image.shape) == 2:
        result_image = _convolution(two_d_image=image, flt=flt, mode='same')
    else:
        raise Exception
    return result_image


def gaussian_filter(shape=(5, 5), cutoff_frequency=1.):
    assert len(shape) == 2, "filter must be 2d!"
    assert shape[0] % 2 == 1 and shape[1] % 2 == 1
    ax1 = np.linspace(-(shape[0] - 1) / 2., (shape[0] - 1) / 2., shape[0])
    ax2 = np.linspace(-(shape[1] - 1) / 2., (shape[1] - 1) / 2., shape[1])
    gauss1 = np.exp(-0.5 * np.square(ax1) / np.square(cutoff_frequency))
    gauss2 = np.exp(-0.5 * np.square(ax2) / np.square(cutoff_frequency))
    filter = np.outer(gauss1, gauss2)
    return filter / np.sum(filter)


def box_filter(shape=(3, 3)):
    filter = np.ones(shape)
    return filter / np.sum(filter)
