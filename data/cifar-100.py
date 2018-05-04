# Pickling original CIFAR-100

import pickle
import os
import numpy as np

PATH = '/home/hongsikkim/HDD/data/CIFAR-100/cifar-100-python'


def reshape_flat_img(img_flat):
    """

    Args:
        img_flat: Numpy array size of [3072], flattened image

    Returns: Numpy array size of [32, 32, 3], RGB image

    """
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))

    img = np.dstack((img_R, img_G, img_B))

    return img


def cifar_train():
    """CIFAR-100

    Returns: list of dict A,
        A['filename']: String, name of an image file
        A['image']: Numpy array size of [32, 32, 3], RGB image file
        A['fine_labels']: Integer 0 ~ 99, original fine labels
        A['coarse_labels']: Integer 0 ~ 19, original coarse labels
    """
    res = []
    train_filename = os.path.join(PATH, 'train')
    with open(train_filename, 'rb') as f:
        _dict = pickle.load(f, encoding='bytes')

    rng = len(_dict[b'filenames'])
    for i in range(rng):
        _res = dict()
        _res['filename'] = _dict[b'filenames'][i].decode('utf-8')
        _res['fine_labels'] = _dict[b'fine_labels'][i]
        _res['coarse_labels'] = _dict[b'coarse_labels'][i]
        _res['image'] = reshape_flat_img(_dict[b'data'][i])

        res.append(_res)
    # Pickling for the future use.
    pickle_filename = os.path.join(PATH, 'cifar-100_train.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    return res


def cifar_test():
    """CIFAR-100

    Returns: list of dict A,
        A['filename']: String, name of an image file
        A['image']: Numpy array size of [32, 32, 3], RGB image file
        A['fine_labels']: Integer 0 ~ 99, original fine labels
        A['coarse_labels']: Integer 0 ~ 19, original coarse labels
    """
    res = []
    test_filename = os.path.join(PATH, 'test')
    with open(test_filename, 'rb') as f:
        _dict = pickle.load(f, encoding='bytes')

    rng = len(_dict[b'filenames'])
    for i in range(rng):
        _res = dict()
        _res['filename'] = _dict[b'filenames'][i].decode('utf-8')
        _res['fine_labels'] = _dict[b'fine_labels'][i]
        _res['coarse_labels'] = _dict[b'coarse_labels'][i]
        _res['image'] = reshape_flat_img(_dict[b'data'][i])

        res.append(_res)
    # Pickling for the future use.
    pickle_filename = os.path.join(PATH, 'cifar-100_test.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    return res

if __name__ == '__main__':
    _ = cifar_test()
    _ = cifar_train()
