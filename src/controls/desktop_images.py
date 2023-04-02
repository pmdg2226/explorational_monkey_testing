

import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageGrab

from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


def take_screenshot(bbox):
    im = ImageGrab.grab(bbox=bbox)
    return im


def take_greyscale_screenshot(bbox):
    im = take_screenshot(bbox=bbox)
    im = np.array(im)
    im = rgb2gray(im)
    return im


def convert_to_greyscale(im):
    im = rgb2gray(im)
    return im


def compare_two_greyscale_images_mse(image1, image2):
    err = mean_squared_error(image1, image2)
    return err


def compare_two_greyscale_images_ssim(image1, image2):
    res = ssim(image1, image2)
    return res


def show_grey_image(image):
    plt.figure()
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.show()


def convert_rbga_to_grey_scale(image_as_numpy):
    # img_gray = 255 - image[:, :, 0]
    img_gray = image_as_numpy[:, :, 0]
    return img_gray


def show_rbga_grey_image(grey_image, non_blocking=True):
    plt.figure()
    plt.imshow(grey_image, cmap='gray', vmin=0, vmax=255)
    if not non_blocking:
        plt.show()


def show():
    plt.show()

