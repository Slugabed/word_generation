from albumentations.augmentations.transforms import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from albumentations import Compose


def value(x_0, y_0, x, y, sigma):
    r2 = (x - x_0) ** 2 + (y - y_0) ** 2
    return np.exp(-r2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def generate_gaussian(w, h, center, sigma):
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return value(*center, x, y, sigma)


def add_gaussian(image, radius, center, coef=35):
    img = np.asarray(image)
    w, h = img.shape[0], img.shape[1]
    gaus_ = generate_gaussian(w, h, center, radius / 2)
    gaus_ = coef * radius * gaus_
    result = np.clip(img + gaus_[:, :, np.newaxis], 0, 255).astype(np.int32)
    return result


def add_random_gauss(img, count, radius, coef):
    img_ = np.asarray(img)
    radius_ = np.random.randint(*radius, size=count)
    coef_ = np.random.randint(*coef, size=count)
    x_ = np.random.rand(count)
    y_ = np.random.rand(count)
    for x, y, r, c in zip(x_, y_, radius_, coef_):
        random_center = (int(x * img_.shape[0]), int(y * img_.shape[1]))
        img_ = add_gaussian(img_, r, random_center, c)
    return img_


img = Image.open('test/transformed.jpg')
result = add_random_gauss(img, 2, radius=(100, 120), coef=(30, 50))
result = add_random_gauss(result, 2, radius=(100, 120), coef=(-50, -30))
# plt.imshow(result)
plt.imsave('test/gaus_added.jpg', result.astype(np.uint8))
# plt.imshow(result)  # , vmin=0, vmax=255
