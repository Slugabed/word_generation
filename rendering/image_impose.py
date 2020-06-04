from albumentations.augmentations.transforms import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from albumentations import Compose


def impose(background, overlay, shape):
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    background = background.resize(shape, Image.ANTIALIAS)
    overlay = overlay.resize(shape, Image.ANTIALIAS)

    # background.thumbnail((300, 300), Image.ANTIALIAS)
    # overlay.thumbnail((300, 300), Image.ANTIALIAS)

    new_img = Image.blend(background, overlay, 0.2)
    return new_img.convert('RGB')


background = Image.open('text_gen.jpg')
overlay = Image.open('hello_gem.png')
imposed = impose(background, overlay, (300, 300))
plt.imsave('imposed.png', np.asarray(imposed))
plt.imshow(imposed)