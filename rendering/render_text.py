from PIL import Image, ImageFont, ImageDraw
import numpy as np
import string
from matplotlib import pyplot as plt
from PIL import Image
from albumentations import Compose
from albumentations.augmentations.transforms import *
import argparse
import sys
import os


def text_generator(text, font_size=25, font_path="resources/opensans/OpenSans-Italic.ttf", x_indent_left=0.05,
                   x_indent_right=0.05, y_indent_top=0.15, y_indent_bottom=0.15, indexnt_sigma=0.5):
    x_indent_left = (1 + 0.2 * np.random.randn() * indexnt_sigma) * x_indent_left
    x_indent_right = (1 + 0.2 * np.random.randn() * indexnt_sigma) * x_indent_right
    y_indent_top = (1 + 0.2 * np.random.randn() * indexnt_sigma) * y_indent_top
    y_indent_bottom = (1 + 0.2 * np.random.randn() * indexnt_sigma) * y_indent_bottom

    def generate_text():
        nonlocal x_indent_left, x_indent_right, y_indent_top, y_indent_bottom
        usr_font = ImageFont.truetype(font_path, font_size)
        x_indent_left = (1 + np.random.randn() * indexnt_sigma / 20) * x_indent_left
        y_indent_top = (1 + np.random.randn() * indexnt_sigma / 20) * y_indent_top
        x_indent_right = (1 + np.random.randn() * indexnt_sigma / 20) * x_indent_right
        y_indent_bottom = (1 + np.random.randn() * indexnt_sigma / 20) * y_indent_bottom

        width_ratio = 1 + x_indent_left + x_indent_right
        height_ratio = 1 + y_indent_top + y_indent_bottom
        # (int(width_ratio * width), int(height_ratio * height))
        image = Image.new("RGB", (600, 400), (255, 255, 255))
        d_usr = ImageDraw.Draw(image)
        y_current = 0
        for line in text.split('\n'):
            width, height = usr_font.getsize(line)
            y_current += height
            d_usr.text((25, y_current), line, (0, 0, 0), font=usr_font)
        return image

    return generate_text


gen_ = text_generator(
    'Hello world! I am glad to see you!\nHow are you? Will you have a nice day, dude?\nPlease, I ask you '
    'only one thing:\nWhatever is going on\nJust be happy!')
image = gen_()

# print(type(image))
plt.imsave('/home/master/PycharmProjects/tomophantom/diploma/rendering/hello_gem.png', np.asarray(image))
# plt.imshow(image)
