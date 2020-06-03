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


def aug(img_size, p=0.5, steps=10):
    step = 0

    gauss_state = max(0, 60 + 20 * np.random.randn())
    gauss_step = 5
    # -0.2;0.2
    br_state = np.random.randn() * 0.2
    br_step = 0.01 * np.random.randn()
    br_interval = 0.01
    # -0.2;0.2
    contr_state = np.random.randn() * 0.2
    contr_step = 0.01 * np.random.randn()
    contr_interval = br_interval

    # 30;70
    jpeg_state = max(0, 50 + 20 * np.random.randn())
    jpeg_step = 8 * np.random.randn()

    # -10;10
    rotate_state = 1.0 * np.random.randn()
    rotate_step = 1.0 * np.random.randn()

    sun_rad_state = max(0, 30 + 10 * np.random.randn())
    sun_rad_step = 5 * sun_rad_state

    start_x, start_y, end_x, end_y = np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()
    delta_x, delta_y = (end_x - start_x) / steps, (end_y - start_y) / steps
    roi_region = 0.02
    sun_roi_state = (max(0, start_x), max(0, start_y), min(1, start_x + roi_region), min(1, start_y + roi_region))

    def strong_aug(*args, **kwargs):
        # border_mode=cv2.BORDER_CONSTANT, value=255
        # random pad values
        # Crop
        nonlocal step, br_state, contr_state, jpeg_state, rotate_state, sun_rad_state, sun_roi_state, gauss_state
        step += 1
        gauss_state += max(0, gauss_step * np.random.randn())
        br_state += br_step * np.random.randn()
        brightness_int = (br_state, min(1.0, br_state + br_interval))
        contr_state += contr_step * np.random.randn()
        contrast_int = (contr_state, min(1.0, contr_state + contr_interval))
        jpeg_state += jpeg_step * np.random.randn()
        rotate_state += rotate_step * np.random.randn()
        rotate_int = (int(rotate_state), int(min(1.0, rotate_state + 0.1)))
        sun_rad_state += sun_rad_step * np.random.randn()
        sun_roi_x, sun_roi_y = max(0, sun_roi_state[0] + delta_x * (step - 1)), max(0, sun_roi_state[1] + delta_y * (
                step - 1))
        sun_roi_state = (sun_roi_x, sun_roi_y, min(1.0, sun_roi_x + roi_region), min(1.0, sun_roi_y + roi_region))

        crop_width = int(img_size[0] / max(0.4, ((0.5 + 0.15 * np.random.randn()) * img_size[2])))
        return Compose([
            GaussNoise(p=0.9, var_limit=gauss_state),
            # RandomScale(p=p),
            RandomBrightnessContrast(p=0.8, brightness_limit=brightness_int, contrast_limit=contrast_int),  # brightness_limit=0.2, contrast_limit=0.2 -> linear
            JpegCompression(p=0.4, quality_lower=30, quality_upper=40),  # linear quality
            # ElasticTransform(p=p, alpha=0.3, alpha_affine=1),
            # GridDistortion(p=p, distort_limit=0.2),
            Rotate(limit=rotate_int, p=0.9),  # limit -> -10;+10
            # OpticalDistortion(p=p, distort_limit=0.1, shift_limit=0.5),
            MotionBlur(p=0.3),
            RandomSunFlare(p=p, src_radius=int(img_size[0] / 4), flare_roi=sun_roi_state),
            Cutout(p=0.25, max_h_size=img_size[1]*2, max_w_size=crop_width, num_holes=1)
            # flare_roi, src_radius, src_color
        ])(*args, **kwargs)

    return strong_aug


def augmentation(image, aug, p=0.5, steps=10):
    mask = np.ones(image.shape, dtype=np.uint8)
    whatever_data = "my name"
    data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
    augmented = aug(**data)
    image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], \
                                             augmented["additional"]
    return image


def text_generator(text, font_size=25, font_path="resources/opensans/OpenSans-Italic.ttf", x_indent_left=0.05,
                   x_indent_right=0.05, y_indent_top=0.15, y_indent_bottom=0.15, indexnt_sigma=0.5):
    x_indent_left = (1 + 0.2 * np.random.randn() * indexnt_sigma) * x_indent_left
    x_indent_right = (1 + 0.2*np.random.randn() * indexnt_sigma) * x_indent_right
    y_indent_top = (1 + 0.2*np.random.randn() * indexnt_sigma) * y_indent_top
    y_indent_bottom = (1 + 0.2*np.random.randn() * indexnt_sigma) * y_indent_bottom

    def generate_text():
        nonlocal x_indent_left, x_indent_right, y_indent_top, y_indent_bottom
        usr_font = ImageFont.truetype(font_path, font_size)
        width, height = usr_font.getsize(text)
        x_indent_left = (1 + np.random.randn() * indexnt_sigma / 20) * x_indent_left
        y_indent_top = (1 + np.random.randn() * indexnt_sigma / 20) * y_indent_top
        x_indent_right = (1 + np.random.randn() * indexnt_sigma / 20) * x_indent_right
        y_indent_bottom = (1 + np.random.randn() * indexnt_sigma / 20) * y_indent_bottom

        width_ratio = 1 + x_indent_left + x_indent_right
        height_ratio = 1 + y_indent_top + y_indent_bottom
        image = Image.new("RGB", (int(width_ratio * width), int(height_ratio * height)), (255, 255, 255))
        d_usr = ImageDraw.Draw(image)
        d_usr.text((int(x_indent_left * width), int(y_indent_top * height)), text, (0, 0, 0), font=usr_font)
        return image, width, height

    return generate_text


def main(args):
    with open(args.text_file) as f:
        txt = f.read()
        txt = txt.translate(str.maketrans('', '', string.punctuation + r"""«»"""))
        tokens = txt.split()
        for _ in range(args.image_count):
            random_index = np.random.randint(0, len(tokens))
            word = tokens[random_index]
            steps = 15
            generate_text = text_generator(word, indexnt_sigma=1, y_indent_top=0.3, y_indent_bottom=0.3)

            for step in range(args.sample_count):
                fig_, img_width, img_height = generate_text()
                image = np.asarray(fig_)
                aug_ = aug((img_width, img_height, len(word)), p=1.0, steps=steps)
                image = augmentation(image, aug_)
                plt.imsave(os.path.join(args.output_dir, f'{word}_{step}.png'), image)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--text_file", required=True,
                           help="Path to the text file",
                           type=str)
    argparser.add_argument("--output_dir", required=True,
                           help="Path where generated images will be saved",
                           type=str)
    argparser.add_argument("--image_count",
                           help="How much words to generate", type=int, nargs='?', const=15,
                           default=15)
    argparser.add_argument("--sample_count",
                           help="Amount of samples of one word.",
                           type=int, nargs='?', const=15, default=15)

    try:
        args = argparser.parse_args()
    except:
        argparser.error("Invalid arguments")
        sys.exit(0)
    main(args)
