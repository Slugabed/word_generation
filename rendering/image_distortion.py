from albumentations.augmentations.transforms import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from albumentations import Compose


def impose(background, overlay):
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.5)
    return new_img.convert('RGB')


def strong_aug(p=0.5):
    return Compose([
        # RandomGamma, ISONoise, RandomFog(p=p), RandomSnow(p=p), RandomRain(p=p)
        # RandomFog(p=p),
        # RandomShadow(p=p),
        RandomContrast(p=1.0),
        # RandomSnow(p=p),
        # ElasticTransform(p=p, alpha=0.5, sigma=50, alpha_affine=2),
        # ISONoise(p=p),
        # RandomSnow(p=p),
        # RandomRain(p=p, rain_type='drizzle', blur_value=1, brightness_coefficient=0.9),
        # ShiftScaleRotate(p=p, rotate_limit=15, shift_limit=0.00425),
        # GaussNoise(p=p, var_limit=100),
        # RandomScale(p=p),
        # RandomBrightnessContrast(p=p),
        # JpegCompression(p=p, quality_lower=30, quality_upper=40),
        # ElasticTransform(p=p),
        # GridDistortion(p=p, distort_limit=0.5, num_steps=15),
        # OpticalDistortion(p=p, distort_limit=0.1, shift_limit=0.5),
        # MotionBlur(p=p),
        # RandomSunFlare(p=p, src_radius=120),
    ], p=p)


def augmentation(image):
    mask = np.ones(image.shape, dtype=np.uint8)
    whatever_data = "my name"
    augmentation = strong_aug(p=1.0)
    data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
    augmented = augmentation(**data)
    image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], \
                                             augmented["additional"]
    return image


img_orig = Image.open('imposed.png')
img_orig = np.asarray(img_orig)
img = augmentation(img_orig)
plt.imsave('test/orig.jpg', img_orig)
plt.imsave('test/transformed.jpg', img)
