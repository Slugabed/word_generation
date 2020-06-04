from albumentations.augmentations.transforms import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from albumentations import Compose

def strong_aug(p=0.5):
    return Compose([
        RandomFog(p=1.0, fog_coef_lower=0.2, fog_coef_upper=0.4),
        GaussNoise(p=1.0),
        ISONoise(p=1.0)
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


img_orig = Image.open('test/gaus_added.jpg')
img_orig = np.asarray(img_orig)
img = augmentation(img_orig)
plt.imsave('test/unfocused.jpg', img)