from functools import lru_cache

import torch

from py_real_esrgan.model import RealESRGAN


SUPER_RESOLUTION_SCALE = 4

@lru_cache(maxsize=None)
def get_realesrgan_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealESRGAN(device, scale=SUPER_RESOLUTION_SCALE)
    model.load_weights()
    return model


def upscale(image):
    return get_realesrgan_model().predict(image)
