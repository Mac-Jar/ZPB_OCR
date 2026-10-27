import PIL
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

TESTING_DATASET_PATH = "./dataset/testing_data/images/"
TRAINING_DATASET_PATH = "./dataset/training_data/images/"


def load_image(image_path):
    return cv2.imread(image_path)


def resize(img, size_ratio: float):
    # return img.resize((int(img.size[0]*size_ratio), int(img.size[1]*size_ratio)),Image.Resampling.LANCZOS)

    return cv2.resize(
        img, (int(img.shape[1] * size_ratio), int(img.shape[0] * size_ratio))
    )
    # img.resize((int(img.shape[0] * size_ratio), int(img.shape[1] * size_ratio)), Image.Resampling.LANCZOS)


def convert_to_jpg(img, quality: int):
    success, img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise Exception("Failed to convert image to JPEG")
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def add_noise_to_img(img, mean=0, std=100):

    gaussian_noise = np.random.normal(mean, std, img.shape).astype(np.int8)
    print(img.dtype)
    return cv2.add(img, gaussian_noise, dtype=cv2.CV_8U)


def pipeline_jpg(img, jpg_quality: int):
    img_jpg = convert_to_jpg(img, jpg_quality)
    cv2.imshow(f"Compressed, quality={jpg_quality}", img_jpg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_jpg


def pipeline_resolution(img, size_ratio: float):
    img_res = resize(img, size_ratio)
    cv2.imshow(f"Resized, ratio={size_ratio}", img_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_res


def pipeline_noise(img, noise_mean: float, noise_std: float):
    img_noise = add_noise_to_img(img, noise_mean, noise_std)
    cv2.imshow(f"Noise, std={noise_std}", img_noise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_noise


def pipeline_all(limit: int = 10):
    filepaths = [f for f in Path(TRAINING_DATASET_PATH).glob("*.png")]
    print(f"Łączna liczba obrazów w zbiorze: {len(filepaths)}")

    filepaths = filepaths[:limit]

    print(f"Przetwarzanie {len(filepaths)} obrazów")
    
    jpg_quals = [2, 10, 20]
    size_ratios = [0.1, 0.3, 0.5, 1.5]
    noise_stds = [50, 100, 200]

    print(f"Liczba obrazów w zbiorze: {len(filepaths)}")

    for f in filepaths:
        img = load_image(f)
        for jpg_qual in jpg_quals:
            pipeline_jpg(img, jpg_qual)
        
        for size_ratio in size_ratios:
            pipeline_resolution(img, size_ratio)
            
        for noise_std in noise_stds:
            pipeline_noise(img, 0, noise_std)


if __name__ == "__main__":
    pipeline_all()
