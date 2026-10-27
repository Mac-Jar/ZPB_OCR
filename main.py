import PIL
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

TESTING_DATASET_IMAGE_PATH = "./dataset/testing_data/images/"
TESTING_DATASET_ANNOTATION_PATH = "./dataset/testing_data/annotations/"
TRAINING_DATASET_IMAGE_PATH = "./dataset/training_data/images/"
TRAINING_DATASET_ANNOTATION_PATH = "./dataset/training_data/annotations/"

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
    tot_img_filepaths = [f for f in Path(TRAINING_DATASET_IMAGE_PATH).glob("*.png")]
    print(f"Łączna liczba obrazów w zbiorze: {len(tot_img_filepaths)}")
    tot_label_filepaths = [f for f in Path(TRAINING_DATASET_ANNOTATION_PATH).glob("*.json")]


    img_filepaths = tot_img_filepaths[:limit]
    label_filepaths = tot_label_filepaths[:limit]
    # print(img_filepaths)
    # print(label_filepaths)
    # exit()
    # print(f"Przetwarzanie {len(img_filepaths)} obrazów")

    jpg_quals = [2, 10, 20]
    size_ratios = [0.1, 0.3, 0.5, 1.5]
    noise_stds = [50, 100, 200]
    print(f"Całkowita liczba zaszumionych obrazów : {len(tot_img_filepaths)  * len(noise_stds)}")
    print(f"Parametry szumu  std {noise_stds}, srednia 0")
    print(f"Całkowita liczba przeskalowanych obrazów : {len(tot_img_filepaths) * len(size_ratios)}")
    print(f"Parametry skalowania {size_ratios}")
    print(f"Całkowita liczba skompresowanych obrazów : {len(tot_img_filepaths) * len(jpg_quals)}")
    print(f"Parametry jakosci jpg {jpg_quals}")

    import json

    suma = 0
    for f in tot_label_filepaths:
    # Wczytanie pliku JSON z dysku (podaj swoją nazwę pliku, np. "dane.json")
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Pobranie listy obiektów z klucza "form"
        boxes_list = data.get("form", [])

        # Obliczenie liczby boxów
        number_of_boxes = len(boxes_list)
        suma += number_of_boxes
        # print(f"Liczba boxów w pliku {f}: {number_of_boxes}")

    print(f"całkowita liczba boxów : {suma}")
    # exit()
    for f in img_filepaths:

        img = load_image(f)
        for jpg_qual in jpg_quals:
            pipeline_jpg(img, jpg_qual)
        
        for size_ratio in size_ratios:
            pipeline_resolution(img, size_ratio)
            
        for noise_std in noise_stds:
            pipeline_noise(img, 0, noise_std)


if __name__ == "__main__":
    pipeline_all()
