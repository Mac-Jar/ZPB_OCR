import PIL
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

TESTING_DATASET_PATH = './dataset/testing_data/images/'
TRAINING_DATASET_PATH = './dataset/training_data/images/'

def load_image(image_path):
    return cv2.imread(image_path)

def resize(img, size_ratio: float):
    #return img.resize((int(img.size[0]*size_ratio), int(img.size[1]*size_ratio)),Image.Resampling.LANCZOS)

    return cv2.resize(img,(int(img.shape[1] * size_ratio), int(img.shape[0] * size_ratio)))
    #img.resize((int(img.shape[0] * size_ratio), int(img.shape[1] * size_ratio)), Image.Resampling.LANCZOS)

def convert_to_jpg(img, quality : int):
    success ,img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise Exception("Failed to convert image to JPEG")
    return cv2.imdecode(img, cv2.IMREAD_COLOR)

def add_noise_to_img(img,mean=0,std=100 ):

    gaussian_noise = np.random.normal(mean,std,img.shape).astype(np.int8)
    print(img.dtype)
    return cv2.add(img, gaussian_noise,dtype=cv2.CV_8U)


def pipeline(filename):
    img = load_image(TESTING_DATASET_PATH + filename)
    img_jpg= convert_to_jpg(img, 10)
    img_new = resize(img_jpg, 0.25)
    img_noise = add_noise_to_img(img_jpg)
    print(img_new.shape)

    cv2.imshow("original",img)
    cv2.waitKey(0)

    cv2.imshow("compressed", img_jpg)
    cv2.waitKey(0)

    cv2.imshow("resized", img_new)
    cv2.waitKey(0)

    cv2.imshow("noise", img_noise)
    cv2.waitKey(0)

if __name__ == '__main__':
    filename="82092117.png"
    pipeline(filename)
    # print(img.shape)
    # print(img_new.shape)
    #
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    #
    # cv2.imshow("test", img_new)
    # cv2.waitKey(0)
    # plt.imshow(img)
    # plt.show()
    #
    # plt.imshow(img_new)
    # plt.show()