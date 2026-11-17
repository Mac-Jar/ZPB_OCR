# import PIL
import numpy as np
from PIL import Image
import cv2
import json
import time
# import matplotlib.pyplot as plt
from pathlib import Path
from ocr import run_ocr, run_ocr_timed, shutdown_ocr
from plots import plot_metrics_seperate, plot_metrics, plot_example
from eval import evaluate_ocr_vs_ground_truth

TESTING_DATASET_IMAGE_PATH = "./dataset/testing_data/images/"
TESTING_DATASET_ANNOTATION_PATH = "./dataset/testing_data/annotations/"
TRAINING_DATASET_IMAGE_PATH = "./dataset/training_data/images/"
TRAINING_DATASET_ANNOTATION_PATH = "./dataset/training_data/annotations/"


CHOSEN_METHODS=[ 'easyocr','tesseract','paddleocr']

def load_image(image_path):
    return cv2.imread(image_path)


def resize(img, size_ratio: float):
    return cv2.resize(
        img,
        (
            max(1, int(img.shape[1] * size_ratio)),
            max(1, int(img.shape[0] * size_ratio))
        )
    )


def convert_to_jpg(img, quality: int):
    success, img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise Exception("Failed to convert image to JPEG")
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def add_noise_to_img(img, mean=0, std=100):
    gaussian_noise = np.random.normal(mean, std, img.shape)
    noisy_img = img.astype(np.float32) + gaussian_noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def pipeline_jpg(img, jpg_quality: int, verbose: bool = False):
    img_jpg = convert_to_jpg(img, jpg_quality)
    #ocr_result = run_ocr(img_jpg, method='tesseract')
    if verbose:
        cv2.imshow(f"Compressed, quality={jpg_quality}", img_jpg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [(method, run_ocr(img_jpg, method=method)) for method in CHOSEN_METHODS]


def pipeline_resolution(img, size_ratio: float, verbose: bool = False):
    img_res = resize(img, size_ratio)
    if verbose:
        cv2.imshow(f"Resized, ratio={size_ratio}", img_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [(method, run_ocr(img_res, method=method)) for method in CHOSEN_METHODS]


def pipeline_noise(img, noise_mean: float, noise_std: float, verbose: bool = False):
    img_noise = add_noise_to_img(img, noise_mean, noise_std)
    if verbose:
        cv2.imshow(f"Noise, std={noise_std}", img_noise)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [(method, run_ocr(img_noise, method=method)) for method in CHOSEN_METHODS]

def pipeline_all(limit=10):
    jpg_quals = [100, 75, 50, 25, 5]
    size_ratios = [1, 1.5, 0.75, 0.5, 0.33]
    noise_stds = [0, 50, 100, 200]

    img_filepaths = sorted(
        Path(TRAINING_DATASET_IMAGE_PATH).glob("*.png")
    )

    # ============================================================
    # WARM-UP MODELI OCR
    # ============================================================

    if img_filepaths:
        print("\nRozgrzewanie silników OCR...")

        warmup_img = load_image(str(img_filepaths[0]))

        for method in CHOSEN_METHODS:
            _, warmup_time = run_ocr_timed(warmup_img, method)
            print(f"{method:12} | Warm-up: {warmup_time:.3f} s")

        print("Warm-up zakończony.\n")


    if limit is not None:
        img_filepaths = img_filepaths[:limit]

    results = []

    experiment_times = {
        "jpeg": 0.0,
        "resolution": 0.0,
        "noise": 0.0,
    }

    processed_images = 0

    # ============================================================
    # EKSPERYMENT 1 - JAKOŚĆ JPEG
    # ============================================================
    whole_experiment_start = time.perf_counter()
    experiment_start = time.perf_counter()

    for f in img_filepaths:
        img = load_image(str(f))

        annotation_path = (
            Path(TRAINING_DATASET_ANNOTATION_PATH) / f"{f.stem}.json"
        )

        with open(annotation_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        boxes_list = data.get("form", [])
        processed_images += 1

        for jpg_quality in jpg_quals:
            print(f"\nJPEG: {f.name}, jakość: {jpg_quality}")

            img_test = convert_to_jpg(img, jpg_quality)

            ocr_results = {}

            for method in CHOSEN_METHODS:
                out, ocr_time = run_ocr_timed(img_test, method)
                ocr_results[method] = (out, ocr_time)

            evaluations = {}

            for method, (out, ocr_time) in ocr_results.items():
                evaluation = evaluate_ocr_vs_ground_truth(
                    out, boxes_list
                )

                evaluations[method] = evaluation

                results.append({
                    "experiment": "jpeg",
                    "jpg_quality": jpg_quality,
                    "size_ratio": None,
                    "noise_std": None,
                    "image": f.name,
                    "method": method,
                    "ocr_time": ocr_time,
                    **evaluation,
                })

                print(
                    f"{method:12} | "
                    f"CER: {evaluation['average_cer']:.4f} | "
                    f"WER: {evaluation['average_wer']:.4f} | "
                    f"Czas: {ocr_time:.3f} s"
                )

            plot_example(
                img_test,
                evaluations,
                f"{f.stem}_jpg{jpg_quality}"
            )
            # if f == img_filepaths[0]:
            #     plot_example(
            #         img_test,
            #         evaluations,
            #         f"{f.stem}_jpg{jpg_quality}"
            #     )

    experiment_times["jpeg"] = (
        time.perf_counter() - experiment_start
    )

    # ============================================================
    # EKSPERYMENT 2 - ROZDZIELCZOŚĆ
    # ============================================================

    experiment_start = time.perf_counter()

    for f in img_filepaths:
        img = load_image(str(f))

        annotation_path = (
            Path(TRAINING_DATASET_ANNOTATION_PATH) / f"{f.stem}.json"
        )

        with open(annotation_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        boxes_list = data.get("form", [])

        for size_ratio in size_ratios:
            print(f"\nROZDZIELCZOŚĆ: {f.name}, skala: {size_ratio}")

            img_test = resize(img, size_ratio)

            scale_x = img_test.shape[1] / img.shape[1]
            scale_y = img_test.shape[0] / img.shape[0]

            scaled_boxes = []

            for box in boxes_list:
                scaled_box = box.copy()

                if isinstance(box["box"][0], (int, float)):
                    # Format: [x1, y1, x2, y2]
                    scaled_box["box"] = [
                        int(coord * (scale_x if i % 2 == 0 else scale_y))
                        for i, coord in enumerate(box["box"])
                    ]
                else:
                    # Format: [[x1, y1], [x2, y2], ...]
                    scaled_box["box"] = [
                        [
                            int(point[0] * scale_x),
                            int(point[1] * scale_y)
                        ]
                        for point in box["box"]
                    ]

                scaled_boxes.append(scaled_box)

            ocr_results = {}

            for method in CHOSEN_METHODS:
                out, ocr_time = run_ocr_timed(img_test, method)
                ocr_results[method] = (out, ocr_time)

            evaluations = {}

            for method, (out, ocr_time) in ocr_results.items():
                evaluation = evaluate_ocr_vs_ground_truth(
                    out, scaled_boxes
                )

                evaluations[method] = evaluation

                results.append({
                    "experiment": "resolution",
                    "jpg_quality": None,
                    "size_ratio": size_ratio,
                    "noise_std": None,
                    "image": f.name,
                    "method": method,
                    "ocr_time": ocr_time,
                    **evaluation,
                })

                print(
                    f"{method:12} | "
                    f"CER: {evaluation['average_cer']:.4f} | "
                    f"WER: {evaluation['average_wer']:.4f} | "
                    f"Czas: {ocr_time:.3f} s"
                )
            if f == img_filepaths[0]:
                plot_example(
                    img_test,
                    evaluations,
                    f"{f.stem}_resolution{size_ratio}"
                )

    experiment_times["resolution"] = (
        time.perf_counter() - experiment_start
    )

    # ============================================================
    # EKSPERYMENT 3 - SZUM
    # ============================================================

    experiment_start = time.perf_counter()

    for f in img_filepaths:
        img = load_image(str(f))

        annotation_path = (
            Path(TRAINING_DATASET_ANNOTATION_PATH) / f"{f.stem}.json"
        )

        with open(annotation_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        boxes_list = data.get("form", [])

        for noise_std in noise_stds:
            print(f"\nSZUM: {f.name}, odchylenie: {noise_std}")

            img_test = add_noise_to_img(img, 0, noise_std)

            ocr_results = {}

            for method in CHOSEN_METHODS:
                out, ocr_time = run_ocr_timed(img_test, method)
                ocr_results[method] = (out, ocr_time)

            evaluations = {}

            for method, (out, ocr_time) in ocr_results.items():
                evaluation = evaluate_ocr_vs_ground_truth(
                    out, boxes_list
                )

                evaluations[method] = evaluation

                results.append({
                    "experiment": "noise",
                    "jpg_quality": None,
                    "size_ratio": None,
                    "noise_std": noise_std,
                    "image": f.name,
                    "method": method,
                    "ocr_time": ocr_time,
                    **evaluation,
                })

                print(
                    f"{method:12} | "
                    f"CER: {evaluation['average_cer']:.4f} | "
                    f"WER: {evaluation['average_wer']:.4f} | "
                    f"Czas: {ocr_time:.3f} s"
                )

            if f == img_filepaths[0]:
                plot_example(
                    img_test,
                    evaluations,
                    f"{f.stem}_noise{noise_std}"
                )

    experiment_times["noise"] = (
        time.perf_counter() - experiment_start
    )

    # ============================================================
    # PODSUMOWANIE WYNIKÓW
    # ============================================================

    print("\n" + "=" * 90)
    print("ŚREDNIE WYNIKI EKSPERYMENTÓW")
    print("=" * 90)

    experiments = [
        ("jpeg", "JAKOŚĆ JPEG", "jpg_quality", jpg_quals),
        ("resolution", "ROZDZIELCZOŚĆ", "size_ratio", size_ratios),
        ("noise", "SZUM", "noise_std", noise_stds),
    ]

    for experiment, experiment_name, parameter, values in experiments:
        print(f"\n{'=' * 30}")
        print(f"EKSPERYMENT: {experiment_name}")
        print(f"{'=' * 30}")

        for value in values:
            print(f"\n{parameter}: {value}")

            for method in CHOSEN_METHODS:
                matching = [
                    r for r in results
                    if r["experiment"] == experiment
                    and r[parameter] == value
                    and r["method"] == method
                ]

                if not matching:
                    continue

                avg_cer = np.mean([
                    r["average_cer"] for r in matching
                ])

                avg_wer = np.mean([
                    r["average_wer"] for r in matching
                ])

                avg_time = np.mean([
                    r["ocr_time"] for r in matching
                ])

                print(
                    f"{method:12} | "
                    f"CER: {avg_cer:.4f} | "
                    f"WER: {avg_wer:.4f} | "
                    f"Średni czas OCR: {avg_time:.3f} s | "
                    f"Liczba obrazów: {len(matching)}"
                )

        print(
            f"\nŁączny czas eksperymentu: "
            f"{experiment_times[experiment]:.2f} s"
        )

        if processed_images:
            print(
                f"Średni czas eksperymentu na obraz: "
                f"{experiment_times[experiment] / processed_images:.2f} s"
            )

    whole_experiment_end = time.perf_counter()
    print("\n" + "=" * 90)
    print("PODSUMOWANIE CZASU OCR")
    print("=" * 90)

    total_ocr_time = sum(r["ocr_time"] for r in results)
    total_ocr_calls = len(results)
    total_time=sum(experiment_times[experiment] for experiment in experiment_times )

    print(f"Łączny czas samego OCR: {total_ocr_time:.2f} s")
    print(f"Średni czas pojedynczego OCR: {total_ocr_time / total_ocr_calls:.3f} s")
    print(f"Czas z preprocessingiem: {whole_experiment_end-whole_experiment_start:.2f} s")
    print(f"Czas z preprocessingiem: {total_time:.2f} s")
    print("\nŚREDNI CZAS OCR WEDŁUG METODY:")

    for method in CHOSEN_METHODS:
        method_times = [
            r["ocr_time"] for r in results
            if r["method"] == method
        ]

        if method_times:
            print(
                f"{method:<15} "
                f"Średnio: {np.mean(method_times):.3f} s | "
                f"Łącznie: {np.sum(method_times):.2f} s | "
                f"Liczba wywołań: {len(method_times)}"
            )

    plot_metrics(results)
    plot_metrics_seperate(results)


    return results

# def pipeline_all(limit: int = 10):
#     tot_img_filepaths = [f for f in Path(TRAINING_DATASET_IMAGE_PATH).glob("*.png")]
#     # print(f"Łączna liczba obrazów w zbiorze: {len(tot_img_filepaths)}")
#     tot_label_filepaths = [f for f in Path(TRAINING_DATASET_ANNOTATION_PATH).glob("*.json")]
#
#
#     img_filepaths = tot_img_filepaths[:limit]
#     label_filepaths = tot_label_filepaths[:limit]
#     # print(img_filepaths)
#     # print(label_filepaths)
#     # exit()
#     # print(f"Przetwarzanie {len(img_filepaths)} obrazów")
#
#     jpg_quals = [100,90, 10, 20]
#     size_ratios = [1,0.1, 0.3, 0.5, 1.5]
#     noise_stds = [0, 50, 100, 200]
#     # print(f"Całkowita liczba zaszumionych obrazów : {len(tot_img_filepaths)  * len(noise_stds)}")
#     # print(f"Parametry szumu  std {noise_stds}, srednia 0")
#     # print(f"Całkowita liczba przeskalowanych obrazów : {len(tot_img_filepaths) * len(size_ratios)}")
#     # print(f"Parametry skalowania {size_ratios}")
#     # print(f"Całkowita liczba skompresowanych obrazów : {len(tot_img_filepaths) * len(jpg_quals)}")
#     # print(f"Parametry jakosci jpg {jpg_quals}")
#
#     import json
#
#     # suma = 0
#     # for f in tot_label_filepaths:
#     # # Wczytanie pliku JSON z dysku (podaj swoją nazwę pliku, np. "dane.json")
#     #     with open(f, "r", encoding="utf-8") as file:
#     #         data = json.load(file)
#
#     #     # Pobranie listy obiektów z klucza "form"
#     #     boxes_list = data.get("form", [])
#
#     #     # Obliczenie liczby boxów
#     #     number_of_boxes = len(boxes_list)
#     #     suma += number_of_boxes
#     #     # print(f"Liczba boxów w pliku {f}: {number_of_boxes}")
#
#     # print(f"całkowita liczba boxów : {suma}")
#     # exit()
#     # tot_label_filepaths.sort(key=lambda x: x.name)
#     # img_filepaths.sort(key=lambda x: x.name)
#     # print(img_filepaths[:5])
#     # print(tot_label_filepaths[:5])
#     # exit()
#     for f in img_filepaths[:1]:
#         img = load_image(f)
#         # out = []
#         for jpg_qual in jpg_quals:
#             out = pipeline_jpg(img, jpg_qual)
#             # print(type(out))
#             # print(out)
#             # print(out[0])
#             # print(f.name)
#             annotation_path = (Path(TRAINING_DATASET_ANNOTATION_PATH+f.name[:-3]+"json"))
#             with open(annotation_path, "r", encoding="utf-8") as file:
#                 data = json.load(file)
#                 boxes_list = data.get("form", [])
#                 # print(boxes_list)
#                 # print(boxes_list[0])
#                 print("easyOCR:","JPG Quality:",jpg_qual,evaluate_ocr_vs_ground_truth(out[0][1], boxes_list))
#                 print("Tesseract:","JPG Quality:",jpg_qual,evaluate_ocr_vs_ground_truth(out[1][1], boxes_list))
#                 print("Paddle:","JPG Quality:",jpg_qual,evaluate_ocr_vs_ground_truth(out[2][1], boxes_list))
#
#             # exit()
#         # for size_ratio in size_ratios:
#         #     pipeline_resolution(img, size_ratio)
#
#         # for noise_std in noise_stds:
#         #     pipeline_noise(img, 0, noise_std)
#
if __name__ == "__main__":
    try:
        pipeline_all(limit=20)
    finally:
        shutdown_ocr()