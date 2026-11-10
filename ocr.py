import os

os.environ["FLAGS_use_mkldnn"] = "0"

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class OcrResult:
    text: str
    confidence: float
    box: list[float]


def to_bgr_array(img):
    if isinstance(img, np.ndarray):
        return img
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


@lru_cache(maxsize=None)
def get_easyocr_reader(lang: str):
    import easyocr
    return easyocr.Reader([lang])


@lru_cache(maxsize=None)
def get_paddleocr_reader(lang: str):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
    )


def run_ocr(
    img,
    method: Literal['tesseract', 'easyocr', 'paddleocr'],
    lang: str = 'en',
) -> list[OcrResult]:
    results: list[OcrResult] = []

    if method == 'tesseract':

        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            if not text or conf < 0:
                continue
            x, y = data['left'][i], data['top'][i]
            w, h = data['width'][i], data['height'][i]
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            results.append(OcrResult(text=text, confidence=conf / 100.0, box=box[0]+box[2],
))

    elif method == 'easyocr':
        reader = get_easyocr_reader(lang)
        for box, text, conf in reader.readtext(to_bgr_array(img)):
            # print(box[0]+box[2])
            results.append(
                OcrResult(
                    text=text,
                    confidence=float(conf),
                    box=box[0]+box[2],
                )
            )

    elif method == 'paddleocr':

        ocr = get_paddleocr_reader(lang)
        raw = ocr.predict(to_bgr_array(img))
        for page in raw:
            texts = page['rec_texts']
            scores = page['rec_scores']
            polys = page['rec_polys']
            for text, score, poly in zip(texts, scores, polys):
                results.append(
                    OcrResult(
                        text=text,
                        confidence=float(score),
                        box = list(np.hstack((poly[0],poly[2])))
                    )
                )

    else:
        raise ValueError(f"Unsupported OCR method: {method}")

    return results


def ocr_text(results: list[OcrResult]) -> str:
    return " ".join(r.text for r in results)


def ocr_mean_confidence(results: list[OcrResult]) -> float:
    if not results:
        return 0.0
    return sum(r.confidence for r in results) / len(results) 