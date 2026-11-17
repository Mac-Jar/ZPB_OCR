import os

os.environ["FLAGS_use_mkldnn"] = "0"

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import pytesseract
from pytesseract import Output

import sys
import atexit
import traceback
import subprocess

from multiprocessing.connection import Listener, Client

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
    return easyocr.Reader(
        [lang],
        gpu=True
    )


@lru_cache(maxsize=None)
@lru_cache(maxsize=None)
def get_paddleocr_reader(lang: str):
    import sys
    import types

    modelscope_stub = types.ModuleType("modelscope")
    modelscope_stub.__path__ = []

    def snapshot_download(*args, **kwargs):
        raise RuntimeError("ModelScope downloads are disabled in this process")

    modelscope_stub.snapshot_download = snapshot_download
    sys.modules["modelscope"] = modelscope_stub

    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
    )

def _run_ocr_local(img, method, lang='en') -> list[OcrResult]:
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

# Procesy OCR
_WORKERS = {}


def _get_worker(method):
    """Uruchamia proces OCR, jeśli jeszcze nie działa."""

    if method in _WORKERS:
        process, connection = _WORKERS[method]

        if process.poll() is None:
            return connection

        connection.close()
        del _WORKERS[method]

    authkey = os.urandom(32)

    listener = Listener(
        ("127.0.0.1", 0),
        authkey=authkey
    )

    port = listener.address[1]

    process = subprocess.Popen([
        sys.executable,
        os.path.abspath(__file__),
        "--ocr-worker",
        method,
        str(port),
        authkey.hex()
    ])

    try:
        connection = listener.accept()
    finally:
        listener.close()

    _WORKERS[method] = (process, connection)

    return connection


def _run_in_worker(img, method, lang):
    """Przekazuje obraz do odpowiedniego procesu OCR."""

    connection = _get_worker(method)

    connection.send((to_bgr_array(img), lang))

    success, result = connection.recv()

    if not success:
        raise RuntimeError(
            f"Błąd w procesie {method}:\n{result}"
        )

    return [
        OcrResult(
            text=item["text"],
            confidence=item["confidence"],
            box=item["box"]
        )
        for item in result
    ]


def run_ocr(img, method, lang='en'):
    """Uruchamia OCR w odpowiednim procesie."""

    method = method.lower()

    if method in ("easyocr", "paddleocr"):
        return _run_in_worker(img, method, lang)

    return _run_ocr_local(img, method, lang)


def _worker_main(method, port, authkey):
    """Kod wykonywany w procesie potomnym."""

    connection = Client(
        ("127.0.0.1", int(port)),
        authkey=bytes.fromhex(authkey)
    )

    while True:
        request = connection.recv()

        if request is None:
            break

        img, lang = request

        try:
            results = _run_ocr_local(
                img,
                method,
                lang
            )

            response = [
                {
                    "text": result.text,
                    "confidence": result.confidence,
                    "box": result.box
                }
                for result in results
            ]

            connection.send((True, response))

        except Exception:
            connection.send((False, traceback.format_exc()))

    connection.close()


def _stop_workers():
    """Zamyka procesy potomne przy zakończeniu programu."""

    for process, connection in _WORKERS.values():
        try:
            connection.send(None)
            connection.close()
        except (OSError, EOFError):
            pass

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()

    _WORKERS.clear()


atexit.register(_stop_workers)


if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == "--ocr-worker":
        _worker_main(
            method=sys.argv[2],
            port=sys.argv[3],
            authkey=sys.argv[4]
        )