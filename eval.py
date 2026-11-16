def levenshtein(a, b):
    """Odległość Levenshteina dla stringów lub list słów."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return levenshtein(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    ref_w, hyp_w = ref.split(), hyp.split()
    return levenshtein(ref_w, hyp_w) / max(len(ref_w), 1)


def overlap_ratio(gt_box, ocr_box):
    gx1, gy1, gx2, gy2 = map(float, gt_box)
    ox1, oy1, ox2, oy2 = map(float, ocr_box)

    x1 = max(gx1, ox1)
    y1 = max(gy1, oy1)
    x2 = min(gx2, ox2)
    y2 = min(gy2, oy2)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)

    return inter / area if area > 0 else 0.0


def evaluate_ocr_vs_ground_truth(ocr_results, ground_truth, min_overlap=0.5):
    """
    ocr_results:  lista obiektów z polami .text i .box  ([x1, y1, x2, y2])
    ground_truth: lista słowników {'text': ..., 'box': [x1, y1, x2, y2]}
    """
    # ocr_results = [item for _, item in ocr_results]
    details = []

    for gt in ground_truth:
        if not gt.get('text') or not gt.get('box'):
            continue

        # boksy OCR leżące (w większości) wewnątrz boksa GT
        matched = [o for o in ocr_results
                   if overlap_ratio(gt['box'], o.box) >= min_overlap]

        # kolejność czytania: góra -> dół, lewo -> prawo
        matched.sort(key=lambda o: (o.box[1], o.box[0]))
        ocr_text = " ".join(o.text for o in matched)

        details.append({
            'gt_text': gt['text'],
            'ocr_text': ocr_text,
            'gt_box': gt['box'],
            'ocr_boxes': [o.box for o in matched],
            'correct': gt['text'].strip().casefold() == ocr_text.strip().casefold(),
            'n_boxes': len(matched),
            'cer': cer(gt['text'], ocr_text),
            'wer': wer(gt['text'], ocr_text),
        })

    n = len(details)
    return {
        'average_cer': sum(d['cer'] for d in details) / n if n else 0.0,
        'average_wer': sum(d['wer'] for d in details) / n if n else 0.0,
        'matched_records_count': n,
        'details': details,
    }