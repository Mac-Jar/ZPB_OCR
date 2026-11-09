import numpy as np

def calculate_levenshtein(s1, s2):
    """Oblicza odległość Levenshteina dla ciągów znaków lub list słów."""
    if len(s1) < len(s2):
        return calculate_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(ref, hyp):
    """Oblicza Character Error Rate (CER)."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return calculate_levenshtein(ref, hyp) / len(ref)

def calculate_wer(ref, hyp):
    """Oblicza Word Error Rate (WER)."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return calculate_levenshtein(ref_words, hyp_words) / len(ref_words)

def normalize_box(box, is_ocr=True):
    """
    Normalizuje współrzędne boksów do formatu [ymin, xmin, ymax, xmax].
    - OCR (Lista 1): [[y1, x1], [y2, x1], [y2, x2], [y1, x2]]
    - Wzorzec (Lista 2): [ymin, xmin, ymax, xmax] lub podobne
    """
    return box
    print(box, is_ocr)
    if is_ocr:
        # box ma postać 4 punktów [ [y, x], [y, x], ... ]
        ymin = min(p[0] for p in box)
        xmin = min(p[1] for p in box)
        ymax = max(p[0] for p in box)
        xmax = max(p[1] for p in box)
        return [ymin, xmin, ymax, xmax]
    else:
        # wzorzec ma zazwyczaj [ymin, xmin, ymax, xmax] lub [xmin, ymin, xmax, ymax]
        # w formacie FUNSD jest to [ymin, xmin, ymax, xmax]
        return list(box)

def compute_iou(box1, box2):
    """Oblicza wskaźnik Intersection over Union (IoU) dla dwóch boksów."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2
    
    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)
    
    if inter_ymax < inter_ymin or inter_xmax < inter_xmin:
        return 0.0
    
    inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
    area1 = (y1_max - y1_min) * (x1_max - x1_min)
    area2 = (y2_max - y2_min) * (x2_max - x2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def evaluate_ocr_vs_ground_truth(ocr_results, ground_truth):
    """
    Porównuje wyniki OCR z wzorcem (Ground Truth) i wylicza średni CER oraz WER.
    """
    # 1. Standaryzacja boksów i tekstów OCR
    ocr_items = []
    # engine_name, ocr_items = ocr_results[0]

    for m, item in ocr_results:
        # text = item.text
        # box = item.box
        # Obsługa zarówno obiektów (np. OcrResult), jak i słowników
        if hasattr(item, 'text') and hasattr(item, 'box'):
            text = item.text
            box = item.box
        elif isinstance(item, dict):
            text = item.get('text', '')
            box = item.get('box', [])
        else:
            continue
            
        norm_box = normalize_box(box, is_ocr=True)
        ocr_items.append({'text': text, 'box': norm_box})
        
    total_cer = 0.0
    total_wer = 0.0
    matched_count = 0
    evaluation_results = []
    
    # 2. Dopasowanie rekordów z wzorca (Ground Truth) do wyników OCR
    for gt in ground_truth:
        gt_text = gt.get('text', '')
        gt_box = gt.get('box', [])
        
        # Pomijamy elementy bez tekstu (np. puste kontenery typu 'other' bez treści)
        if not gt_text or not gt_box:
            continue
        # print("-------------------------")
        norm_gt_box = normalize_box(gt_box, is_ocr=False)
        
        best_match_text = ""
        best_iou = 0.0
        
        # Szukanie najlepiej pokrywającego się elementu OCR
        for ocr in ocr_items:
            iou = compute_iou(norm_gt_box, ocr['box'])
            if iou > best_iou:
                best_iou = iou
                best_match_text = ocr['text']
                
        # 3. Wyliczenie błędów CER i WER dla sparowanej pary
        cer = calculate_cer(gt_text, best_match_text)
        wer = calculate_wer(gt_text, best_match_text)
        
        total_cer += cer
        total_wer += wer
        matched_count += 1
        
        evaluation_results.append({
            'gt_text': gt_text,
            'ocr_text': best_match_text,
            'iou': best_iou,
            'cer': cer,
            'wer': wer
        })
        
    # 4. Agregacja wyników końcowych (średnie wartości)
    avg_cer = total_cer / matched_count if matched_count > 0 else 0.0
    avg_wer = total_wer / matched_count if matched_count > 0 else 0.0
    
    return {
        'average_cer': avg_cer,
        'average_wer': avg_wer,
        'matched_records_count': matched_count,
        'details': evaluation_results
    }