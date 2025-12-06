import os
from pathlib import Path
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path, output_path=None, verbose=True):
    """
    Process `image_path` and save/display final image with tumor contour drawn.
    Returns dict with results: {'output_path':..., 'area': area or None, 'perimeter': perim or None, 'detected': bool}
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    # Read two copies:
    # - gray for processing
    # - color for final overlay (so we can draw colored contours safely)
    gray = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
    color = cv.imread(str(p), cv.IMREAD_COLOR)

    if gray is None or color is None:
        raise IOError(f"cv.imread failed to read the image. Check file integrity: {image_path}")

    # Keep shapes
    X = gray.shape[0]
    copy = np.copy(gray)

    # -------- First enhancement ----------
    blur = cv.GaussianBlur(copy, (5,5), 2)
    enh = cv.add(copy, (cv.add(blur, -100)))

    # -------- Denoising ----------
    median = cv.medianBlur(enh, 5)

    # -------- Morphological Gradient ----------
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    gradient = cv.morphologyEx(median, cv.MORPH_GRADIENT, kernel)

    # -------- Second enhancement ----------
    enh2 = cv.add(median, gradient)

    # -------- First thresholding ----------
    t = np.percentile(enh2, 85)
    _, th = cv.threshold(enh2, t, 255, cv.THRESH_BINARY)

    # -------- Morphology operations ----------
    kernel_c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((5*X)/100), int((5*X)/100)))
    kernel_e = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((3*X)/100), int((3*X)/100)))
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((7*X)/100), int((7*X)/100)))

    opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel_e)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_c)
    erosion = cv.erode(closing, kernel_e, iterations=1)
    dilation = cv.dilate(erosion, kernel_e, iterations=1)

    # -------- Masking ----------
    masked = cv.bitwise_and(copy, copy, mask=dilation)

    # -------- Second round of morphology operations ----------
    s_erosion = cv.erode(masked, kernel, iterations=1)
    final = cv.morphologyEx(s_erosion, cv.MORPH_OPEN, ker)

    # -------- Third enhancement ----------
    blur3 = cv.GaussianBlur(final, (3,3), 0)
    enh3 = cv.add(final, (cv.add(blur3, -100)))

    # -------- Second thresholding ----------
    upper = np.percentile(enh3, 92)
    res = cv.inRange(enh3, 0, upper)

    # -------- Final morphology step ----------
    fin = cv.morphologyEx(res, cv.MORPH_CLOSE,
                          cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((7*X)/100), int((7*X)/100))))

    # -------- Contours (improved selection) ----------
    # findContours modifies input, so pass a copy
    contours, hierarchy = cv.findContours(fin.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Prepare color image for drawing: convert BGR to RGB for matplotlib display
    final_color = cv.cvtColor(color, cv.COLOR_BGR2RGB)

    area = None
    perimeter = None
    detected = False

    # Precompute some sizes
    rows, cols = fin.shape
    image_area = rows * cols

    # Heuristics thresholds (tunable)
    min_area_ratio = 0.0005   # ignore contours smaller than this fraction of image area (0.05%)
    max_area_ratio = 0.85     # ignore contours bigger than this fraction (likely border) (85%)
    min_area_px = 50          # absolute minimum area in pixels
    min_solidity = 0.4        # contour area / convex hull area
    intensity_percentile_threshold = np.percentile(enh3, 70)  # threshold for mean intensity inside contour

    # Collect candidate contours
    candidates = []
    for cnt in contours:
        a = cv.contourArea(cnt)
        if a < min_area_px:
            continue
        if a < min_area_ratio * image_area:
            continue
        if a > max_area_ratio * image_area:
            continue

        # bounding rect - ignore contours touching the image border
        x, y, w, h = cv.boundingRect(cnt)
        if x <= 1 or y <= 1 or (x + w) >= (cols - 1) or (y + h) >= (rows - 1):
            continue

        # solidity check
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = a / hull_area
        if solidity < min_solidity:
            continue

        # intensity check: mean intensity of enh3 inside contour mask should be relatively high
        mask = np.zeros_like(enh3, dtype=np.uint8)
        cv.drawContours(mask, [cnt], -1, 255, -1)  # filled
        mean_intensity = cv.mean(enh3, mask=mask)[0]
        if mean_intensity < intensity_percentile_threshold:
            # skip if mean intensity is lower than expected (likely background)
            continue

        # passed all checks -> candidate
        candidates.append((cnt, a, solidity, mean_intensity))

    # If we have candidates, choose the one with largest area (or you could pick by mean_intensity)
    if candidates:
        # sort by area (largest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen_cnt, area_val, solidity, mean_intensity = candidates[0]
        area = int(area_val)
        perimeter = int(cv.arcLength(chosen_cnt, True))
        cv.drawContours(final_color, [chosen_cnt], -1, (0, 255, 0), 3)  # color is in RGB now
        detected = True
        if verbose:
            print(f"Selected contour: area={area} px, perimeter={perimeter} px, solidity={solidity:.2f}, mean_intensity={mean_intensity:.1f}")
    else:
        # Fallback: choose largest contour that is reasonably big (closest to previous logic)
        filtered = [c for c in contours if cv.contourArea(c) > max(min_area_px, min_area_ratio * image_area)]
        if filtered:
            cnt = max(filtered, key=cv.contourArea)
            area = int(cv.contourArea(cnt))
            perimeter = int(cv.arcLength(cnt, True))
            # avoid drawing if this contour touches borders heavily
            x, y, w, h = cv.boundingRect(cnt)
            if not (x <= 1 or y <= 1 or (x + w) >= (cols - 1) or (y + h) >= (rows - 1)):
                cv.drawContours(final_color, [cnt], -1, (0, 255, 0), 3)
                detected = True
                if verbose:
                    print(f"Fallback contour selected: area={area} px, perimeter={perimeter} px")
            else:
                # final fallback: don't draw contour (likely border)
                if verbose:
                    print("No suitable internal contour found; detected contours likely touch borders. No tumor marked.")
        else:
            if verbose:
                print("No Tumor Detected (no contour large enough)")

    # -------- Save and show the final output ----------
    if output_path is None:
        # create an output path next to the input image
        out_dir = p.parent
        out_name = p.stem + "_detected" + p.suffix
        output_path = str(out_dir / out_name)
    else:
        output_path = str(Path(output_path))

    # Convert back to BGR for cv.imwrite (since matplotlib uses RGB)
    save_img = cv.cvtColor(final_color, cv.COLOR_RGB2BGR)
    cv.imwrite(output_path, save_img)

    # Display once (matplotlib expects RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(final_color)
    plt.axis("off")
    title = "DETECTED TUMOR" if detected else "NO TUMOR DETECTED"
    plt.title(title)
    plt.show()

    return {"output_path": output_path, "area": area, "perimeter": perimeter, "detected": detected}

if __name__ == "__main__":
    #change the path below
    image_path = r"E:\Collage\CNN Brain Tumor\dataset\test\test_20.jpg"
    
    output_path = None
    results = process_image(image_path=image_path, output_path=output_path, verbose=True)

    print("Final saved to:", results["output_path"])
    if results["detected"]:
        print("Area (px):", results["area"])
        print("Perimeter (px):", results["perimeter"])