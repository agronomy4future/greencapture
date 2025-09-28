if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)

# Ensure numpy, pandas, and cv2 exist in the active interpreter
ensure_pydeps <- function(require_gui = FALSE, verbose = TRUE) {
  # Initialize Python (binds to RETICULATE_PYTHON above)
  invisible(reticulate::py_available(initialize = TRUE))

  if (verbose) {
    cfg <- tryCatch(reticulate::py_config(), error = function(e) NULL)
    if (!is.null(cfg)) message("Using Python: ", cfg$python)
  }

  # Best-effort upgrade of pip tooling
  if (verbose) message("Upgrading pip/setuptools/wheel ...")
  try({
    reticulate::py_run_string("
import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
")
  }, silent = TRUE)

  # Ensure numpy + pandas
  for (pkg in c("numpy", "pandas")) {
    if (!reticulate::py_module_available(pkg)) {
      if (verbose) message("Installing '", pkg, "' ...")
      tryCatch(
        { reticulate::py_install(pkg, pip = TRUE) },
        error = function(e) stop("Failed to install '", pkg, "': ", conditionMessage(e))
      )
    }
  }

  # Ensure OpenCV (cv2). If GUI is requested, prefer opencv-python; else headless
  target_pkg <- if (isTRUE(require_gui)) "opencv-python" else "opencv-python-headless"
  if (!reticulate::py_module_available("cv2")) {
    if (verbose) message("Installing '", target_pkg, "' ...")
    ok <- TRUE
    tryCatch({ reticulate::py_install(target_pkg, pip = TRUE) },
             error = function(e) { ok <<- FALSE; message("Install failed: ", conditionMessage(e)) })

    # Try the alternate wheel if still missing
    if (!reticulate::py_module_available("cv2")) {
      alt_pkg <- if (identical(target_pkg, "opencv-python")) "opencv-python-headless" else "opencv-python"
      if (verbose) message("Retrying with '", alt_pkg, "' ...")
      tryCatch({ reticulate::py_install(alt_pkg, pip = TRUE) },
               error = function(e) message("Install failed: ", conditionMessage(e)))
    }
  }

  # Final check
  if (!reticulate::py_module_available("cv2")) {
    stop(
      "OpenCV ('cv2') is not available in the active Python interpreter.\n",
      "- Active Python: ", tryCatch(reticulate::py_config()$python, error = function(e) "<unknown>"), "\n",
      "- If you need windowed viewing (show_windows=TRUE), install the Microsoft Visual C++ 2015–2022 Redistributable (x64), restart R, and run again.\n",
      "- Otherwise run with show_windows=FALSE so headless OpenCV is sufficient."
    )
  }

  if (verbose) {
    ver <- reticulate::py_eval("(__import__('cv2')).__version__")
    message("Dependencies ready. cv2 version: ", ver)
  }
  invisible(TRUE)
}

# Set TRUE if you plan to open windows (show_windows = TRUE), FALSE otherwise
ensure_pydeps(require_gui = FALSE, verbose = TRUE)

#' Segment and Measure Green Objects in Images (Calibrated by Real-World Size)
#'
#' `greencapture()` processes all images in a folder, segments green objects
#' (e.g., leaves) using HSV + Lab thresholds, computes area and perimeter in
#' real units using a pixel-to-centimeter calibration, saves annotated images,
#' writes a CSV summary, and (optionally) opens a zoomable viewer.
#'
#' @description
#' The function bridges R and Python via **reticulate**. It requires Python
#' packages **numpy**, **pandas**, and **OpenCV** (`opencv-python` for GUI,
#' or `opencv-python-headless` for headless use). Use \code{ensure_pydeps()}
#' to install/verify dependencies for the active Python interpreter.
#'
#' @section Calibration:
#' Provide the real-world size (in cm) of the full image field of view via
#' \code{image_real_cm = c(width_cm, height_cm)}. Pixel measurements are scaled
#' by \eqn{sx = width\_cm / image\_width\_px} and \eqn{sy = height\_cm / image\_height\_px}.
#' Areas are computed as \eqn{area\_px * (sx * sy)} (cm^2), and perimeters are
#' computed by scaling x and y distances separately (accurate for rectangular calibration).
#'
#' @param input_folder Character. Directory containing input images
#'   (supported: \code{.jpg}, \code{.jpeg}, \code{.png}; case-insensitive).
#' @param output_folder Character. Directory where outputs will be written.
#'   Created recursively if it does not exist.
#' @param image_real_cm Numeric length-2. Real-world width and height
#'   of the imaged area in centimeters; e.g., \code{c(50, 80)} for 50 cm (x) by 80 cm (y).
#'   If a single value is provided, it is recycled to both dimensions.
#' @param show_windows Logical. If \code{TRUE}, opens an OpenCV window for each
#'   annotated image \emph{after} saving results. Press \kbd{ESC} to close.
#'   Use \code{FALSE} for headless servers.
#' @param max_win_w,max_win_w Integer. Maximum viewer window size (pixels).
#' @param zoom_step,min_scale,max_scale Numeric. Viewer zoom step and bounds.
#' @param lower_hsv,upper_hsv Integer length-3. HSV lower/upper bounds used for
#'   green segmentation (OpenCV HSV space; H in [0,179]).
#' @param lab_a_max Integer. Upper cutoff for the \code{a} channel in CIELab
#'   (values above are suppressed) to reduce red-magenta interference.
#' @param min_component_area_px Integer. Minimum connected-component area
#'   (in pixels) to consider as a candidate object.
#' @param k_open,k_close Integer length-2. Kernel sizes for morphological
#'   opening and closing (\code{cv2.morphologyEx}).
#' @param object_min_area_cm2 Numeric. Absolute minimum object area (cm^2)
#'   to keep. If no object passes this threshold, a relative filter is applied
#'   via \code{rel_min_frac_of_largest}.
#' @param rel_min_frac_of_largest Numeric in (0,1]. Relative minimum area as a
#'   fraction of the largest detected candidate (used as a fallback filter).
#' @param max_keep Integer. Maximum number of largest objects to keep per image.
#'
#' @examples
#' \dontrun{
#'# Install from GitHub (only once). Comment these lines out after install.
#'if(!require(remotes)) install.packages("remotes")
#'if (!requireNamespace("greencapture", quietly = TRUE)) {
#'  remotes::install_github("agronomy4future/greencapture", force= TRUE)
#'}
#'library(remotes)
#'library(greencapture)
#'
#'# Example: Calibrate to a 50 cm (width) × 80 cm (height) field of view
#'res= greencapture(
#'   input_folder  = r"(C:/Users/agron/Desktop/Coding)",  # folder with input images
#'   output_folder = r"(C:/Users/agron/Desktop/Coding/output)", # where outputs (CSV + images) will be saved
#'   image_real_cm = c(50, 80), # real-world width (50 cm) and height (80 cm) for calibration
#'   show_windows  = FALSE (or TRUE), # set TRUE to open viewer after saving
#'   max_win_w = 1200L, # max viewer window width (px)
#'   max_win_h = 800L, # max viewer window height (px)
#'   zoom_step = 0.1, # zoom increment per key press
#'   min_scale = 0.1, # minimum zoom (10% of image size)
#'   max_scale = 1.0, # maximum zoom (100% of image size)
#'   lower_hsv = c(30L, 25L, 30L), # lower bound (H, S, V) for green segmentation
#'   upper_hsv = c(90L, 255L, 255L), # upper bound (H, S, V) for green segmentation
#'   lab_a_max = 135L, # Lab color filter to reduce reddish interference
#'   min_component_area_px = 1000L, # ignore tiny specks (min contour area in px)
#'   k_open = c(3L,3L), # kernel size for morphological opening
#'   k_close = c(7L,7L), # kernel size for morphological closing
#'   object_min_area_cm2 = 20.0, # minimum object area (cm²) to keep
#'   rel_min_frac_of_largest = 0.35, # relative cutoff (keep >=35% of largest object)
#'   max_keep = 3L # max number of objects to retain per image
#' )
#' print(res) # display resulting summary table in R
#'
#'# Simple default code for green leaf (or fruit)
#'green= greencapture(
#'   input_folder = r"(C:/Users/agron/Desktop/Coding)",
#'   output_folder= r"(C:/Users/agron/Desktop/Coding/output)",
#'   image_real_cm= c(75, 75),
#'   show_windows= FALSE
#' )
#' print(green)
#'
#'# Code advanced: Wheat grain detection (brown grain on white background)
#'grains= greencapture(
#'   input_folder = r"(C:/Users/agron/Desktop/Coding)",
#'   output_folder= r"(C:/Users/agron/Desktop/Coding/output)",
#'   image_real_cm= c(30, 30),
#'   show_windows= FALSE,
#'   lower_hsv= c(10L, 100L, 25L),
#'   upper_hsv= c(30L, 255L, 210L),
#'   lab_a_max= 255L,
#'   min_component_area_px= 120L,
#'   k_open= c(3L, 3L),
#'   k_close= c(5L, 5L),
#'   object_min_area_cm2= 0.03,
#'   rel_min_frac_of_largest= 0.15,
#'   max_keep= 10000L
#' )
#' print(grains)
#' }
#'
#'* Github: https://github.com/agronomy4future/greencapture
#'
#' @export
greencapture = function(
    input_folder,
    output_folder,
    image_real_cm = c(20.0, 20.0),  # c(width_cm, height_cm)
    show_windows = FALSE,            # viewer opens AFTER CSV is saved
    max_win_w = 1200L, max_win_h = 800L,
    zoom_step = 0.1, min_scale = 0.1, max_scale = 1.0,
    lower_hsv = c(30L, 25L, 30L),
    upper_hsv = c(90L, 255L, 255L),
    lab_a_max = 135L,
    min_component_area_px = 1000L,
    k_open = c(3L,3L),
    k_close = c(7L,7L),
    object_min_area_cm2 = 20.0,
    rel_min_frac_of_largest = 0.35,
    max_keep = 5L
) {
  # Validate inputs
  if (length(image_real_cm) == 1) image_real_cm <- rep(image_real_cm, 2)
  stopifnot(length(image_real_cm) == 2, all(is.finite(image_real_cm)), all(image_real_cm > 0))
  if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

  # Ensure deps now (respect the show_windows choice)
  ensure_pydeps(require_gui = show_windows, verbose = FALSE)

  # Python worker (CSV written before any windows)
  py_code <- "
import cv2
import numpy as np
import os, glob, pandas as pd

def _fit_scale(h, w, max_w, max_h):
    return min(max_w/float(w), max_h/float(h), 1.0)

def _resize_for_display(img, scale):
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)

def _show_result_with_zoom(image, window_name, max_win_w, max_win_h, zoom_step, min_scale, max_scale):
    h, w = image.shape[:2]
    scale = _fit_scale(h, w, max_win_w, max_win_h)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        display = _resize_for_display(image, scale)
        cv2.putText(display, 'Zoom: +/- | ESC: close', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        elif k in (43, ord('=')):  # '+' or '='
            scale = min(max_scale, round(scale+zoom_step, 2))
        elif k in (45, ord('_')):  # '-' or '_'
            scale = max(min_scale, round(scale-zoom_step, 2))
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

def _safe_imwrite(path, img):
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f'Failed to write: {path}')
    return ok

def _discover_unique_images(folder):
    patterns = ['*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG']
    all_paths = []
    for p in patterns:
        all_paths.extend(glob.glob(os.path.join(folder, p)))
    seen = {}
    for p in all_paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen[key] = p
    unique_paths = list(seen.values())
    unique_paths.sort()
    dup_count = len(all_paths) - len(unique_paths)
    print(f'Found {len(all_paths)} file(s); using {len(unique_paths)} unique file(s). Duplicates removed: {dup_count}')
    return unique_paths

def _make_object_mask(bgr, lower_hsv, upper_hsv, lab_a_max, k_open, k_close):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    a = lab[:, :, 1]
    _, a_mask = cv2.threshold(a, lab_a_max, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(hsv_mask, a_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones(k_open,  np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(k_close, np.uint8), iterations=2)
    return mask

def _perimeter_cm_from_cnt(cnt, sx, sy):
    c = cnt.reshape(-1, 2).astype(np.float64)
    c_scaled = np.empty_like(c)
    c_scaled[:,0] = c[:,0] * sx
    c_scaled[:,1] = c[:,1] * sy
    d = np.diff(np.vstack([c_scaled, c_scaled[0]]), axis=0)
    seg_lens = np.sqrt((d[:,0]**2) + (d[:,1]**2))
    return float(seg_lens.sum())

def process_images(input_folder, output_folder,
                   image_real_cm_W, image_real_cm_H,
                   max_win_w, max_win_h, zoom_step, min_scale, max_scale,
                   lower_hsv, upper_hsv, lab_a_max,
                   min_component_area_px, k_open, k_close,
                   object_min_area_cm2, rel_min_frac_of_largest, max_keep,
                   show_windows):

    if not os.path.isdir(input_folder):
        print(f'Input folder does not exist: {input_folder}')
        return pd.DataFrame()

    os.makedirs(output_folder, exist_ok=True)
    image_paths = _discover_unique_images(input_folder)
    if not image_paths:
        print('No images found. Ensure .jpg/.jpeg/.png files exist.')
        return pd.DataFrame()

    rows = []
    annotated_paths = []

    for path in image_paths:
        filename = os.path.basename(path)
        image = cv2.imread(path)
        if image is None:
            print(f'Cannot read image: {filename} — skipping')
            continue

        h, w = image.shape[:2]
        area_per_pixel_cm2 = (image_real_cm_W / w) * (image_real_cm_H / h)
        sx = image_real_cm_W / w
        sy = image_real_cm_H / h

        mask = _make_object_mask(
            image,
            np.array(lower_hsv, dtype=np.uint8),
            np.array(upper_hsv, dtype=np.uint8),
            int(lab_a_max),
            tuple(k_open), tuple(k_close)
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= float(min_component_area_px)]

        candidates = []
        for c in contours:
            apx = float(cv2.contourArea(c))
            acm2 = apx * area_per_pixel_cm2
            candidates.append((c, apx, acm2))

        kept = [t for t in candidates if t[2] >= object_min_area_cm2]
        if not kept and candidates:
            largest_cm2 = max(t[2] for t in candidates)
            kept = [t for t in candidates if t[2] >= rel_min_frac_of_largest * largest_cm2]

        kept.sort(key=lambda t: t[2], reverse=True)
        kept = kept[:int(max_keep)]
        contours = [t[0] for t in kept]

        annotated = image.copy()
        total_area_px = 0.0
        total_area_cm2 = 0.0
        total_perim_cm = 0.0

        if contours:
            for idx, cnt in enumerate(contours, start=1):
                object_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(object_mask, [cnt], -1, 255, thickness=-1)

                area_px = float(cv2.countNonZero(object_mask))
                area_cm2 = area_px * area_per_pixel_cm2
                perim_cm = _perimeter_cm_from_cnt(cnt, sx, sy)
                pct_of_img = 100.0 * area_px / (w * h)

                total_area_px += area_px
                total_area_cm2 += area_cm2
                total_perim_cm += perim_cm

                cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 3)
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (0, 200, 0), 2)
                cv2.putText(annotated, f'Object {idx}: {area_cm2:.2f} cm^2',
                            (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2, cv2.LINE_AA)

                rows.append({
                    'File Name': filename,
                    'Object ID': idx,
                    'Image Path': path,
                    'Object Area (cm²)': round(area_cm2, 2),
                    'Object Area (px)': int(round(area_px)),
                    'Object Perimeter (cm)': round(perim_cm, 2),
                    'Object % of Image': round(pct_of_img, 2),
                    'Pixel Area (cm²/px)': round(area_per_pixel_cm2, 8),
                    'Num objects in Image': len(contours)
                })
        else:
            cv2.putText(annotated, 'No objects detected', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        if contours:
            rows.append({
                'File Name': filename,
                'Object ID': 'TOTAL',
                'Image Path': path,
                'Object Area (cm²)': round(total_area_cm2, 2),
                'Object Area (px)': int(round(total_area_px)),
                'Object Perimeter (cm)': round(total_perim_cm, 2),
                'Object % of Image': round(100.0 * total_area_px / (w * h), 2),
                'Pixel Area (cm²/px)': round(area_per_pixel_cm2, 8),
                'Num objects in Image': len(contours)
            })

        out_img = os.path.join(output_folder, os.path.splitext(filename)[0] + '_processed.jpg')
        if _safe_imwrite(out_img, annotated):
            print(f'Saved image: {out_img}')
        annotated_paths.append(out_img)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_folder, 'image_processed.csv')
    if len(df) > 0:
        try:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            try:
                size = os.path.getsize(csv_path)
                print(f'CSV saved: {csv_path} ({size} bytes)')
            except Exception:
                print(f'CSV saved: {csv_path}')
        except Exception as e:
            print(f'Failed to save CSV: {e}')

    # After CSV is written, optionally show annotated images
    if len(annotated_paths) > 0 and show_windows:
        for p in annotated_paths:
            try:
                img = cv2.imread(p)
                if img is None:
                    print(f'Could not open {p} for viewing.')
                    continue
                _show_result_with_zoom(img, window_name=p,
                                       max_win_w=max_win_w, max_win_h=max_win_h,
                                       zoom_step=zoom_step, min_scale=min_scale, max_scale=max_scale)
            except Exception as e:
                print(f'Viewer disabled (likely headless): {e}')
                break

    return df
"
  # Load Python logic
  reticulate::py_run_string(py_code)

  # Normalize paths and call into Python
  nf_in  <- normalizePath(input_folder, winslash = "\\", mustWork = FALSE)
  nf_out <- normalizePath(output_folder, winslash = "\\", mustWork = FALSE)

  width_cm  <- as.numeric(image_real_cm[1])
  height_cm <- as.numeric(image_real_cm[2])

  df_py <- reticulate::py$process_images(
    input_folder        = nf_in,
    output_folder       = nf_out,
    image_real_cm_W     = width_cm,
    image_real_cm_H     = height_cm,
    max_win_w           = as.integer(max_win_w),
    max_win_h           = as.integer(max_win_h),
    zoom_step           = as.numeric(zoom_step),
    min_scale           = as.numeric(min_scale),
    max_scale           = as.numeric(max_scale),
    lower_hsv           = as.integer(lower_hsv),
    upper_hsv           = as.integer(upper_hsv),
    lab_a_max           = as.integer(lab_a_max),
    min_component_area_px = as.integer(min_component_area_px),
    k_open              = as.integer(k_open),
    k_close             = as.integer(k_close),
    object_min_area_cm2   = as.numeric(object_min_area_cm2),
    rel_min_frac_of_largest = as.numeric(rel_min_frac_of_largest),
    max_keep            = as.integer(max_keep),
    show_windows        = isTRUE(show_windows)
  )

  out <- reticulate::py_to_r(df_py)
  if (is.null(out)) out <- data.frame()
  out
}
