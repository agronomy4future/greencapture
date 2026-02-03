if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)

ensure_pydeps= function(require_gui = FALSE, verbose = TRUE) {
  invisible(reticulate::py_available(initialize = TRUE))
  if (verbose) {
    cfg= tryCatch(reticulate::py_config(), error = function(e) NULL)
    if (!is.null(cfg)) message("Using Python: ", cfg$python)
  }

  for (pkg in c("numpy", "pandas")) {
    if (!reticulate::py_module_available(pkg)) {
      reticulate::py_install(pkg, pip = TRUE)
    }
  }

  target_pkg= if (isTRUE(require_gui)) "opencv-python" else "opencv-python-headless"
  if (!reticulate::py_module_available("cv2")) {
    reticulate::py_install(target_pkg, pip = TRUE)
  }
}

#' Segment and Measure Green Objects in Images
#'
#' @description
#' \code{greencapture()} processes all images in a folder and segments green
#' objects (such as leaves, fruits, or grains) using HSV color thresholds.
#' It computes real-world area and perimeter measurements via pixel-to-centimeter
#' calibration and exports annotated images and a CSV summary table.
#'
#' @param input_folder Character. Directory containing input images.
#' @param output_folder Character. Directory where outputs will be saved.
#' @param image_real_cm Numeric length 2. Real-world width and height of the image
#'   field of view in centimeters.
#' @param show_windows Logical. If TRUE, shows interactive OpenCV viewer.
#' @param lower_hsv Integer length 3. Base HSV lower bound.
#' @param upper_hsv Integer length 3. Base HSV upper bound.
#' @param extra_hsv List or NULL. Optional additional HSV ranges.
#'   Each element must be a list with \code{lower} and \code{upper}.
#' @param lab_a_max Integer. Upper cutoff for Lab a-channel.
#' @param min_component_area_px Integer. Minimum connected component area (pixels).
#' @param k_open,k_close Integer length 2. Kernel sizes for morphology.
#' @param object_min_area_cm2 Numeric. Minimum object area (cm^2).
#' @param rel_min_frac_of_largest Numeric. Relative fallback threshold.
#' @param max_keep Integer. Maximum number of objects per image.
#'
#' @return
#' A data frame with one row per detected object containing:
#' \itemize{
#'   \item File Name
#'   \item Object ID
#'   \item Object Area (cm²)
#'   \item Object Perimeter (cm)
#' }
#'
#' @examples
#' \dontrun{
#'
#' # Without designating HSV, the default is green color
#' greencapture(
#'   input_folder = "C:/Users/agron",
#'   output_folder = "C:/Users/agron/output",
#'   image_real_cm = c(50, 50) # image frame size (for calibration)
#' )
#'
#' # Designating HSV, different colors can be captured
#' greencapture(
#'   input_folder = "C:/Users/agron",
#'   output_folder = "C:/Users/agron/output",
#'   image_real_cm = c(30, 30),
#'   extra_hsv = list(
#'     list(lower = c(35,40,40), upper = c(85,255,255)),
#'     list(lower = c(15,60,60), upper = c(30,255,255))
#'   )
#' )
#' □ Website: https://agronomy4future.com/archives/24628
#' □ Github: https://github.com/agronomy4future/greencapture
#' - All Rights Reserved © J.K Kim (kimjk@agronomy4future.com)
#' }
#' @export
greencapture= function(
    input_folder,
    output_folder,
    image_real_cm = c(20,20),
    show_windows = FALSE,
    lower_hsv = c(0L,0L,0L),
    upper_hsv = c(180L,255L,50L),
    extra_hsv = NULL,  # NEW
    lab_a_max = 135L,
    min_component_area_px = 1000L,
    k_open = c(3L,3L),
    k_close = c(7L,7L),
    object_min_area_cm2 = 20,
    rel_min_frac_of_largest = 0.35,
    max_keep = 5L
) {

  if (length(image_real_cm)==1) image_real_cm <- rep(image_real_cm,2)
  if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)

  ensure_pydeps(show_windows, FALSE)

  py_code= "
import cv2, numpy as np, os, glob, pandas as pd

def _perimeter_cm_from_cnt(cnt, sx, sy):
    c = cnt.reshape(-1,2).astype(float)
    c[:,0] *= sx
    c[:,1] *= sy
    d = np.diff(np.vstack([c,c[0]]), axis=0)
    return float(np.sqrt((d[:,0]**2)+(d[:,1]**2)).sum())

def process_images(input_folder, output_folder,
                   image_real_cm_W, image_real_cm_H,
                   lower_hsv, upper_hsv, extra_hsv,
                   lab_a_max,
                   min_component_area_px, k_open, k_close,
                   object_min_area_cm2, rel_min_frac_of_largest, max_keep,
                   show_windows):

    rows=[]
    for path in glob.glob(os.path.join(input_folder,'*.jpg')):
        img=cv2.imread(path)
        if img is None: continue
        h,w=img.shape[:2]
        sx=image_real_cm_W/w
        sy=image_real_cm_H/h
        area_per_pixel=(sx*sy)

        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)

        mask=cv2.inRange(hsv,np.array(lower_hsv),np.array(upper_hsv))

        if extra_hsv is not None:
            for rng in extra_hsv:
                lo=np.array(rng['lower'],dtype=np.uint8)
                hi=np.array(rng['upper'],dtype=np.uint8)
                mask2=cv2.inRange(hsv,lo,hi)
                mask=cv2.bitwise_or(mask,mask2)

        a=lab[:,:,1]
        _,a_mask=cv2.threshold(a,lab_a_max,255,cv2.THRESH_BINARY_INV)
        mask=cv2.bitwise_and(mask,a_mask)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones(k_open,np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones(k_close,np.uint8),iterations=2)

        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=[c for c in cnts if cv2.contourArea(c)>=min_component_area_px]

        objs=[]
        for c in cnts:
            area_px=cv2.contourArea(c)
            area_cm2=area_px*area_per_pixel
            objs.append((c,area_cm2))

        kept=[o for o in objs if o[1]>=object_min_area_cm2]
        if not kept and objs:
            largest=max(o[1] for o in objs)
            kept=[o for o in objs if o[1]>=rel_min_frac_of_largest*largest]

        kept=sorted(kept,key=lambda x:x[1],reverse=True)[:max_keep]
        annotated=img.copy()

        for i,(cnt,area_cm2) in enumerate(kept,1):
            perim_cm=_perimeter_cm_from_cnt(cnt,sx,sy)
            cv2.drawContours(annotated,[cnt],-1,(0,255,0),3)
            x,y=cnt[0][0]
            cv2.putText(annotated,f'Obj {i}: {area_cm2:.2f} cm2',
                        (int(x),int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,120,0),2)
            rows.append({
                'File':os.path.basename(path),
                'Object':i,
                'Area_cm2':round(area_cm2,2),
                'Perimeter_cm':round(perim_cm,2)
            })

        out=os.path.join(output_folder,
             os.path.splitext(os.path.basename(path))[0]+'_processed.jpg')
        cv2.imwrite(out,annotated)

        if show_windows:
            cv2.imshow(out,annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(output_folder,'image_processed.csv'),index=False)
    return df
"
  reticulate::py_run_string(py_code)

  df= reticulate::py$process_images(
    normalizePath(input_folder),
    normalizePath(output_folder),
    image_real_cm[1], image_real_cm[2],
    as.integer(lower_hsv),
    as.integer(upper_hsv),
    extra_hsv,   # pass list directly
    as.integer(lab_a_max),
    as.integer(min_component_area_px),
    as.integer(k_open),
    as.integer(k_close),
    object_min_area_cm2,
    rel_min_frac_of_largest,
    as.integer(max_keep),
    show_windows
  )

  reticulate::py_to_r(df)
}
# All Rights Reserved © J.K Kim (kimjk@agronomy4future.com). Last updated on 02/03/2026
