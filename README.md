<!-- README.md is generated from README.Rmd. Please edit that file -->
# greencapture
<!-- badges: start -->
<!-- badges: end -->

The goal of the greencapture package is to provides tools for capturing, segmenting, 
and quantifying green plant tissue area from digital images.

□ Code explained: https://agronomy4future.com/archives/24628

## Installation

You can install datacume() like so:

Before installing, please download Rtools (https://cran.r-project.org/bin/windows/Rtools)

``` r
if(!require(remotes)) install.packages("remotes")
if (!requireNamespace("greencapture", quietly = TRUE)) {
  remotes::install_github("agronomy4future/greencapture", force= TRUE)
}
library(remotes)
library(greencapture)
```

## Example

This is a basic code for datacume()

``` r
greencapture(
  input_folder = r"(C:/)",
  output_folder= r"(C:/)",
  image_real_cm= c(20, 20),
  object_min_area_cm2 = 10.0,
  show_windows = FALSE
)
