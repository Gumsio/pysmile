# Optional fixture regeneration only; R is not a Python runtime/test dependency.
# Usage: Rscript tests/reference/generate.R SOURCE_DIR OUTPUT_DIR
# SOURCE_DIR must contain the six pinned upstream R files listed in README.md.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Supply SOURCE_DIR and OUTPUT_DIR.")
source_dir <- args[1L]
output_dir <- args[2L]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
options(digits = 17)
write_fixture <- function(data, filename) {
  write.table(data, file.path(output_dir, filename), sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
}
source(file.path(source_dir, "2_function_UScalc_raw.R"))
source(file.path(source_dir, "3_function_USplot.R"))
raw <- data.frame(
  y = c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
  p_ref = c(.10, .35, .40, .20, .60, .60, .85, .40, .50, .70),
  p = c(.05, .40, .40, .20 + 1e-10, .10, .55, .92, .40, .50 - 1e-10, .90)
)
write_fixture(raw, "predictions.tsv")
ref <- list(y = raw$y, p = raw$p_ref, n_vars = 1)
new <- list(y = raw$y, p = raw$p, n_vars = 2)
for (coefficient in c("rLR", "BA", "RB")) {
  result <- UScalc_raw(raw, coefficient, 1)
  points <- ggplot2::ggplot_build(USplot(result, coefficient, net = TRUE, crit = 2))$data
  point_layer <- if (coefficient == "rLR") points[[4L]] else points[[2L]]
  write_fixture(point_layer[c("x", "y", "colour", "fill", "size", "shape")],
                paste0("us_", coefficient, "_points.tsv"))
  write_fixture(data.frame(
    subclass = result$data.Y$subclass, value = result$data.Y$value,
    proportion = result$point_sizes
  ), paste0("us_", coefficient, "_values.tsv"))
  write_fixture(data.frame(
    p_Y_overall = result$p_Y_overall,
    netto_Y_nonev = result$netto_Y_nonev, netto_Y_event = result$netto_Y_event,
    nonev_solid = result$nonev_solid, event_solid = result$event_solid
  ), paste0("us_", coefficient, "_summary.tsv"))
}
source(file.path(source_dir, "3_function_PIWplot.R"))
piw <- PIWplot(ref, new)
write_fixture(ggplot2::ggplot_build(piw)$data[[2L]][c("x", "y", "colour", "fill", "shape", "alpha")],
              "piw_tolerance_points.tsv")
source(file.path(source_dir, "3_function_PRCplot.R"))
piw <- PIWplot(ref, new)
write_fixture(ggplot2::ggplot_build(piw)$data[[2L]][c("x", "y", "colour", "fill", "shape", "alpha")],
              "piw_exact_points.tsv")
source(file.path(source_dir, "3_function_ROCplot.R"))
roc <- ROCplot(ref, new)
write_fixture(roc$plot$data, "roc_curves.tsv")
write_fixture(ggplot2::ggplot_build(roc$plot)$data[[2L]][c("x", "y", "colour", "linetype", "group")],
              "roc_lines.tsv")
write_fixture(roc$results_table, "roc_results.tsv")
source(file.path(source_dir, "3_function_CLBplot.R"))
calibration <- CLBplot(ref, new, n_bins = 5)
write_fixture(calibration$plot$data, "calibration_curves.tsv")
write_fixture(ggplot2::ggplot_build(calibration$plot)$data[[2L]][c("x", "y", "colour", "linetype", "shape", "group")],
              "calibration_lines.tsv")
write_fixture(calibration$results_table, "calibration_results.tsv")
writeLines(c(
  paste("R", getRversion()),
  paste("ggplot2", packageVersion("ggplot2")),
  paste("pROC", packageVersion("pROC"))
), file.path(output_dir, "versions.txt"))
