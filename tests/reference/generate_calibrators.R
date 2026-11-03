# Optional calibration fixtures from the pinned R source; no R packages needed.
# Usage: Rscript tests/reference/generate_calibrators.R SOURCE_DIR OUTPUT_DIR
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Supply SOURCE_DIR and OUTPUT_DIR.")
source(file.path(args[1L], "1_function_USclbr_mdl.R"))
dir.create(args[2L], recursive = TRUE, showWarnings = FALSE)
options(digits = 17)
cases <- list(
  ordinary = data.frame(y = c(0, 0, 0, 1, 0, 1, 1, 1),
                       p = c(.05, .20, .35, .40, .55, .60, .75, .90)),
  ties = data.frame(y = c(0, 1, 0, 1, 0, 1, 0, 1),
                   p = c(.2, .2, .2, .4, .4, .6, .8, .8)),
  endpoints = data.frame(y = c(0, 1, 0, 1, 0, 1, 0, 1),
                        p = c(0, .1, .2, .4, .6, .8, .9, 1))
)
inputs <- list()
outputs <- list()
for (case_name in names(cases)) {
  input <- cases[[case_name]]
  inputs[[case_name]] <- data.frame(case = case_name, input)
  # Include observed knots, intermediate probabilities, and both endpoints.
  queries <- sort(unique(c(seq(0, 1, by = .05), input$p)))
  for (method in c("logistic", "intercept", "isotonic")) {
    fit <- USfit_calibrator(input$y, input$p, method = method)
    outputs[[paste(case_name, method)]] <- data.frame(
      case = case_name, method = method, p = queries,
      calibrated = USapply_calibrator(fit, queries)
    )
  }
}
write.table(do.call(rbind, inputs), file.path(args[2L], "calibrator_inputs.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
write.table(do.call(rbind, outputs), file.path(args[2L], "calibrator_results.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
