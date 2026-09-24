args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Usage: prepare_v22_figure_sources.R <project_root> <comparator_dir> <output_root>")
project <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
comparator <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
output <- normalizePath(args[[3]], winslash = "/", mustWork = TRUE)
dir.create(file.path(output, "figures", "source_data"), recursive=TRUE, showWarnings=FALSE)
# Input below must be the exact restricted curve source regenerated inside the private workspace.
old <- read.csv(file.path(project, "04_formal_modeling", "09_figures", "source_data", "Figure_2_roc_pr_source.csv"), check.names = FALSE, fileEncoding = "UTF-8-BOM")
internal <- read.csv(file.path(comparator, "level_only_internal_oof_predictions.csv"), check.names = FALSE)
external <- read.csv(file.path(comparator, "level_only_external_predictions_sicdb.csv"), check.names = FALSE)
internal_mean <- aggregate(model12h_level_only ~ analysis_id + outcome, internal, mean)
roc_points <- function(y, p) {
  ord <- order(p, decreasing = TRUE); y <- y[ord]; p <- p[ord]
  tp <- cumsum(y == 1); fp <- cumsum(y == 0); keep <- which(!duplicated(p, fromLast = TRUE))
  data.frame(curve="ROC", x=c(0, fp[keep]/sum(y==0)), y=c(0, tp[keep]/sum(y==1)), threshold=c(Inf, p[keep]))
}
pr_points <- function(y, p) {
  ord <- order(p, decreasing = TRUE); y <- y[ord]; p <- p[ord]
  tp <- cumsum(y == 1); fp <- cumsum(y == 0); keep <- which(!duplicated(p, fromLast = TRUE))
  data.frame(curve="PR", x=c(0, tp[keep]/sum(y==1)), y=c(sum(y==1)/length(y), tp[keep]/(tp[keep]+fp[keep])), threshold=c(Inf, p[keep]))
}
make_curves <- function(dat, dataset, representation) {
  z <- rbind(roc_points(dat$outcome, dat$model12h_level_only), pr_points(dat$outcome, dat$model12h_level_only))
  data.frame(dataset=dataset, representation=representation, model="model12h_level_only", z, check.names=FALSE)
}
new_curves <- rbind(make_curves(internal_mean, "MIMIC-IV internal OOF", "patient_mean_5_repeats"), make_curves(external, "SICdb external", "hourly_median"))
write.csv(rbind(old[, c("dataset","representation","model","curve","x","y","threshold")], new_curves[, c("dataset","representation","model","curve","x","y","threshold")]), file.path(output, "figures", "source_data", "Figure_2_roc_pr_source.csv"), row.names=FALSE, na="")

