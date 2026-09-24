suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(scales)
  library(svglite)
  library(ragg)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: render_final_figures.R <project_root> <output_root>")
project <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output <- normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
source_root <- file.path(project, "figures", "source_data")
table_root <- file.path(project, "tables")
fig_dir <- file.path(output, "figures")
sup_fig_dir <- file.path(output, "supplement", "figures")
ga_dir <- file.path(output, "18_GRAPHICAL_ABSTRACT")
dir.create(file.path(fig_dir, "source_data"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(sup_fig_dir, "source_data"), recursive = TRUE, showWarnings = FALSE)
dir.create(ga_dir, recursive = TRUE, showWarnings = FALSE)

rcsv <- function(name) read.csv(file.path(source_root, name), check.names = FALSE, fileEncoding = "UTF-8-BOM")
copy_source <- function(name, supplementary = FALSE) {
  destination <- if (supplementary) file.path(sup_fig_dir, "source_data", name) else file.path(fig_dir, "source_data", name)
  if (!file.copy(file.path(source_root, name), destination, overwrite = TRUE)) stop(paste("Could not copy", name))
}

model_colours <- c(
  model0_snapshot = "#777777",
  model12h_level_only = "#D08428",
  model1_elastic_net = "#1F5A9D",
  model2_xgboost = "#16857A",
  model3_grud = "#7655A5",
  weighted = "#7655A5",
  unweighted = "#16857A",
  treat_all = "#9CA3AF",
  treat_none = "#111827"
)
model_labels <- c(
  model0_snapshot = "Snapshot logistic",
  model12h_level_only = "12-h level-only penalized logistic",
  model1_elastic_net = "Full temporal-summary L1 logistic",
  model2_xgboost = "XGBoost temporal summaries",
  model3_grud = "Compact GRU-D"
)
signal_colours <- c(HR = "#C44E52", MAP = "#1F5A9D", SpO2 = "#16857A")
signal_labels <- c(HR = "HR >100", MAP = "MAP <65", SpO2 = "SpO2 <92")

theme_final <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35), axis.ticks = element_line(linewidth = 0.35),
      axis.text = element_text(colour = "#202020"), strip.background = element_rect(fill = "#EEF3F8", colour = NA),
      strip.text = element_text(face = "bold"), plot.tag = element_text(face = "bold", size = 10),
      plot.title = element_text(face = "bold", size = 9), plot.subtitle = element_text(size = 7.4, colour = "#404040"),
      legend.position = "bottom", legend.key.width = unit(12, "pt"), panel.grid.major.y = element_line(colour = "#E6E9ED", linewidth = 0.25),
      panel.grid.minor = element_blank(), plot.margin = margin(5, 7, 5, 7)
    )
}
theme_set(theme_final())

save_publication <- function(plot, stem, width_mm = 183, height_mm = 125) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4
  svglite::svglite(paste0(stem, ".svg"), width = width_in, height = height_in, bg = "white")
  print(plot); dev.off()
  grDevices::cairo_pdf(paste0(stem, ".pdf"), width = width_in, height = height_in, family = "Arial", bg = "white")
  print(plot); dev.off()
  ragg::agg_tiff(paste0(stem, ".tiff"), width = width_in, height = height_in, units = "in", res = 600,
                 background = "white", compression = "lzw")
  print(plot); dev.off()
  ragg::agg_png(paste0(stem, ".png"), width = width_in, height = height_in, units = "in", res = 300,
                background = "white")
  print(plot); dev.off()
}

# Figure 1: two-column cohort construction plus common landmark timeline.
flow_raw <- rcsv("Figure_1_cohort_flow_source.csv")
flow_keep <- c("raw_icu", "major_abdominal_surgery_evidence", "postoperative_aligned", "first_eligible",
               "landmark_alive_and_in_icu", "final_n")
flow_labels <- c(
  raw_icu = "ICU admissions screened",
  major_abdominal_surgery_evidence = "Major abdominal surgery evidence",
  postoperative_aligned = "Postoperative alignment met",
  first_eligible = "First eligible ICU admission",
  landmark_alive_and_in_icu = "Alive and in ICU at 12-h landmark",
  final_n = "Final analysis cohort"
)
flow <- flow_raw |>
  filter(stage %in% flow_keep) |>
  mutate(stage = factor(stage, levels = rev(flow_keep)), label = paste0(flow_labels[as.character(stage)], "\nN = ", comma(n)))
events <- flow_raw |> filter(stage == "death28_events") |> transmute(dataset, events = n)
flow <- left_join(flow, events, by = "dataset")
p_flow <- ggplot(flow, aes(x = 1, y = stage)) +
  geom_segment(aes(x = 1, xend = 1, yend = dplyr::lag(as.numeric(stage))), colour = "#9AA5B1", linewidth = 0.45,
               arrow = arrow(length = unit(2.2, "mm"), type = "closed"), na.rm = TRUE) +
  geom_label(aes(label = label), size = 2.55, label.size = 0.3, label.padding = unit(0.16, "lines"),
             fill = "white", colour = "#1D2A38", lineheight = 0.95) +
  facet_wrap(~dataset, nrow = 1) +
  annotate("text", x = 1, y = 0.52, label = "", size = 1) +
  scale_y_discrete(drop = FALSE) + coord_cartesian(clip = "off") +
  labs(title = "Cohort construction", subtitle = "Identical 12-hour landmark logic; database-specific, audited surgery mappings") +
  theme_void(base_family = "Arial", base_size = 8) +
  theme(strip.text = element_text(face = "bold", size = 10), plot.title = element_text(face = "bold", size = 9),
        plot.subtitle = element_text(size = 7.3), plot.margin = margin(5, 10, 0, 10))

timeline <- data.frame(xmin = c(0, 1), xmax = c(1, 2), ymin = c(0.25, 0.25), ymax = c(0.75, 0.75),
                       segment = c("Predictor window", "Outcome window"))
p_time <- ggplot(timeline) +
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = segment), colour = "white", linewidth = 0.6) +
  geom_vline(xintercept = 1, linewidth = 0.65, colour = "#1D2A38") +
  annotate("text", x = 0.5, y = 0.5, label = "0-12 h postoperative physiology", size = 3.0, colour = "white", fontface = "bold") +
  annotate("text", x = 1.5, y = 0.5, label = "12 h landmark to day 28 mortality", size = 3.0, colour = "white", fontface = "bold") +
  annotate("text", x = 1, y = 0.93, label = "Landmark: alive and still in ICU", size = 2.8, fontface = "bold") +
  scale_fill_manual(values = c("Predictor window" = "#1F5A9D", "Outcome window" = "#16857A"), guide = "none") +
  scale_x_continuous(breaks = c(0, 1, 2), labels = c("ICU admission\n0 h", "12 h", "Day 28"), expand = c(0.02, 0.02)) +
  coord_cartesian(ylim = c(0, 1.05), clip = "off") + labs(x = NULL, y = NULL) +
  theme_void(base_family = "Arial", base_size = 8) + theme(axis.text.x = element_text(size = 7.5), plot.margin = margin(0, 18, 4, 18))
fig1 <- p_flow / p_time + plot_layout(heights = c(4.2, 1)) +
  plot_annotation(title = "Postoperative cohorts and landmark design", tag_levels = "a")
save_publication(fig1, file.path(fig_dir, "Figure_1_cohort_flow_landmark"), 183, 154)
copy_source("Figure_1_cohort_flow_source.csv")

# Figure 2: ROC and precision-recall curves, with the prespecified temporal-summary contrast visually prioritised.
curves <- rcsv("Figure_2_roc_pr_source.csv") |>
  filter(model %in% names(model_labels)) |>
  mutate(dataset = recode(dataset, `MIMIC-IV internal OOF` = "MIMIC-IV internal OOF", `SICdb external` = "SICdb locked external"),
         curve = factor(curve, levels = c("ROC", "PR")),
         model = factor(model, levels = names(model_labels)))
curve_lw <- c(model0_snapshot = 0.8, model12h_level_only = 0.95, model1_elastic_net = 1.25, model2_xgboost = 0.55, model3_grud = 0.55)
pr_base <- data.frame(dataset = c("MIMIC-IV internal OOF", "SICdb locked external"), curve = factor("PR", levels = c("ROC", "PR")),
                      y = c(168 / 1740, 185 / 2376))
p2 <- ggplot(curves, aes(x, y, colour = model, linewidth = model, alpha = model)) +
  geom_abline(data = data.frame(dataset = unique(curves$dataset), curve = factor("ROC", levels = c("ROC", "PR"))),
              slope = 1, intercept = 0, linetype = "dashed", colour = "#B0B7BF", linewidth = 0.35, inherit.aes = FALSE) +
  geom_hline(data = pr_base, aes(yintercept = y), linetype = "dashed", colour = "#B0B7BF", linewidth = 0.35, inherit.aes = FALSE) +
  geom_path() + facet_grid(curve ~ dataset) + coord_equal() +
  scale_colour_manual(values = model_colours, labels = model_labels, name = NULL) +
  scale_linewidth_manual(values = curve_lw, guide = "none") +
  scale_alpha_manual(values = c(model0_snapshot = 0.9, model12h_level_only = 1, model1_elastic_net = 1, model2_xgboost = 0.55, model3_grud = 0.55), guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) + scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
  labs(title = "Temporal information beyond the admission snapshot",
       subtitle = "Full temporal summaries are compared with the admission snapshot and 12-h updated levels",
       x = "False-positive rate (ROC) / Recall (PR)", y = "True-positive rate (ROC) / Precision (PR)")
save_publication(p2, file.path(fig_dir, "Figure_2_temporal_vs_snapshot_roc_pr"), 183, 133)
copy_source("Figure_2_roc_pr_source.csv")

# Figure 3: locked external calibration for snapshot and the two temporal-summary models.
cal <- rcsv("Figure_3_calibration_source.csv") |>
  filter(representation == "hourly_median", model %in% c("model0_snapshot", "model1_elastic_net", "model2_xgboost")) |>
  mutate(model = factor(model, levels = c("model0_snapshot", "model1_elastic_net", "model2_xgboost")))
lim3 <- max(c(cal$mean_predicted, cal$observed), na.rm = TRUE) * 1.05
p3 <- ggplot(cal, aes(mean_predicted, observed, colour = model, group = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.45, colour = "#7B8794") +
  geom_line(linewidth = 0.8) + geom_point(aes(size = n), alpha = 0.9) +
  scale_colour_manual(values = model_colours, labels = model_labels, name = NULL) + scale_size_continuous(range = c(1.5, 3.2), guide = "none") +
  coord_equal(xlim = c(0, lim3), ylim = c(0, lim3)) +
  labs(title = "Calibration in locked SICdb external validation", subtitle = "Decile bins; N = 2,376, deaths = 185. GRU-D sensitivity: Figure S3.",
       x = "Mean predicted 28-day mortality", y = "Observed 28-day mortality")
save_publication(p3, file.path(fig_dir, "Figure_3_external_calibration"), 120, 100)
copy_source("Figure_3_calibration_source.csv")

# Figure 4: discrimination-focused resolution experiment plus paired 5-minute minus 60-minute contrasts.
res <- rcsv("Figure_4_resolution_performance_source.csv") |>
  filter(metric %in% c("auroc", "auprc")) |>
  mutate(metric = factor(metric, levels = c("auroc", "auprc"), labels = c("AUROC", "AUPRC")), model = factor(model, levels = names(model_labels)))
diffs <- rcsv("Figure_4_bootstrap_source.csv") |>
  filter(metric %in% c("auroc", "auprc")) |>
  mutate(metric = factor(metric, levels = c("auroc", "auprc"), labels = c("AUROC", "AUPRC")),
         model_label = factor(model_labels[model], levels = rev(model_labels[c("model1_elastic_net", "model2_xgboost", "model3_grud")])) )
p4a <- ggplot(res, aes(resolution_min, estimate, colour = model, group = model)) +
  geom_line(linewidth = 0.7) + geom_point(size = 1.8) + geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 1.8, linewidth = 0.4) +
  facet_wrap(~metric, scales = "free_y") + scale_x_continuous(breaks = c(5, 15, 60)) +
  scale_colour_manual(values = model_colours, labels = model_labels, name = NULL) +
  labs(x = "Sampling resolution (minutes)", y = "Estimate (95% CI)")
p4b <- ggplot(diffs, aes(estimate, model_label, colour = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, colour = "#7B8794") +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.17, linewidth = 0.45) + geom_point(size = 1.9) +
  facet_wrap(~metric, scales = "free_x") + scale_colour_manual(values = model_colours, guide = "none") +
  labs(x = "Paired 5-min minus 60-min difference (95% CI)", y = NULL)
fig4 <- p4a / p4b + plot_layout(heights = c(1.15, 1)) +
  plot_annotation(title = "Same-patient sampling-resolution performance",
                  subtitle = "SICdb dense core: N = 2,061, deaths = 151; repeated 5 x 5 OOF evaluation", tag_levels = "a")
save_publication(fig4, file.path(fig_dir, "Figure_4_sampling_resolution_performance"), 183, 142)
copy_source("Figure_4_resolution_performance_source.csv")
copy_source("Figure_4_bootstrap_source.csv")

# Figure 5: population-level information loss at 60-minute aggregation.
loss <- rcsv("Figure_5_measurement_loss_source.csv") |>
  filter(cohort == "dense_core", resolution_min == 60) |>
  mutate(signal = factor(signal, levels = c("HR", "MAP", "SpO2")))
loss_long <- bind_rows(
  transmute(loss, signal, metric = "Positive-status loss (%)", value = event_miss_pct, unit = "%"),
  transmute(loss, signal, metric = "SD retained (%)", value = 100 * sd_retention_median, unit = "%"),
  transmute(loss, signal, metric = "Median extreme attenuation", value = extreme_attenuation_median, unit = "signal units"),
  transmute(loss, signal, metric = "90th percentile burden error (pp)", value = burden_absolute_error_p90_pp, unit = "percentage points")
) |>
  mutate(metric = factor(metric, levels = c("Positive-status loss (%)", "SD retained (%)", "Median extreme attenuation", "90th percentile burden error (pp)")))
p5 <- ggplot(loss_long, aes(signal, value, fill = signal)) +
  geom_col(width = 0.68) + geom_text(aes(label = sprintf("%.1f", value)), vjust = -0.35, size = 2.6) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) + scale_fill_manual(values = signal_colours, guide = "none") +
  scale_x_discrete(labels = signal_labels) + expand_limits(y = 0) +
  labs(title = "Information lost by hourly aggregation", subtitle = "SICdb dense core (N = 2,061); one-minute recordings as reference",
       x = NULL, y = NULL) + theme(axis.text.x = element_text(angle = 25, hjust = 1))
save_publication(p5, file.path(fig_dir, "Figure_5_measurement_information_loss"), 183, 92)
copy_source("Figure_5_measurement_loss_source.csv")

# Figure S1: decision curve analysis.
dca <- rcsv("Supplement_Figure_S1_decision_curve_source.csv")
dca_labels <- c(model_labels, treat_all = "Treat all", treat_none = "Treat none")
pS1 <- ggplot(dca, aes(threshold, net_benefit, colour = model, linetype = model)) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "#D1D5DB") + geom_line(linewidth = 0.7) +
  scale_colour_manual(values = model_colours, labels = dca_labels, name = NULL) +
  scale_linetype_manual(values = c(model0_snapshot = "solid", model1_elastic_net = "solid", model2_xgboost = "solid", model3_grud = "solid", treat_all = "dashed", treat_none = "dotted"), labels = dca_labels, name = NULL) +
  scale_x_continuous(breaks = c(.02, .05, .10, .15, .20, .25), labels = percent_format(accuracy = 1)) +
  labs(title = "Decision-curve analysis", subtitle = "Locked SICdb external validation; N = 2,376, deaths = 185",
       x = "Risk threshold", y = "Net benefit")
save_publication(pS1, file.path(sup_fig_dir, "Figure_S1_decision_curve"), 120, 90)
copy_source("Supplement_Figure_S1_decision_curve_source.csv", TRUE)

# Figure S2: all prespecified sparse phases for weighted GRU-D.
sp <- rcsv("Supplement_Figure_S2_sparse_phase_source.csv") |>
  filter(model == "model3_grud") |>
  mutate(phase = factor(sub("sparse_phase_", "", representation), levels = c("00", "15", "30", "45", "59")),
         metric = factor(metric, levels = c("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope"),
                         labels = c("AUROC", "AUPRC", "Brier score", "Log loss", "Calibration intercept", "Calibration slope")))
refs <- data.frame(metric = factor(c("Calibration intercept", "Calibration slope"), levels = levels(sp$metric)), ref = c(0, 1))
pS2 <- ggplot(sp, aes(phase, estimate, group = 1)) +
  geom_hline(data = refs, aes(yintercept = ref), linetype = "dashed", linewidth = 0.35, colour = "#9CA3AF", inherit.aes = FALSE) +
  geom_line(linewidth = 0.65, colour = model_colours[["model3_grud"]]) + geom_point(size = 1.8, colour = model_colours[["model3_grud"]]) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.12, linewidth = 0.4, colour = model_colours[["model3_grud"]]) +
  facet_wrap(~metric, scales = "free_y", ncol = 3) +
  labs(title = "Prespecified sparse-phase sensitivity", subtitle = "Weighted GRU-D; one minute sampled per hour at phases 00, 15, 30, 45, and 59",
       x = "Minute phase", y = "Estimate (95% CI)")
save_publication(pS2, file.path(sup_fig_dir, "Figure_S2_sparse_phase_sensitivity"), 183, 115)
copy_source("Supplement_Figure_S2_sparse_phase_source.csv", TRUE)

# Figure S3: weighted versus fixed unweighted GRU-D external calibration.
wu <- rcsv("Supplement_Figure_S3_weighted_unweighted_calibration_source.csv") |>
  mutate(model = factor(model, levels = c("weighted", "unweighted"), labels = c("Weighted BCE", "Unweighted BCE")))
limS3 <- max(c(wu$mean_predicted, wu$observed), na.rm = TRUE) * 1.05
pS3 <- ggplot(wu, aes(mean_predicted, observed, colour = model, group = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.4, colour = "#7B8794") +
  geom_line(linewidth = 0.75) + geom_point(aes(size = n), alpha = 0.9) +
  scale_colour_manual(values = c("Weighted BCE" = model_colours[["weighted"]], "Unweighted BCE" = model_colours[["unweighted"]]), name = NULL) +
  scale_size_continuous(range = c(1.6, 3.2), guide = "none") + coord_equal(xlim = c(0, limS3), ylim = c(0, limS3)) +
  labs(title = "GRU-D loss-weighting sensitivity", subtitle = "Locked SICdb hourly-median external validation",
       x = "Mean predicted probability", y = "Observed event proportion")
save_publication(pS3, file.path(sup_fig_dir, "Figure_S3_weighted_unweighted_grud_calibration"), 115, 95)
copy_source("Supplement_Figure_S3_weighted_unweighted_calibration_source.csv", TRUE)

# Figure S4: true standardized L1-penalized logistic-regression coefficients; no confidence intervals are implied.
imp <- rcsv("Supplement_Figure_S4_feature_importance_source.csv") |>
  filter(elastic_net_absolute_coefficient > 0) |>
  arrange(desc(elastic_net_absolute_coefficient)) |>
  slice_head(n = 15) |>
  mutate(feature = factor(feature, levels = rev(feature)), direction = ifelse(elastic_net_standardized_coefficient >= 0, "Positive", "Negative"))
pS4 <- ggplot(imp, aes(elastic_net_standardized_coefficient, feature, fill = direction)) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "#7B8794") + geom_col(width = 0.72) +
  scale_fill_manual(values = c(Positive = "#1F5A9D", Negative = "#C56A2D"), name = NULL) +
  labs(title = "L1-penalized logistic temporal-feature coefficients", subtitle = "Locked standardized coefficients; descriptive only; no coefficient CI was estimated",
       x = "Standardized coefficient", y = NULL)
save_publication(pS4, file.path(sup_fig_dir, "Figure_S4_elastic_net_coefficients"), 140, 105)
copy_source("Supplement_Figure_S4_feature_importance_source.csv", TRUE)

# Critical Care graphical abstract, exact 920 x 300 px review PNG plus vector SVG/PDF.
ga <- ggplot() +
  annotate("rect", xmin = 0, xmax = 31.5, ymin = 5, ymax = 29, fill = "#EAF1F8", colour = "#B8C8D8") +
  annotate("rect", xmin = 34.25, xmax = 65.75, ymin = 5, ymax = 29, fill = "#EDF7F5", colour = "#B2D5CF") +
  annotate("rect", xmin = 68.5, xmax = 100, ymin = 5, ymax = 29, fill = "#F5F0FA", colour = "#CDBBDD") +
  annotate("text", x = 15.75, y = 26.6, label = "DEVELOPMENT", fontface = "bold", size = 3.8, colour = "#1F5A9D") +
  annotate("text", x = 15.75, y = 22.7, label = "Major abdominal surgery", fontface = "bold", size = 4.2) +
  annotate("text", x = 15.75, y = 18.5, label = "MIMIC-IV  N = 1,740", size = 3.7) +
  annotate("segment", x = 5, xend = 26.5, y = 12.8, yend = 12.8, linewidth = 2.1, colour = "#1F5A9D") +
  annotate("segment", x = 26.5, xend = 26.5, y = 11.6, yend = 14, linewidth = 0.8, colour = "#1D2A38") +
  annotate("text", x = 15.75, y = 9.1, label = "0-12 h physiology -> day 28", size = 3.4) +
  annotate("text", x = 50, y = 26.6, label = "EARLY-COURSE INFORMATION", fontface = "bold", size = 3.8, colour = "#16857A") +
  annotate("text", x = 50, y = 21.8, label = "Snapshot  vs  12-h summaries", fontface = "bold", size = 4.0) +
  annotate("text", x = 50, y = 15.8, label = "+0.087", fontface = "bold", size = 8.0, colour = "#1F5A9D") +
  annotate("text", x = 50, y = 10.6, label = "Delta AUROC (95% CI 0.054-0.120)", size = 3.4) +
  annotate("text", x = 84.25, y = 26.6, label = "EXTERNAL VALIDATION", fontface = "bold", size = 3.8, colour = "#7655A5") +
  annotate("text", x = 84.25, y = 22.2, label = "SICdb  N = 2,376", fontface = "bold", size = 4.0) +
  annotate("text", x = 84.25, y = 17.3, label = "+0.040", fontface = "bold", size = 7.0, colour = "#7655A5") +
  annotate("text", x = 84.25, y = 13.2, label = "95% CI 0.019-0.063", size = 3.4) +
  annotate("text", x = 84.25, y = 8.7, label = "Finer sampling retained more instability", size = 3.25) +
  annotate("segment", x = 31.8, xend = 34, y = 17, yend = 17, arrow = arrow(type = "closed", length = unit(2.2, "mm")), linewidth = 0.7, colour = "#5A6570") +
  annotate("segment", x = 66, xend = 68.2, y = 17, yend = 17, arrow = arrow(type = "closed", length = unit(2.2, "mm")), linewidth = 0.7, colour = "#5A6570") +
  annotate("rect", xmin = 0, xmax = 100, ymin = 0, ymax = 4.2, fill = "#1D2A38", colour = NA) +
  annotate("text", x = 50, y = 2.1, label = "Twelve-hour summaries improved discrimination beyond admission snapshots; finer recordings captured additional instability with smaller prediction gains.",
           colour = "white", size = 2.75, fontface = "bold") +
  coord_cartesian(xlim = c(0, 100), ylim = c(0, 30), clip = "off") + theme_void(base_family = "Arial")
svglite::svglite(file.path(ga_dir, "Graphical_Abstract_CC.svg"), width = 920 / 96, height = 300 / 96, bg = "white")
print(ga); dev.off()
grDevices::cairo_pdf(file.path(ga_dir, "Graphical_Abstract_CC.pdf"), width = 920 / 96, height = 300 / 96, family = "Arial", bg = "white")
print(ga); dev.off()
ragg::agg_png(file.path(ga_dir, "Graphical_Abstract_CC.png"), width = 920, height = 300, units = "px", res = 96, background = "white")
print(ga); dev.off()

status <- data.frame(
  figure = c(paste0("Figure_", 1:5), paste0("Figure_S", 1:4), "Graphical_Abstract"),
  backend = "R 4.5.2", source = "aggregate_csv_only", pdf = TRUE, svg = TRUE,
  tiff_600dpi = c(rep(TRUE, 9), FALSE), png_300dpi = c(rep(TRUE, 9), TRUE), status = "PASS"
)
write.csv(status, file.path(output, "QC", "FIGURE_RENDER_STATUS.csv"), row.names = FALSE, fileEncoding = "UTF-8")
message("FINAL_FIGURES_SUCCESS")
