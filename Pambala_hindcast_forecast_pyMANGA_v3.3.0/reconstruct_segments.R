################################################################################
# Band-wise MTPPR reconstruction of the 1997 adult stand from PCQM quarters.
#
# Three choices define the procedure:
#
#   Density   The transect strip is a PCQM sample, not a census, so its point
#             density is not the stand density. The number of trees per cell
#             is set from the PCQM distances instead (Cottam & Curtis 1956:
#             density = 1 / mean distance^2), scaled by the share of model
#             species among the recorded adults.
#   Zonation  Each transect band (layout "bands", used for the 1997 stand) is
#             reconstructed from its own PCQM trees only, so species and sizes
#             stay in the band where they were recorded. Layout "segments" cuts
#             every band into five segments along x.
#   Marks     Generated trees are whole PCQM trees (species, stem radius and
#             height together) resampled within the segment. MTPPR arranges
#             them so that the mark correlation functions of the segment's
#             PCQM trees are reproduced; their positions are uniform in the
#             cell, because nearest-tree-per-quarter sampling carries no
#             information on the pair correlation function of the stand.
#
# The quarter around a sample point out to the recorded tree holds no other
# adult by definition of the method. These quarter-discs (and vacant quarters
# out to r_vacant) are therefore kept free of generated trees.
#
# The MTPPR v2.0.0 engine is used as published, with two substitutions made
# when it is loaded (load_engine): the initial number of points can be given,
# and the energy can compare pair statistics normalised by the pair density.
#
# Usage (as run for the 1997 stand, S0):
#   Rscript reconstruct_segments.R pcqm_1997_quarters.csv Initialization/reconstructed_raw_1997.csv bands
#   quarters_csv  from pcqm_quarters_1997.py
#   layout        "bands" = one cell per transect band; "segments" = five cells per band
# Random seed per cell: set.seed(1000 * segment + band). Then postprocess_csv.py.
# A per-cell log is written next to the output (<output>_log.csv).
################################################################################

cli_args <- commandArgs(trailingOnly = TRUE)

# --- Configuration -----------------------------------------------------------

steps_per_tree <- 100     # MTPPR steps per tree in the cell
r_vacant       <- 5.0     # a vacant quarter holds no adult out to this distance (m)
min_reference  <- 8       # cells with fewer PCQM trees borrow from the next band
n_cores        <- max(1L, parallel::detectCores() - 2L)

# Simulation domain (from d_2_p.xml) and band limits (as in reconstruct_bands.R)
domain <- list(x = c(-17.9, 282.1), y = c(-18.1, 95.0))
zone_y <- c(domain$y[1], 12.50, 37.49, 62.54, domain$y[2])
# Segment limits halfway between sample points 5|6, 10|11, 15|16 and 20|21.
layout <- if (length(cli_args) >= 3) cli_args[3] else "segments"
stopifnot(layout %in% c("segments", "bands"))
seg_x  <- if (layout == "bands") domain$x else c(domain$x[1], 45, 95, 145, 195, domain$x[2])

species_levels <- c("Avicennia", "Bruguiera", "Excoecaria", "Lumnitzera", "Rhizophora")

# --- Paths -------------------------------------------------------------------

# Resolve script directory (works under both Rscript and source()).
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(sub("^--file=", "", file_arg))
    else getwd()
  }
)

# MTPPR engine (Wudel et al.), not part of this package: download
# https://github.com/ChrisWudel/Multi-trait-point-pattern-reconstruction (v2.0.0, commit 6982c39)
# and set MTPPR_PATH to its func/MTPPR-v.2.0.0.R, or place the repository folder as MTPPR/ next to this script.
mtppr_path <- Sys.getenv("MTPPR_PATH", file.path(script_dir, "MTPPR", "func", "MTPPR-v.2.0.0.R"))
stopifnot(file.exists(mtppr_path))

csv_path <- if (length(cli_args) >= 1 && nchar(cli_args[1]) > 0) cli_args[1] else
  file.path(script_dir, "pcqm_1997_quarters.csv")
out_path <- if (length(cli_args) >= 2 && nchar(cli_args[2]) > 0) cli_args[2] else
  file.path(script_dir, "Initialization", "reconstructed_raw_1997.csv")
stopifnot(file.exists(csv_path))

# --- MTPPR engine ------------------------------------------------------------

# Source the engine with two substitutions. Each target must occur exactly once,
# so a changed engine file fails here instead of running unpatched.
load_engine <- function(path, envir = globalenv()) {
  code <- paste(readLines(path, warn = FALSE), collapse = "\n")
  substitute_once <- function(code, old, new) {
    stopifnot(lengths(regmatches(code, gregexpr(old, code, fixed = TRUE))) == 1)
    sub(old, new, code, fixed = TRUE)
  }
  # New arguments of construct().
  code <- substitute_once(code,
    "  fixed            = function(p) FALSE,",
    "  fixed            = function(p) FALSE,\n  n_initial        = NULL,\n  normalise_pairs  = FALSE,")
  # Expected number of initial points, if given.
  code <- substitute_once(code,
    "rpois(1, num(p_) *\n    prod(wnd$max-wnd$min) / prod(core_wnd$max-core_wnd$min))",
    paste0("rpois(1, if (is.null(n_initial)) num(p_) *\n",
           "    prod(wnd$max-wnd$min) / prod(core_wnd$max-core_wnd$min) else n_initial)"))
  # Rescale the pair statistics at every r so that the pair density (first row,
  # mark 1 x mark 1) equals that of the reference. What is then compared are the
  # mark correlation functions, weighted by the reference pair density.
  code <- substitute_once(code,
    "    if (!n || !n_) return(Inf)\n",
    paste0("    if (!n || !n_) return(Inf)\n",
           "    if (normalise_pairs) f <- f * rep((f_[1L,]*wedge_/n_) /\n",
           "      pmax(f[1L,]*wedge/n, .Machine$double.eps), each=nrow(f))\n"))
  eval(parse(text = code), envir = envir)
}

# --- Read input --------------------------------------------------------------

quarters <- read.csv(csv_path, stringsAsFactors = FALSE)
quarters$model   <- as.logical(quarters$model)
quarters$band    <- quarters$transect - 1L
quarters$segment <- findInterval(quarters$xp, seg_x)

# --- One cell (band x segment) -----------------------------------------------

# Quarter-discs that hold no generated tree: out to the recorded tree, or to
# r_vacant in a vacant quarter. Quadrants as in pcqm_to_demo.py (a = 0-90 deg).
exclusion_sectors <- function(q) {
  q <- q[q$status %in% c("tree", "vacant"), ]
  data.frame(
    xp = q$xp, yp = q$yp,
    east  = q$quadrant %in% c("a", "d"),
    north = q$quadrant %in% c("a", "b"),
    r2 = ifelse(q$status == "vacant", r_vacant, q$radius + 1e-6)^2)
}

make_fixed <- function(sectors, observed) {
  function(p) {
    px <- p$points[, 1]; py <- p$points[, 2]; n <- length(px)
    dx <- outer(px, sectors$xp, "-"); dy <- outer(py, sectors$yp, "-")
    in_sector <- dx^2 + dy^2 <= rep(sectors$r2, each = n) &
      (dx >= 0) == rep(sectors$east, each = n) &
      (dy >= 0) == rep(sectors$north, each = n)
    rowSums(in_sector) > 0 | (px %in% observed$x & py %in% observed$y)
  }
}

reconstruct_cell <- function(job) {
  b <- job$band; s <- job$segment
  set.seed(1000 * s + b)
  in_cell <- function(q, band) q[q$band == band & q$segment == s, ]
  own <- in_cell(quarters, b)
  observed <- own[!is.na(own$x), ]

  # Reference trees and density come from the cell's own sample points, or from
  # the same segment of the neighbouring band if the transect does not reach it.
  donor <- b
  if (nrow(observed) < min_reference) {
    donor <- b + if (b == 3) -1L else 1L
    observed <- observed[0, ]
  }
  src <- in_cell(quarters, donor)
  ref <- src[!is.na(src$x), ]
  trees <- src[src$status == "tree", ]
  density <- mean(trees$model) / mean(trees$distance_m)^2

  wnd <- list(min = c(seg_x[s], zone_y[b + 1]), max = c(seg_x[s + 1], zone_y[b + 2]))
  area <- prod(wnd$max - wnd$min)
  near <- quarters[quarters$band == b & donor == b &
                   quarters$xp > wnd$min[1] - 10 & quarters$xp < wnd$max[1] + 10, ]
  fixed <- make_fixed(exclusion_sectors(near), observed)

  # Points are drawn uniformly in the cell and those inside a quarter-disc are
  # dropped, so the draw is inflated by the excluded share of the cell.
  probe <- list(points = cbind(runif(20000, wnd$min[1], wnd$max[1]),
                               runif(20000, wnd$min[2], wnd$max[2])))
  excluded <- mean(fixed(probe))
  n_target <- density * area
  n_initial <- max(0, n_target - nrow(observed)) / (1 - excluded)

  # The reference is a copy of the PCQM trees shifted far outside the cell, so
  # that it is not confused with the observed trees kept inside the cell.
  ref_points <- cbind(x = ref$x + 1e5, y = ref$y)
  marked_pattern <- list(
    points = rbind(ref_points, cbind(x = observed$x, y = observed$y)),
    marks = list(
      dbh     = c(ref$r_stem, observed$r_stem) * 20,     # dm, of order 1 like the dummies
      height  = c(ref$h_stem, observed$h_stem) / 10,     # 10 m
      species = factor(c(ref$species, observed$species), levels = species_levels)
    ),
    core_wnd = list(min = apply(ref_points, 2, min) - 0.01,
                    max = apply(ref_points, 2, max) + 0.01),
    wnd = wnd
  )

  log <- list()
  result <- construct_pattern(
    marked_pattern,
    fixed = fixed,
    n_initial = n_initial,
    normalise_pairs = TRUE,
    # Only whole trees change place; the points themselves stay put.
    probabilities = c(move_point = 0, exchange_coords = 1, exchange_mark = 0,
                      draw_mark = 0, delete_point = 0, add_point = 0),
    # Distances beyond half the length of the reference strip are not used:
    # too few of its pairs are that far apart for a stable estimate.
    edge_correction = function(r, wnd) {
      w <- edge_correction_OhserStoyan(r, wnd)
      replace(w, !is.finite(w) | r > 0.5 * max(wnd$max - wnd$min), 0)
    },
    progress_stop = function(p, step, stat, ...) {
      log$e_initial <<- stat$e_initial; log$e_final <<- stat$e; log$steps <<- step
      step >= steps_per_tree * nrow(p$points)
    }
  )

  cell <- data.frame(
    x       = result$points[, "x"],
    y       = result$points[, "y"],
    r_stem  = round(result$marks$dbh / 20, 4),
    h_stem  = round(result$marks$height * 10, 2),
    species = as.character(result$marks$species),
    band    = b,
    segment = s,
    stringsAsFactors = FALSE
  )
  cell$is_original <- cell$x %in% observed$x & cell$y %in% observed$y

  list(
    trees = cell[, c("x", "y", "r_stem", "h_stem", "species", "band", "segment", "is_original")],
    log = data.frame(
      band = b, segment = s, donor_band = donor,
      x_min = wnd$min[1], x_max = wnd$max[1], y_min = wnd$min[2], y_max = wnd$max[2],
      n_reference = nrow(ref), n_observed = nrow(observed),
      density_ha = round(density * 1e4), excluded_share = round(excluded, 3),
      n_target = round(n_target, 1), n_trees = nrow(cell),
      steps = log$steps, e_initial = log$e_initial, e_final = log$e_final)
  )
}

# --- Run all cells -------------------------------------------------------------

jobs <- unlist(lapply(0:3, function(b) lapply(seq_len(length(seg_x) - 1),
  function(s) list(band = b, segment = s))), recursive = FALSE)

cluster <- parallel::makeCluster(min(n_cores, length(jobs)))
parallel::clusterExport(cluster, c("load_engine", "mtppr_path", "quarters", "seg_x", "zone_y",
  "r_vacant", "min_reference", "steps_per_tree", "species_levels",
  "exclusion_sectors", "make_fixed"))
invisible(parallel::clusterEvalQ(cluster, load_engine(mtppr_path)))
results <- parallel::parLapply(cluster, jobs, reconstruct_cell)
parallel::stopCluster(cluster)

# --- Merge and write output ----------------------------------------------------

merged <- do.call(rbind, lapply(results, `[[`, "trees"))
cell_log <- do.call(rbind, lapply(results, `[[`, "log"))
write.csv(merged, out_path, row.names = FALSE)
write.csv(cell_log, sub("\\.csv$", "_log.csv", out_path), row.names = FALSE)
print(cell_log, row.names = FALSE)
cat(sprintf("wrote %d trees (%d observed) to %s\n",
            nrow(merged), sum(merged$is_original), out_path))
