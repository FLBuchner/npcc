#!/usr/bin/env Rscript
# gamCopula's generalized additive conditional copula with the family and the
# formula of the model that generated the data -- the oracle -- as an npcc
# external method (`kind = "external_r"`).
#
#   Rscript fit_gamcopula_oracle.R <train.csv> <eval.csv> <pred.csv> <seed> \
#     <scenario> <family>
#
# The exchange is `npcc.experiments._external`'s with fifteen covariates in
# place of x: the script fits once on `train.csv` (u1, u2, b1..b5, z1..z5,
# t1..t5), predicts at every row of `eval.csv` (grid, u1, u2, and the same
# fifteen), and writes `pred.csv` with one row per evaluation row, in order --
# the row itself, then pdf, hfunc1, hfunc2, tau, fit_time, and the seconds
# each prediction step took. The b's are 0/1, the z's standard normal, the
# t's uniform on [0, 1].
#
# **The formula.** `scenario` names the data-generating model, and the
# calibration function 2 atanh(tau) (tau = TRUE), the scale the true model's
# predictor lives on, gets that model's terms and no others:
#
#   gampcc       ~ b1 + b2 + b3 + z1 + z2 + z3 + s(t1) + s(t2) + s(t3)
#   interaction  ~ s(t1, by = b1) + b1, with b1 a factor
#   null         ~ 1
#
# Each smooth is a penalized cubic regression spline, s(t, k = K, bs = "cr"),
# the basis gamBiCopSelect builds and gampcc's true model is written in. A
# factor `by` gives one smooth of t1 per level of b1, each centered, so the
# factor's main effect carries the difference in level; gamBiCopFit hands
# the formula to mgcv::gam unchanged, and gamBiCopPredict to predict.gam,
# which accept it as it is. This is the model fit_gamcopula_select.R aims at
# when its selection keeps exactly the true covariates -- though its doubling
# step can enlarge a basis beyond K -- except for the interaction, which its
# additive formula cannot express.
#
# **The family** is the true one, as gampcc's oracle had it: `family` is the
# cell's (gaussian or clayton), fitted as gamCopula's Gaussian (1) or its
# double Clayton type I (301), the Clayton where tau > 0 and its 90-degree
# rotation below. The fit runs gampcc's settings, method = "FS" and
# n.iters = 30, with the default tol.rel = 0.001.
#
# The h-functions are exact: VineCopula's, at the family, parameter and second
# parameter the fitted model has at each row. hfunc1 is P(U2 <= u2 | U1 = u1),
# the derivative of the copula in u1, which is npcc's P(V <= v | U = u).

# Basis dimension of every smooth.
K <- 10
# The Fisher-scoring iteration cap, gampcc's.
N_ITERS <- 30

LINEAR <- c(paste0("b", 1:5), paste0("z", 1:5))
SMOOTH <- paste0("t", 1:5)
SCENARIOS <- c("gampcc", "interaction", "null")
# Each family of the study as gamCopula's family code.
FAMILIES <- c(gaussian = 1L, clayton = 301L)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop("usage: Rscript fit_gamcopula_oracle.R ",
       "<train.csv> <eval.csv> <pred.csv> <seed> <scenario> <family>")
}
scenario <- args[[5]]
if (!scenario %in% SCENARIOS) {
  stop("scenario must be one of ", paste(SCENARIOS, collapse = ", "),
       "; got ", scenario)
}
if (!args[[6]] %in% names(FAMILIES)) {
  stop("family must be one of ", paste(names(FAMILIES), collapse = ", "),
       "; got ", args[[6]])
}
family <- FAMILIES[[args[[6]]]]

suppressPackageStartupMessages({
  library(gamCopula)
  library(VineCopula)
})

# The value of `expr` and the wall-clock seconds it took.
timed <- function(expr) {
  start <- Sys.time()
  value <- force(expr)
  list(value = value, seconds = as.numeric(Sys.time() - start, units = "secs"))
}

# Writes every number with 17 significant digits, which reads back to the
# same double; write.csv alone keeps 15.
write_exact <- function(df, path) {
  numeric <- vapply(df, is.numeric, logical(1))
  df[numeric] <- lapply(df[numeric], sprintf, fmt = "%.17g")
  write.csv(df, path, row.names = FALSE, quote = FALSE)
}

# The packages a fit ran on, with the commit of any installed from source.
versions <- function(packages) {
  described <- vapply(packages, function(p) {
    d <- packageDescription(p)
    sha <- ""
    if (!is.null(d$RemoteSha)) {
      sha <- sprintf(" (%s)", substr(d$RemoteSha, 1, 7))
    }
    paste0(p, " ", d$Version, sha)
  }, character(1))
  paste(described, collapse = ", ")
}

# Stops unless `df` has the copula columns and the fifteen covariates, each
# holding what the exchange says it holds.
check_columns <- function(df, what, extra = character(0)) {
  needed <- c(extra, "u1", "u2", LINEAR, SMOOTH)
  missing <- setdiff(needed, names(df))
  if (length(missing) > 0) {
    stop(what, " must have the columns ", paste(needed, collapse = ", "),
         "; got no ", paste(missing, collapse = ", "))
  }
  if (nrow(df) == 0) {
    stop(what, " must have at least one row; got none")
  }
  numbers <- setdiff(needed, extra)
  bad <- numbers[!vapply(df[numbers], function(v) {
    is.numeric(v) && all(is.finite(v))
  }, logical(1))]
  if (length(bad) > 0) {
    stop(what, " must hold finite numbers in ", paste(bad, collapse = ", "))
  }
  inside <- function(columns, ok, values) {
    bad <- columns[!vapply(df[columns], function(v) all(ok(v)), logical(1))]
    if (length(bad) > 0) {
      stop(what, " must hold ", values, " in ", paste(bad, collapse = ", "))
    }
  }
  inside(c("u1", "u2"), function(v) v > 0 & v < 1, "values in (0, 1)")
  inside(paste0("b", 1:5), function(v) v == 0 | v == 1, "0 or 1")
  inside(SMOOTH, function(v) v >= 0 & v <= 1, "values in [0, 1]")
}

# VineCopula's family code at each row. gamCopula's double families switch
# rotation with the sign of the parameter: 301-304 pair a Clayton (3 or 13)
# with a 90- or 270-degree one (23 or 33), and 401-404 do the same for the
# Gumbel. The pairing and the `par > 0` test are gamCopula's own, from
# `getFams` and `bicoppd1d2`.
vine_family <- function(family, par) {
  pairs <- list(
    "301" = c(3, 23), "302" = c(3, 33), "303" = c(13, 23), "304" = c(13, 33),
    "401" = c(4, 24), "402" = c(4, 34), "403" = c(14, 24), "404" = c(14, 34)
  )
  pair <- pairs[[as.character(family)]]
  if (is.null(pair)) {
    return(rep(family, length(par)))
  }
  ifelse(par > 0, pair[1], pair[2])
}

smooth <- function(t, by = NULL) {
  sprintf("s(%s, k = %d, bs = \"cr\"%s)", t, K,
          if (is.null(by)) "" else paste0(", by = ", by))
}
formula <- as.formula(switch(scenario,
  gampcc = paste("~", paste(
    c("b1", "b2", "b3", "z1", "z2", "z3", smooth(c("t1", "t2", "t3"))),
    collapse = " + "
  )),
  interaction = paste("~", smooth("t1", by = "b1"), "+ b1"),
  null = "~ 1"
))

# The covariates as the formula reads them: b1 is a factor where it is a
# `by` variable, with both levels whatever the rows hold.
covariates_of <- function(df) {
  out <- df[, c("u1", "u2", LINEAR, SMOOTH)]
  if (scenario == "interaction") {
    out$b1 <- factor(out$b1, levels = c(0, 1))
  }
  out
}

seed <- suppressWarnings(as.integer(args[[4]]))
if (is.na(seed)) {
  stop("seed must be an integer; got ", args[[4]])
}
set.seed(seed)
train <- read.csv(args[[1]])
points <- read.csv(args[[2]])
check_columns(train, "train.csv")
check_columns(points, "eval.csv", extra = "grid")
if (scenario == "interaction" && length(unique(train$b1)) < 2) {
  stop("train.csv must hold both levels of b1 for s(t1, by = b1); got only ",
       unique(train$b1))
}

fit <- timed(gamBiCopFit(
  covariates_of(train), formula, family, tau = TRUE, method = "FS",
  tol.rel = 0.001, n.iters = N_ITERS
))
model <- fit$value$res
# One line on the fitted model, which the harness logs: the formula, its
# total EDF (mgcv's, which logLik counts) and each smooth's, and whether the
# Fisher scoring converged within n.iters.
terms <- summary(model@model)
smooths <- if (terms$m > 0) {
  paste(sprintf("%s %.2f", rownames(terms$s.table), terms$edf),
        collapse = ", ")
} else {
  "none"
}
cat(sprintf(
  paste0(
    "%s: scenario %s, family %d, %s, EDF %.2f (smooths: %s), par2 %g, ",
    "K %d, n.iters %d, converged %s\n"
  ),
  versions(c("gamCopula", "mgcv", "VineCopula")), scenario,
  model@family, gsub("\\s+", " ", paste(deparse(model@model$formula),
                                        collapse = " ")),
  sum(model@model$edf), smooths, model@par2, K, N_ITERS,
  if (identical(fit$value$conv, 0)) "yes" else "no"
))

# The VineCopula copula at each row of `nd`.
copula_at <- function(nd) {
  par <- gamBiCopPredict(model, newdata = nd, target = "par")$par
  list(
    family = vine_family(model@family, par),
    par = par,
    par2 = rep(model@par2, length(par))
  )
}

hfunc_at <- function(nd, hfunc) {
  cop <- copula_at(nd)
  hfunc(nd$u1, nd$u2, cop$family, cop$par, cop$par2, check.pars = FALSE)
}

quantities <- c("pdf", "hfunc1", "hfunc2", "tau")
values <- matrix(NA_real_, nrow(points), length(quantities),
                 dimnames = list(NULL, quantities))
seconds <- values
for (g in unique(points$grid)) {
  rows <- points$grid == g
  nd <- covariates_of(points[rows, ])
  steps <- list(
    pdf = timed(gamBiCopPDF(model, newdata = nd)),
    hfunc1 = timed(hfunc_at(nd, BiCopHfunc1)),
    hfunc2 = timed(hfunc_at(nd, BiCopHfunc2)),
    tau = timed(gamBiCopPredict(model, newdata = nd, target = "tau")$tau)
  )
  # The h-functions are only exact if they evaluate the copula whose density
  # gamCopula reports, so the family map is checked against that density.
  cop <- copula_at(nd)
  check <- all.equal(
    steps$pdf$value,
    BiCopPDF(nd$u1, nd$u2, cop$family, cop$par, cop$par2, check.pars = FALSE),
    tolerance = 1e-10
  )
  if (!isTRUE(check)) {
    stop("the family map disagrees with gamBiCopPDF on grid ", g, ": ", check)
  }
  for (q in quantities) {
    values[rows, q] <- as.numeric(steps[[q]]$value)
    seconds[rows, q] <- steps[[q]]$seconds
  }
}
bad <- quantities[colSums(!is.finite(values)) > 0]
if (length(bad) > 0) {
  stop("the fitted model must give a finite ", paste(bad, collapse = ", "),
       " at every evaluation point; got non-finite values")
}

colnames(seconds) <- paste0(quantities, "_time")
pred <- cbind(points, values, fit_time = fit$seconds, seconds)
write_exact(pred, args[[3]])
