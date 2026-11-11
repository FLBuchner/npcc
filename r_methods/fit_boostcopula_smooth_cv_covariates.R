#!/usr/bin/env Rscript
# boostCopula's boosted conditional copula on fifteen covariates -- the b's
# and z's as linear columns, a cubic B-spline basis of each t -- stopped by
# the package's 10-fold cross-validation, as an npcc external method
# (`kind = "external_r"`).
#
#   Rscript fit_boostcopula_smooth_cv_covariates.R <train.csv> <eval.csv> \
#     <pred.csv> <seed> <scenario> <family>
#
# The exchange is `npcc.experiments._external`'s with fifteen covariates in
# place of x: the script fits once on `train.csv` (u1, u2, b1..b5, z1..z5,
# t1..t5), predicts at every row of `eval.csv` (grid, u1, u2, and the same
# fifteen), and writes `pred.csv` with one row per evaluation row, in order --
# the row itself, then pdf, hfunc1, hfunc2, tau, fit_time, and the seconds
# each prediction step took. The b's are 0/1, the z's standard normal, the
# t's uniform on [0, 1]. `scenario` (gampcc, interaction or null) and
# `family` (gaussian or clayton) name the data-generating model; this method
# uses neither, and only checks both and logs the scenario.
#
# The family is selected by AIC from boostBiCopSelect's default set (Gaussian,
# and the double Clayton and Gumbel families with every rotation), each family
# evaluated at its own cross-validated stop. Kendall's tau is tanh of a linear
# predictor in the columns below, fitted by componentwise gradient boosting
# (step length nu = 0.1, the default): each step updates the one column that
# best fits the negative gradient, so the stop also selects the columns, and
# a covariate none of whose columns was ever chosen is out of the model.
#
# **Columns.** b1..b5 (as 0/1) and z1..z5 enter as they are, one column and
# one linear effect each. boostBiCopSelect has no smooth base-learner, so each
# t enters as the columns of splines::bs(t, df = DF): degree 3, interior knots
# at the quantiles of the training t, and its first B-spline dropped
# (intercept = FALSE) because the model keeps its own intercept. The boundary
# knots are the ends of the t's support, [0, 1], rather than the training
# range, which a fresh evaluation point can fall outside of; `predict` keeps
# both kinds of knot. With DF = 9 each t has as many coefficients as
# gamCopula's s(t, k = 10) smooth has besides the intercept. A column of t_k is
# named t<k>_bs<j>. boostBiCopSelect centers every column (center = TRUE, the
# default) and folds the means into the intercept, so predictions take the
# raw columns.
#
# **Stopping.** Early stopping is the method's only regularization, and the
# AIC the package also offers stops penalizing once every column is in, so
# the stop is chosen by 10-fold cross-validation instead (mstop = "cv",
# nfolds = 10): the iteration that maximizes the out-of-fold log-likelihood.
# maxit = 5000 is set high so that the cross-validation, not the cap, decides;
# the logged line says when the stop lands within 5% of it. The folds are drawn
# with `sample`, from the seed the harness passes; `cores` and `cores.control`
# stay at 1, so the draws happen in one fixed order and the fit is
# reproducible.
#
# The h-functions are exact: VineCopula's, at the family, parameter and second
# parameter the fitted model has at each row. hfunc1 is P(U2 <= u2 | U1 = u1),
# the derivative of the copula in u1, which is npcc's P(V <= v | U = u).

# Degrees of freedom of each t's B-spline basis.
DF <- 9
# The boosting iteration cap.
MAXIT <- 5000

LINEAR <- c(paste0("b", 1:5), paste0("z", 1:5))
SMOOTH <- paste0("t", 1:5)
SCENARIOS <- c("gampcc", "interaction", "null")
FAMILIES <- c("gaussian", "clayton")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop("usage: Rscript fit_boostcopula_smooth_cv_covariates.R ",
       "<train.csv> <eval.csv> <pred.csv> <seed> <scenario> <family>")
}
scenario <- args[[5]]
if (!scenario %in% SCENARIOS) {
  stop("scenario must be one of ", paste(SCENARIOS, collapse = ", "),
       "; got ", scenario)
}
if (!args[[6]] %in% FAMILIES) {
  stop("family must be one of ", paste(FAMILIES, collapse = ", "),
       "; got ", args[[6]])
}

suppressPackageStartupMessages({
  library(boostCopula)
  library(VineCopula)
  library(splines)
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

# VineCopula's family code at each row. The double families switch rotation
# with the sign of the parameter: 301-304 pair a Clayton (3 or 13) with a 90-
# or 270-degree one (23 or 33), and 401-404 do the same for the Gumbel. The
# pairing and the `par < 0` test are boostCopula's own, from `getFams` and
# `boostBiCopPDF`.
vine_family <- function(family, par) {
  pairs <- list(
    "301" = c(3, 23), "302" = c(3, 33), "303" = c(13, 23), "304" = c(13, 33),
    "401" = c(4, 24), "402" = c(4, 34), "403" = c(14, 24), "404" = c(14, 34)
  )
  pair <- pairs[[as.character(family)]]
  if (is.null(pair)) {
    return(rep(family, length(par)))
  }
  ifelse(par < 0, pair[2], pair[1])
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

bases <- lapply(SMOOTH, function(t) {
  bs(train[[t]], df = DF, Boundary.knots = c(0, 1))
})
names(bases) <- SMOOTH

# The model's columns at the rows of `df`: the b's and z's, then each t's
# basis; `predict` keeps the knots of the training basis.
design_at <- function(df) {
  splines <- lapply(SMOOTH, function(t) {
    columns <- as.data.frame(unclass(predict(bases[[t]], df[[t]])))
    colnames(columns) <- paste0(t, "_bs", seq_len(ncol(columns)))
    columns
  })
  do.call(cbind, c(list(df[LINEAR]), splines))
}

fit <- timed(boostBiCopSelect(
  formula = ~.,
  U = train[, c("u1", "u2")],
  X = design_at(train),
  familyset = NA,
  control = boost_control(maxit = MAXIT, mstop = "cv", nfolds = 10)
))
model <- fit$value
maxit <- model$control$maxit
# Which covariates the boosting chose: each b or z with a nonzero coefficient,
# and each t with how many of its DF basis coefficients are nonzero.
nonzero <- model$coefficients != 0
chosen <- c(
  LINEAR[nonzero[LINEAR]],
  unlist(lapply(SMOOTH, function(t) {
    k <- sum(nonzero[startsWith(names(nonzero), paste0(t, "_bs"))])
    if (k > 0) sprintf("%s (%d/%d)", t, k, DF)
  }))
)
# One line on the fitted model, which the harness logs.
cat(sprintf(
  paste0(
    "%s: scenario %s, family %s (%s), cv-stop %d/%d%s, %d nonzero ",
    "coefficients of %d, covariates chosen %s, par2 %g, bs(t, df = %d) on ",
    "[0, 1]\n"
  ),
  versions(c("boostCopula", "VineCopula")), scenario,
  model$family, model$familyname, model$iterations, maxit,
  if (model$iterations >= 0.95 * maxit) " (within 5% of maxit)" else "",
  sum(nonzero), length(nonzero),
  if (length(chosen) > 0) paste(chosen, collapse = " ") else "none",
  model$par2, DF
))

# The VineCopula copula at each row of `nd`.
copula_at <- function(nd) {
  par <- predict(model, newdata = design_at(nd), type = "parameter")$parameter
  list(
    family = vine_family(as.numeric(model$family), par),
    par = par,
    par2 = rep(model$par2, length(par))
  )
}

hfunc_at <- function(nd, hfunc) {
  cop <- copula_at(nd)
  hfunc(nd$u1, nd$u2, cop$family, cop$par, cop$par2, check.pars = FALSE)
}

tau_at <- function(nd) {
  predict(model, newdata = design_at(nd), type = "tau")$tau
}

quantities <- c("pdf", "hfunc1", "hfunc2", "tau")
values <- matrix(NA_real_, nrow(points), length(quantities),
                 dimnames = list(NULL, quantities))
seconds <- values
for (g in unique(points$grid)) {
  rows <- points$grid == g
  nd <- points[rows, c("u1", "u2", LINEAR, SMOOTH)]
  steps <- list(
    pdf = timed(
      boostBiCopPDF(model, nd[, c("u1", "u2")], design_at(nd))
    ),
    hfunc1 = timed(hfunc_at(nd, BiCopHfunc1)),
    hfunc2 = timed(hfunc_at(nd, BiCopHfunc2)),
    tau = timed(tau_at(nd))
  )
  # The h-functions are only exact if they evaluate the copula whose density
  # boostCopula reports, so the family map is checked against that density.
  cop <- copula_at(nd)
  check <- all.equal(
    as.numeric(steps$pdf$value),
    BiCopPDF(nd$u1, nd$u2, cop$family, cop$par, cop$par2, check.pars = FALSE),
    tolerance = 1e-10
  )
  if (!isTRUE(check)) {
    stop("the family map disagrees with boostBiCopPDF on grid ", g, ": ",
         check)
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
