#!/usr/bin/env Rscript
# boostCopula's boosted conditional copula on a cubic B-spline basis of x,
# stopped by the package's 10-fold cross-validation, as an npcc external method
# (`kind = "external_r"`).
#
#   Rscript fit_boostcopula_smooth_cv.R <train.csv> <eval.csv> <pred.csv> <seed>
#
# The exchange is documented in `npcc.experiments._external`: the script fits
# once on `train.csv` (u1, u2, x), predicts at every row of `eval.csv`
# (grid, u1, u2, x), and writes `pred.csv` with one row per evaluation row, in
# order -- the row itself, then pdf, hfunc1, hfunc2, tau, and the seconds each
# step took.
#
# The family is selected by AIC from boostBiCopSelect's default set (Gaussian,
# and the double Clayton and Gumbel families with every rotation), each family
# evaluated at its own cross-validated stop. Kendall's tau is tanh of a linear
# predictor in a cubic B-spline basis of x, fitted by componentwise gradient
# boosting (step length nu = 0.1, the default).
#
# **Stopping.** Early stopping is the method's only regularization, and the
# AIC the package also offers stops penalizing once every basis coefficient is
# in, so the stop is chosen by 10-fold cross-validation instead (mstop = "cv",
# nfolds = 10): the iteration that maximizes the out-of-fold log-likelihood.
# maxit = 5000 is set high so that the cross-validation, not the cap, decides;
# the logged line says when the stop lands within 5% of it. The folds are drawn
# with `sample`, from the seed the harness passes; `cores` and `cores.control`
# stay at 1, so the draws happen in one fixed order and the fit is
# reproducible.
#
# boostBiCopSelect has no smooth base-learner -- its formula selects columns
# of X, and each boosting step updates one column's coefficient -- so the
# smooth effect enters as columns. The basis is splines::bs(x, df = 9): degree
# 3, interior knots at the quantiles of the training x, and its boundary knots
# at the range of the training x, which every evaluation point lies within.
# Its first B-spline is dropped (intercept = FALSE) because the model keeps its
# own intercept, so the predictor spans the full cubic spline space with the
# intercept and nine basis coefficients -- as many coefficients as gamCopula's
# s(x, k = 10) smooth has besides its intercept.
#
# The h-functions are exact: VineCopula's, at the family, parameter and second
# parameter the fitted model has at each row. hfunc1 is P(U2 <= u2 | U1 = u1),
# the derivative of the copula in u1, which is npcc's P(V <= v | U = u).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("usage: Rscript fit_boostcopula.R ",
       "<train.csv> <eval.csv> <pred.csv> <seed>")
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

set.seed(as.integer(args[[4]]))
train <- read.csv(args[[1]])
points <- read.csv(args[[2]])

basis <- bs(train$x, df = 9)

# The basis at `x`, as the covariate columns b1, ..., b9; `predict` keeps the
# knots of the training basis.
basis_at <- function(x) {
  columns <- as.data.frame(unclass(predict(basis, x)))
  colnames(columns) <- paste0("b", seq_len(ncol(columns)))
  columns
}

fit <- timed(boostBiCopSelect(
  formula = ~.,
  U = train[, c("u1", "u2")],
  X = basis_at(train$x),
  familyset = NA,
  control = boost_control(maxit = 5000, mstop = "cv", nfolds = 10)
))
model <- fit$value
maxit <- model$control$maxit
# One line on the fitted model, which the harness logs.
cat(sprintf(
  paste0(
    "%s: family %s (%s), cv-stop %d/%d%s, %d nonzero coefficients, par2 %g, ",
    "bs(x, df = 9) knots %s, boundary %s\n"
  ),
  versions(c("boostCopula", "VineCopula")),
  model$family, model$familyname, model$iterations, maxit,
  if (model$iterations >= 0.95 * maxit) " (within 5% of maxit)" else "",
  sum(model$coefficients != 0), model$par2,
  paste(signif(attr(basis, "knots"), 4), collapse = " "),
  paste(attr(basis, "Boundary.knots"), collapse = " ")
))

# The VineCopula copula at each row of `nd`.
copula_at <- function(nd) {
  covariates <- basis_at(nd$x)
  par <- predict(model, newdata = covariates, type = "parameter")$parameter
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
  predict(model, newdata = basis_at(nd$x), type = "tau")$tau
}

pred <- points
for (g in unique(points$grid)) {
  rows <- points$grid == g
  nd <- points[rows, c("u1", "u2", "x")]
  steps <- list(
    pdf = timed(
      boostBiCopPDF(model, nd[, c("u1", "u2")], basis_at(nd$x))
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
  for (q in names(steps)) {
    pred[rows, q] <- as.numeric(steps[[q]]$value)
    pred[rows, paste0(q, "_time")] <- steps[[q]]$seconds
  }
}
pred$fit_time <- fit$seconds

write_exact(pred, args[[3]])
