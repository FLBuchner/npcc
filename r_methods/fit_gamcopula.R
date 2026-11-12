#!/usr/bin/env Rscript
# gamCopula's generalized additive conditional copula, as an npcc external
# method (`kind = "external_r"`).
#
#   Rscript fit_gamcopula.R <train.csv> <eval.csv> <pred.csv> <seed>
#
# The exchange is documented in `npcc.experiments._external`: the script fits
# once on `train.csv` (u1, u2, x), predicts at every row of `eval.csv`
# (grid, u1, u2, x), and writes `pred.csv` with one row per evaluation row, in
# order -- the row itself, then pdf, hfunc1, hfunc2, tau, and the seconds each
# step took.
#
# The family is selected by AIC from gamCopula's default set (Gaussian,
# Student t, and the double Clayton and Gumbel families with every rotation),
# with x entering the calibration function as a penalized spline.
#
# The h-functions are exact: VineCopula's, at the family, parameter and second
# parameter the fitted model has at each row. hfunc1 is P(U2 <= u2 | U1 = u1),
# the derivative of the copula in u1, which is npcc's P(V <= v | U = u).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("usage: Rscript fit_gamcopula.R ",
       "<train.csv> <eval.csv> <pred.csv> <seed>")
}

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

set.seed(as.integer(args[[4]]))
train <- read.csv(args[[1]])
points <- read.csv(args[[2]])

fit <- timed(gamBiCopSelect(
  udata = train[, c("u1", "u2")],
  smooth.covs = data.frame(x = train$x)
))
# On failure gamBiCopSelect returns its message rather than raising.
if (is.character(fit$value)) {
  stop("gamBiCopSelect: ", fit$value)
}
model <- fit$value$res
# One line on the fitted model, which the harness logs.
cat(sprintf(
  "%s: family %d, %s, EDF %.2f, par2 %g\n",
  versions(c("gamCopula", "mgcv", "VineCopula")),
  model@family, paste(deparse(model@model$formula), collapse = " "),
  sum(EDF(model)), model@par2
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

pred <- points
for (g in unique(points$grid)) {
  rows <- points$grid == g
  nd <- points[rows, c("u1", "u2", "x")]
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
  for (q in names(steps)) {
    pred[rows, q] <- as.numeric(steps[[q]]$value)
    pred[rows, paste0(q, "_time")] <- steps[[q]]$seconds
  }
}
pred$fit_time <- fit$seconds

write_exact(pred, args[[3]])
