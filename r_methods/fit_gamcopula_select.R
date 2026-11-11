#!/usr/bin/env Rscript
# gamCopula's covariate and family selection on fifteen covariates -- the
# workflow of the GAM-PCC simulation study (Vatter & Nagler, 2018, JCGS,
# Sec. 4.1) -- as an npcc external method (`kind = "external_r"`).
#
#   Rscript fit_gamcopula_select.R <train.csv> <eval.csv> <pred.csv> <seed> \
#     <scenario> <family>
#
# The exchange is `npcc.experiments._external`'s with fifteen covariates in
# place of x: the script fits once on `train.csv` (u1, u2, b1..b5, z1..z5,
# t1..t5), predicts at every row of `eval.csv` (grid, u1, u2, and the same
# fifteen), and writes `pred.csv` with one row per evaluation row, in order --
# the row itself, then pdf, hfunc1, hfunc2, tau, fit_time, and the seconds
# each prediction step took. The b's are 0/1, the z's standard normal, the
# t's uniform on [0, 1]. `scenario` (gampcc, interaction or constant) and
# `family` (gaussian or clayton) name the data-generating model; this method
# uses neither, and only checks both and logs the scenario.
#
# **The model.** gamBiCopSelect, as gampcc's study calls it: the b's and z's
# are linear terms (`lin.covs`, the b's as numeric 0/1 columns) and the t's
# penalized cubic regression splines s(t, k = K, bs = "cr") (`smooth.covs`),
# all in the calibration function 2 atanh(tau) (tau = TRUE). The covariates
# are selected by gamBiCopVarSel: fit every term, drop each linear term and
# each smooth whose p-value is at least level = 0.05, make each remaining
# smooth with an EDF below edf = 1.5 a linear term, and refit until nothing
# changes; then double the basis of a smooth whose EDF exceeds 0.8 (k - 1),
# for as long as every basis is below n / 30. With select.once = FALSE that
# selection is run for each family, and the family is the one whose selected
# model has the lowest AIC, 2 EDF - 2 loglik, with one more degree of freedom
# for the Student t.
#
# **Settings.** gampcc's Rmd reaches gamBiCopSelect through gamVineCopSelect
# and fitAGAMCopula with familyset = c(1, 2, 301, 302, 401, 402),
# rotations = FALSE, familycrit = "AIC", level = 0.05, edf = 1.5, tau = TRUE,
# method = "FS", tol.rel = 0.001, n.iters = 30 and select.once = FALSE. The
# family set here is instead E1b's (fit_gamcopula.R): familyset = NA with
# rotations = TRUE, so Gaussian, Student t, and the double Clayton and Gumbel
# families with every rotation (1, 2, 301-304, 401-404). select.once = FALSE
# is gampcc's; fit_gamcopula.R ran the defaults, n.iters = 10 and TRUE.
# Everything else is the default in both. The vine's fallback to a
# constant BiCopSelect when no covariate survives is not gamBiCopSelect's,
# and a single pair does not take it: the intercept-only fit stands.
#
# **Iterations.** n.iters = 100, where gampcc ran 30: its oracle's Fisher
# scoring can stop at 30 unconverged on gampcc's 30-odd coefficients, with
# gamCopula's warning that the result is not to be trusted, and a higher cap
# leaves every fit that converges sooner as it is. Both gamCopula arms run the
# same cap, so their comparison isolates the selection. Each logs whether its
# final fit converged.
#
# **Basis size.** gamBiCopVarSel hard-codes k = 10 for every smooth, at the
# two sites that build its basis vector. K below replaces that literal, in a
# copy of gamBiCopVarSel that a copy of gamBiCopSelect calls; with K = 10 the
# copy's body is identical to the package's. The script stops if the package
# no longer has exactly those two sites.
#
# The h-functions are exact: VineCopula's, at the family, parameter and second
# parameter the fitted model has at each row. hfunc1 is P(U2 <= u2 | U1 = u1),
# the derivative of the copula in u1, which is npcc's P(V <= v | U = u).

# Basis dimension of every smooth before gamBiCopVarSel's doubling step.
K <- 10
# The Fisher-scoring iteration cap (see **Iterations**).
N_ITERS <- 100

LINEAR <- c(paste0("b", 1:5), paste0("z", 1:5))
SMOOTH <- paste0("t", 1:5)
SCENARIOS <- c("gampcc", "interaction", "constant")
FAMILIES <- c("gaussian", "clayton")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop("usage: Rscript fit_gamcopula_select.R ",
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

# gamBiCopSelect with every smooth's basis dimension `k` where gamBiCopVarSel
# hard-codes 10: a copy of gamBiCopVarSel whose `rep(10, ...)` calls read
# `rep(k, ...)`, and a copy of gamBiCopSelect that finds it first. Both copies
# keep the package namespace as their scope.
select_with_basis <- function(k) {
  sites <- 0
  swap <- function(expr) {
    if (identical(expr[[1]], as.name("rep")) && identical(expr[[2]], 10)) {
      expr[[2]] <- k
      sites <<- sites + 1
    }
    for (i in seq_along(expr)) {
      if (is.call(expr[[i]])) {
        expr[[i]] <- swap(expr[[i]])
      }
    }
    expr
  }
  var_sel <- gamCopula:::gamBiCopVarSel
  body(var_sel) <- swap(body(var_sel))
  if (sites != 2) {
    stop("gamBiCopVarSel must set its basis size at two rep(10, ...) sites ",
         "for K to replace; got ", sites, ", so this gamCopula differs from ",
         "the 0.0-8 this script was written against")
  }
  scope <- new.env(parent = asNamespace("gamCopula"))
  assign("gamBiCopVarSel", var_sel, envir = scope)
  select <- gamBiCopSelect
  environment(select) <- scope
  select
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

fit <- timed(select_with_basis(K)(
  udata = train[, c("u1", "u2")],
  lin.covs = train[, LINEAR],
  smooth.covs = train[, SMOOTH],
  familyset = NA,
  rotations = TRUE,
  familycrit = "AIC",
  n.iters = N_ITERS,
  select.once = FALSE
))
# On failure gamBiCopSelect returns its message rather than raising.
if (is.character(fit$value)) {
  stop("gamBiCopSelect: ", fit$value)
}
model <- fit$value$res
# One line on the fitted model, which the harness logs: the selected formula,
# its total EDF (mgcv's, which logLik counts) and each smooth's, and whether
# the selected family's final fit converged within n.iters.
terms <- summary(model@model)
smooths <- if (terms$m > 0) {
  paste(sprintf("%s %.2f", rownames(terms$s.table), terms$edf),
        collapse = ", ")
} else {
  "none"
}
selected <- all.vars(model@model$pred.formula)
cat(sprintf(
  paste0(
    "%s: scenario %s, family %d, %s, covariates selected %s, EDF %.2f ",
    "(smooths: %s), par2 %g, K %d, n.iters %d, converged %s\n"
  ),
  versions(c("gamCopula", "mgcv", "VineCopula")), scenario,
  model@family, gsub("\\s+", " ", paste(deparse(model@model$formula),
                                        collapse = " ")),
  if (length(selected) > 0) paste(selected, collapse = " ") else "none",
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
  nd <- points[rows, c("u1", "u2", LINEAR, SMOOTH)]
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
