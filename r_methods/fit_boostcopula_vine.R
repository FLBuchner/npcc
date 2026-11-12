#!/usr/bin/env Rscript
# boostCopula's boosted conditional R-vine, the weather application's
# `boostcopula-all` and `boostcopula-time` arms, as run by
# `npcc.experiments.application.copulas`:
#
#   Rscript fit_boostcopula_vine.R <dir> <seed> <n_draws>
#   Rscript fit_boostcopula_vine.R --check
#
# **Exchange.** The script reads `train.csv` and `test.csv` from <dir>, each
# with the columns u1..ud, copula-scale observations inside (0, 1), and
# x1..xp, the covariates, one row per case. It writes into <dir>:
# - `density.csv`, one row per test row in order: u1..ud, echoed, and `pdf`,
#   the fitted copula density there, NA where it is not finite;
# - `draws.csv`, n_draws rows per test row, test row by test row: `case`, the
#   test row's index from 1, and u1..ud, a draw from the fitted copula at that
#   row's covariates;
# - `fitted.csv`, the selected structure's R-vine matrix in VineCopula's
#   convention, and `families.csv`, boostCopula's family code of each pair in
#   the same layout with 0 off the pairs, both without a header;
# - `info.csv`, `key` and `value`: the versions the fit ran on, and the
#   seconds its steps took (`fit_time`, `pdf_time`, `sim_time`).
# Every number is written with 17 significant digits, which reads back to the
# same double.
#
# `--check` loads the packages and prints their versions, without fitting.
#
# **Model.** boostRVineStructureSelect at the package's defaults: Dissmann's
# algorithm with Kendall's tau as the tree criterion, each pair's family
# selected by AIC from the default set, its parameter a boosted function of
# every covariate. The initial matrix below supplies the formulas only, one
# `~ .` per pair; the structure and families are selected. It is 5 x 5, so the
# script fits five variables and refuses any other number.
#
# **Draws.** The seed is set once, before the fit. After the density, each
# test row in turn takes n_draws x d uniforms, `matrix(runif(n_draws * d),
# ncol = d)`, which boostRVineSim maps through the inverse Rosenblatt
# transform at that row's covariates. boostRVineSim draws no random numbers
# and transforms row by row, so one call on every test row's uniforms, stacked
# in that order, gives the draws of one call per row.
#
# Exit status 3 refuses the input or reports a missing package: a defect of
# the harness or the machine, which stops the run. Any other nonzero status is
# the method failing on its sample.

# The initial matrix, whose formula rows the selection reads.
INITIAL <- matrix(c(5, 0, 0, 0, 0,
                    2, 2, 0, 0, 0,
                    3, 3, 3, 0, 0,
                    1, 4, 4, 4, 0,
                    4, 1, 1, 1, 1), 5, 5, byrow = TRUE)
PACKAGES <- c("boostCopula", "VineCopula", "Rfast")

refuse <- function(...) {
  message("Error: ", ...)
  quit(save = "no", status = 3)
}

# The packages a fit runs on, with the commit of any installed from source.
versions <- function(packages) {
  described <- vapply(packages, function(p) {
    d <- packageDescription(p)
    sha <- ""
    if (!is.null(d$RemoteSha)) {
      sha <- sprintf(" (%s)", d$RemoteSha)
    }
    paste0(p, " ", d$Version, sha)
  }, character(1))
  paste(described, collapse = ", ")
}

args <- commandArgs(trailingOnly = TRUE)
absent <- PACKAGES[!vapply(PACKAGES, requireNamespace, logical(1),
                            quietly = TRUE)]
if (length(absent) > 0) {
  refuse("the R packages ", paste(absent, collapse = ", "), " must be ",
         "installed; R's library paths are ", paste(.libPaths(), collapse = ", "))
}
if (identical(args, "--check")) {
  cat(R.version.string, ", ", versions(PACKAGES), "\n", sep = "")
  quit(save = "no", status = 0)
}
if (length(args) != 3) {
  refuse("usage: Rscript fit_boostcopula_vine.R <dir> <seed> <n_draws> ",
         "or --check")
}
dir <- args[[1]]
# as.integer() alone truncates "1.5" to 1; the digits test refuses it.
integer_arg <- function(text) {
  if (grepl("^-?[0-9]+$", text)) suppressWarnings(as.integer(text)) else NA
}
seed <- integer_arg(args[[2]])
if (is.na(seed)) {
  refuse("seed must be an integer within R's range; got ", args[[2]])
}
n_draws <- integer_arg(args[[3]])
if (is.na(n_draws) || n_draws < 1) {
  refuse("n_draws must be a positive integer; got ", args[[3]])
}

suppressPackageStartupMessages({
  library(boostCopula)
  library(VineCopula)
})

# The value of `expr` and the wall-clock seconds it took.
timed <- function(expr) {
  start <- Sys.time()
  value <- force(expr)
  list(value = value, seconds = as.numeric(Sys.time() - start, units = "secs"))
}

exact <- function(v) ifelse(is.finite(v), sprintf("%.17g", v), "NA")

write_exact <- function(df, path) {
  numeric <- vapply(df, is.numeric, logical(1))
  df[numeric] <- lapply(df[numeric], exact)
  write.csv(df, path, row.names = FALSE, quote = FALSE)
}

# Reads `name`, refusing anything but finite numbers in its u and x columns,
# with every u inside (0, 1).
read_cases <- function(name) {
  df <- read.csv(file.path(dir, name))
  u <- grep("^u[0-9]+$", names(df), value = TRUE)
  x <- grep("^x[0-9]+$", names(df), value = TRUE)
  if (length(u) == 0 || length(x) == 0 || nrow(df) == 0) {
    refuse(name, " must have rows and the columns u1.. and x1..; got ",
           nrow(df), " rows and the columns ", paste(names(df), collapse = ", "))
  }
  finite <- vapply(df[c(u, x)], function(v) {
    is.numeric(v) && all(is.finite(v))
  }, logical(1))
  if (!all(finite)) {
    refuse(name, " must hold finite numbers in ",
           paste(c(u, x)[!finite], collapse = ", "))
  }
  inside <- vapply(df[u], function(v) all(v > 0 & v < 1), logical(1))
  if (!all(inside)) {
    refuse(name, " must hold values inside (0, 1) in ",
           paste(u[!inside], collapse = ", "))
  }
  list(u = u, x = x, df = df)
}

set.seed(seed)
train <- read_cases("train.csv")
test <- read_cases("test.csv")
if (!identical(train$u, test$u) || !identical(train$x, test$x)) {
  refuse("train.csv and test.csv must have the same columns; got ",
         paste(c(train$u, train$x), collapse = ", "), " and ",
         paste(c(test$u, test$x), collapse = ", "))
}
d <- length(train$u)
if (d != nrow(INITIAL)) {
  refuse("the data must have ", nrow(INITIAL), " variables, the initial ",
         "matrix's; got ", d)
}
U <- unname(as.matrix(train$df[train$u]))
X <- train$df[train$x]
U_test <- unname(as.matrix(test$df[test$u]))
X_test <- test$df[test$x]

spec <- boostRVineMatrix(INITIAL, formula = matrix("~ .", d, d))
fit <- timed(boostRVineStructureSelect(U = U, X = X, boostRVM = spec))
model <- fit$value
pdf <- timed(as.numeric(boostRVinePDF(model, U = U_test, X = X_test)))
if (length(pdf$value) != nrow(U_test)) {
  stop("boostRVinePDF must return one density per test row; got ",
       length(pdf$value), " for ", nrow(U_test))
}
sim <- timed({
  m <- nrow(U_test)
  base <- do.call(rbind, lapply(seq_len(m), function(i) {
    matrix(runif(n_draws * d), ncol = d)
  }))
  context <- X_test[rep(seq_len(m), each = n_draws), , drop = FALSE]
  boostRVineSim(model, U = base, X = context)
})

density <- data.frame(test$df[test$u], pdf = pdf$value)
draws <- data.frame(case = rep(seq_len(nrow(U_test)), each = n_draws),
                    sim$value)
names(draws) <- c("case", test$u)
write_exact(density, file.path(dir, "density.csv"))
write_exact(draws, file.path(dir, "draws.csv"))
families <- model$boostRVM$family
families[is.na(families)] <- 0
write.table(model$boostRVM$Matrix, file.path(dir, "fitted.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
write.table(families, file.path(dir, "families.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
info <- data.frame(
  key = c("r_version", "packages", "fit_time", "pdf_time", "sim_time"),
  value = c(R.version.string, versions(PACKAGES), exact(fit$seconds),
            exact(pdf$seconds), exact(sim$seconds))
)
write.csv(info, file.path(dir, "info.csv"), row.names = FALSE)
# One line on the fitted model, which the harness logs.
cat(sprintf("%s: d %d, p %d, n %d, %d test rows x %d draws; fit %.1fs\n",
            versions("boostCopula"), d, length(train$x), nrow(U),
            nrow(U_test), n_draws, fit$seconds))
