#!/usr/bin/env Rscript
# GAM-based conditional vines (gamCopula) on Gaussian GAM margins, E2b's
# competitors in the vine study, as run by `npcc.experiments._external_vine`:
#
#   Rscript fit_gamvine.R <dir> <seed> <arm>
#
# The exchange is documented in `npcc.experiments._external_vine`: the script
# reads `train.csv` (x, y1..yd) and `eval.csv` (x, y1..yd, u1..ud) from <dir>,
# and for the oracle `structure.csv`, the true R-vine matrix in VineCopula's
# convention; it writes `pred.csv`, one row per evaluation row in order -- the
# row itself, then pdf, copula_pdf and the seconds each step took --
# `fitted.csv`, the fitted vine's R-vine matrix, and `pairs.csv`, one row per
# pair copula with its family and calibration function.
#
# **Margins.** Each variable is mgcv's gam(y_j ~ s(x)), Gaussian, at the
# package defaults. Its probability integral transform is pnorm of the
# standardized residual, (y_j - mu_j(x)) / sigma_j with sigma_j^2 the fit's
# scale estimate sig2; the vine is fitted on the training sample's. At a test
# point, `pdf` is the fitted copula density at the point's transforms times
# the normal margin densities, and `copula_pdf` the fitted copula density at
# the point's true transforms u1..ud.
#
# **Arms.** x enters every pair's calibration function as a smooth term:
# - gamvine-oracle: gamVineCopSelect on the true structure, simplified;
# - gamvine-selected: gamVineStructureSelect, simplified, treecrit = "tau";
# - gamvine-ns: gamVineStructureSelect, not simplified, so a pair beyond the
#   first tree also takes its conditioning variables as smooth terms (Vatter
#   and Nagler, 2018).
#
# **Settings.** E1b's (fit_gamcopula.R): gamCopula's default family set --
# Gaussian, Student t, and the double Clayton and Gumbel families with every
# rotation (1, 2, 301-304, 401-404) -- selected by familycrit = "AIC", with
# level = 0.05 for the covariates' and the independence tests. n.iters = 100
# is E9's (fit_gamcopula_select.R, **Iterations**): a cap that leaves every
# fit converging sooner as it is. Everything else is gamCopula's default:
# tau = TRUE, method = "FS", tol.rel = 0.001, select.once = TRUE, no
# truncation. A pair whose selection keeps no covariate, or whose GAM fit
# fails, is gamCopula's constant BiCopSelect fit, or independence when its
# test does not reject.
#
# **Names.** The copula columns are v1..vd. gamCopula names a pair's own
# arguments u1 and u2 in the data frame it fits on, beside the conditioning
# variables of a non-simplified pair; a conditioning variable named u1 or u2
# would be read as the pair's argument instead.
#
# **Rotations.** gamCopula 0.0.8 stores a vine pair's double family with its
# 90- and 270-degree members exchanged, the family of the pair with its two
# arguments swapped, whenever its `rotation` flag is set. gamVineCopSelect sets
# it to its `rotations` argument, although it fits each pair in the order the
# vine evaluates it, so a negatively dependent pair would be evaluated as the
# mirror of its fit; `rotations = FALSE` leaves the family as fitted, and with
# the default family set, which holds every rotation, selects from the same
# families. gamVineStructureSelect always sets it, and needs the swap: its
# matrix assembly (as.GVC) swaps back each pair the matrix orients as fitted.
# But the pair's h-functions, which the next tree is fitted on, are then those
# of the swapped family. `structure_select` below runs gamVineStructureSelect
# with each pair fitted unswapped -- h-functions included -- and swapped only
# after, which leaves every stored family as the package stores it and every
# next tree fitted on the fitted pair's h-functions.
#
# **Check.** gamVinePDF is checked, on the first CHECK_ROWS rows of each
# evaluation, against the product of the fitted pairs' gamBiCopPDF densities,
# each at the h-functions of the trees below it, which `pair_product` walks
# edge by edge from the R-vine matrix in VineCopula's convention.

# The Fisher-scoring iteration cap (see **Settings**).
N_ITERS <- 100
# The rows of each evaluation on which gamVinePDF is checked.
CHECK_ROWS <- 200
# Where a margin's transform is clamped into the open unit interval: an exact
# 0 or 1 takes a standardized residual beyond about 8.3.
PIT_EPS <- 1e-12
ARMS <- c("gamvine-oracle", "gamvine-selected", "gamvine-ns")

# Stops with status 3, which the harness reads as a refused input or a failed
# check of this script's own -- its defect or the harness's, which stops the
# run -- rather than the method failing on its sample, which stop() reports.
refuse <- function(...) {
  message("Error: ", ...)
  quit(save = "no", status = 3)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  refuse("usage: Rscript fit_gamvine.R <dir> <seed> <arm>")
}
dir <- args[[1]]
arm <- args[[3]]
if (!arm %in% ARMS) {
  refuse("arm must be one of ", paste(ARMS, collapse = ", "), "; got ", arm)
}
# as.integer() alone truncates "1.5" to 1; the digits test refuses it.
seed <- if (grepl("^-?[0-9]+$", args[[2]])) {
  suppressWarnings(as.integer(args[[2]]))
} else {
  NA_integer_
}
if (is.na(seed)) {
  refuse("seed must be an integer within R's range; got ", args[[2]])
}

suppressPackageStartupMessages({
  library(gamCopula)
  library(mgcv)
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
  write.csv(df, path, row.names = FALSE, quote = TRUE)
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

# Stops unless `df` has the columns `needed`, each holding finite numbers.
check_columns <- function(df, what, needed) {
  missing <- setdiff(needed, names(df))
  if (length(missing) > 0) {
    refuse(what, " must have the columns ", paste(needed, collapse = ", "),
           "; got no ", paste(missing, collapse = ", "))
  }
  if (nrow(df) == 0) {
    refuse(what, " must have at least one row; got none")
  }
  bad <- needed[!vapply(df[needed], function(v) {
    is.numeric(v) && all(is.finite(v))
  }, logical(1))]
  if (length(bad) > 0) {
    refuse(what, " must hold finite numbers in ", paste(bad, collapse = ", "))
  }
}

# VineCopula's family code at each row. gamCopula's double families switch
# rotation with the sign of the parameter: 301-304 pair a Clayton (3 or 13)
# with a 90- or 270-degree one (23 or 33), and 401-404 do the same for the
# Gumbel. The pairing and the `par > 0` test are gamCopula's own, from
# `getFams` and `famTrans`.
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

# A gamCopula family code's name, for the pair table.
FAMILY_NAMES <- c(
  "0" = "independence", "1" = "gaussian", "2" = "student",
  "301" = "clayton 0/90", "302" = "clayton 0/270",
  "303" = "clayton 180/90", "304" = "clayton 180/270",
  "401" = "gumbel 0/90", "402" = "gumbel 0/270",
  "403" = "gumbel 180/90", "404" = "gumbel 180/270"
)

# The double family of the pair with its arguments swapped: the 90- and
# 270-degree members exchange, the others are their own mirror.
swap_rotation <- function(family) {
  if (family %in% c(301, 303, 401, 403)) {
    family + 1
  } else if (family %in% c(302, 304, 402, 404)) {
    family - 1
  } else {
    family
  }
}

family_of <- function(model) {
  if (isS4(model)) attr(model, "family") else model$family
}

# The copula of one pair at each row of `newdata`: VineCopula's family, the
# parameter and the second parameter. A pair is either a gamBiCop or a
# constant list(family, par, par2).
copula_at <- function(model, newdata) {
  if (isS4(model)) {
    par <- gamBiCopPredict(model, newdata = newdata, target = "par")$par
    par2 <- rep(model@par2, length(par))
  } else {
    par <- rep(model$par, nrow(newdata))
    par2 <- rep(model$par2, nrow(newdata))
  }
  list(family = vine_family(family_of(model), par), par = par, par2 = par2)
}

# The vine's density at each row of `v` (columns v1..vd) and `covs`, as the
# product of its pair densities. The R-vine matrix is read in VineCopula's
# convention: column i holds, at row k > i, the pair of M[k, i] and M[i, i]
# given M[(k + 1):d, i], in tree d - k + 1, whose copula takes the
# conditional distribution of M[k, i] first. Each conditional distribution is
# kept under its variable and conditioning set, as an h-function of a pair in
# the tree below.
pair_product <- function(object, v, covs) {
  M <- object@Matrix
  d <- nrow(M)
  index <- gamCopula:::get.modelCount(d)
  key <- function(j, given) {
    paste(j, paste(sort(given), collapse = " "), sep = "|")
  }
  cdf <- list()
  for (j in seq_len(d)) {
    cdf[[key(j, integer(0))]] <- v[[j]]
  }
  newdata <- cbind(v, covs)
  density <- rep(1, nrow(v))
  for (k in d:2) {
    for (i in seq_len(k - 1)) {
      a <- M[k, i]
      b <- M[i, i]
      given <- if (k < d) M[(k + 1):d, i] else integer(0)
      z1 <- cdf[[key(a, given)]]
      z2 <- cdf[[key(b, given)]]
      if (is.null(z1) || is.null(z2)) {
        refuse("the fitted matrix must satisfy the proximity condition; ",
               "pair (", a, ", ", b, " | ", paste(given, collapse = " "),
               ") has no conditional distribution below it")
      }
      model <- object@model[[index[k, i]]]
      pair <- cbind(data.frame(u1 = z1, u2 = z2), newdata)
      if (isS4(model)) {
        density <- density * gamBiCopPDF(model, newdata = pair)
      } else {
        cop <- copula_at(model, pair)
        density <- density * BiCopPDF(z1, z2, cop$family, cop$par, cop$par2,
                                      check.pars = FALSE)
      }
      cop <- copula_at(model, pair)
      cdf[[key(a, c(given, b))]] <- BiCopHfunc2(
        z1, z2, cop$family, cop$par, cop$par2, check.pars = FALSE
      )
      cdf[[key(b, c(given, a))]] <- BiCopHfunc1(
        z1, z2, cop$family, cop$par, cop$par2, check.pars = FALSE
      )
    }
  }
  density
}

# One row per pair copula of `object`: its place in the matrix, its pair and
# conditioning set, its family, and its calibration function.
pair_table <- function(object) {
  M <- object@Matrix
  d <- nrow(M)
  index <- gamCopula:::get.modelCount(d)
  rows <- list()
  for (k in d:2) {
    for (i in seq_len(k - 1)) {
      model <- object@model[[index[k, i]]]
      family <- family_of(model)
      gam <- isS4(model)
      rows[[length(rows) + 1]] <- data.frame(
        tree = d - k + 1,
        first = M[k, i],
        second = M[i, i],
        conditioning = paste(if (k < d) sort(M[(k + 1):d, i]) else c(),
                             collapse = " "),
        family = family,
        family_name = unname(FAMILY_NAMES[as.character(family)]),
        kind = if (gam) "gam" else if (family == 0) "independence" else
          "constant",
        formula = if (gam) {
          gsub("\\s+", " ", paste(deparse(model@model$formula), collapse = " "))
        } else {
          ""
        },
        covariates = if (gam) {
          paste(sort(all.vars(model@model$pred.formula)), collapse = " ")
        } else {
          ""
        },
        edf = if (gam) sum(model@model$edf) else if (family == 0) 0 else 1,
        par = if (gam) NA_real_ else model$par,
        par2 = if (gam) model@par2 else model$par2
      )
    }
  }
  do.call(rbind, rows)
}

# gamVineStructureSelect with each pair's h-functions those of the copula it
# fitted (see **Rotations**): copies of it and of the two tree fitters it
# calls, whose scope finds a pair fitter that fits unswapped and swaps the
# stored family after. Each copy keeps the package namespace as its scope's
# parent. Stops unless the package calls its pair fitter where this replaces
# it.
structure_select <- function() {
  ns <- asNamespace("gamCopula")
  scope <- new.env(parent = ns)
  fit_pair <- function(parameterForACopula, ...) {
    if (length(parameterForACopula) != 3) {
      refuse("every pair must be fitted with its covariates; got a pair ",
             "without")
    }
    out <- ns$fitAGAMCopula(parameterForACopula, ..., rotation = FALSE)
    family <- swap_rotation(family_of(out$model))
    if (isS4(out$model)) {
      attr(out$model, "family") <- family
    } else {
      out$model$family <- family
    }
    out
  }
  if (!"rotation" %in% names(formals(ns$fitAGAMCopula))) {
    refuse("gamCopula's fitAGAMCopula must take a rotation argument; this ",
           "gamCopula differs from the 0.0.8 this script was written against")
  }
  for (f in c("fitFirstTree", "fitTree", "gamVineStructureSelect")) {
    copy <- get(f, envir = ns)
    if (f != "gamVineStructureSelect" &&
        !any(grepl("wrapper.fitACopula", deparse(body(copy)), fixed = TRUE))) {
      refuse("gamCopula's ", f, " must fit its pairs through ",
             "wrapper.fitACopula; this gamCopula differs from the 0.0.8 ",
             "this script was written against")
    }
    environment(copy) <- scope
    assign(f, copy, envir = scope)
  }
  assign("wrapper.fitACopula", fit_pair, envir = scope)
  get("gamVineStructureSelect", envir = scope)
}

set.seed(seed)
train <- read.csv(file.path(dir, "train.csv"))
points <- read.csv(file.path(dir, "eval.csv"))
d <- sum(grepl("^y[0-9]+$", names(train)))
if (d < 2) {
  refuse("train.csv must have at least two columns y1, y2, ...; got ", d)
}
Y <- paste0("y", seq_len(d))
U <- paste0("u", seq_len(d))
V <- paste0("v", seq_len(d))
check_columns(train, "train.csv", c("x", Y))
check_columns(points, "eval.csv", c("x", Y, U))
inside <- vapply(points[U], function(u) all(u > 0 & u < 1), logical(1))
if (!all(inside)) {
  refuse("eval.csv must hold values in (0, 1) in ",
         paste(U[!inside], collapse = ", "))
}

# The margins, and the training sample's transforms.
margin_fit <- timed(lapply(Y, function(y) {
  gam(value ~ s(x), data = data.frame(value = train[[y]], x = train$x))
}))
margins <- margin_fit$value
transform <- function(y, x) {
  clamped <- 0
  columns <- lapply(seq_len(d), function(j) {
    m <- margins[[j]]
    mu <- as.numeric(predict(m, newdata = data.frame(x = x)))
    z <- (y[[j]] - mu) / sqrt(m$sig2)
    p <- pnorm(z)
    clamped <<- clamped + sum(p < PIT_EPS | p > 1 - PIT_EPS)
    list(v = pmin(pmax(p, PIT_EPS), 1 - PIT_EPS),
         log_pdf = dnorm(z, log = TRUE) - 0.5 * log(m$sig2))
  })
  v <- as.data.frame(setNames(lapply(columns, `[[`, "v"), V))
  log_pdf <- Reduce(`+`, lapply(columns, `[[`, "log_pdf"))
  list(v = v, log_pdf = log_pdf, clamped = clamped)
}
fitted_margins <- transform(train[Y], train$x)
vdata <- fitted_margins$v
covs <- data.frame(x = train$x)

settings <- list(familyset = NA, familycrit = "AIC", level = 0.05,
                 n.iters = N_ITERS)
vine_fit <- timed(switch(arm,
  "gamvine-oracle" = {
    M <- unname(as.matrix(read.csv(file.path(dir, "structure.csv"),
                                   header = FALSE)))
    if (!identical(dim(M), c(d, d)) || RVineMatrixCheck(M) != 1) {
      refuse("structure.csv must hold a valid ", d, " x ", d, " R-vine ",
             "matrix in VineCopula's convention")
    }
    do.call(gamVineCopSelect, c(
      list(data = vdata, Matrix = M, smooth.covs = covs, simplified = TRUE,
           rotations = FALSE),
      settings
    ))
  },
  "gamvine-selected" = do.call(structure_select(), c(
    list(udata = vdata, smooth.covs = covs, simplified = TRUE,
         treecrit = "tau"),
    settings
  )),
  "gamvine-ns" = do.call(structure_select(), c(
    list(udata = vdata, smooth.covs = covs, simplified = FALSE,
         treecrit = "tau"),
    settings
  ))
))
model <- vine_fit$value
if (!is(model, "gamVine")) {
  stop(arm, ": the fit must return a gamVine; got ", class(model)[1])
}

# The fitted copula density at `v`, with gamVinePDF checked on its first rows.
copula_density <- function(v, x) {
  data <- cbind(v, x = x)
  density <- gamVinePDF(model, data)
  if (!is.numeric(density)) {
    refuse("gamVinePDF must return densities; got ", density)
  }
  density
}

joint <- timed({
  margin_values <- transform(points[Y], points$x)
  list(
    pdf = exp(log(copula_density(margin_values$v, points$x)) +
                margin_values$log_pdf),
    margin_values = margin_values
  )
})
copula <- timed(copula_density(setNames(points[U], V), points$x))

rows <- seq_len(min(CHECK_ROWS, nrow(points)))
for (check in list(
  list(what = "the fitted transforms", v = joint$value$margin_values$v),
  list(what = "the true transforms", v = setNames(points[U], V))
)) {
  v <- check$v[rows, , drop = FALSE]
  covs_rows <- data.frame(x = points$x[rows])
  agreement <- all.equal(gamVinePDF(model, cbind(v, covs_rows)),
                         pair_product(model, v, covs_rows),
                         tolerance = 1e-10)
  if (!isTRUE(agreement)) {
    refuse("gamVinePDF must equal the product of the pair densities at ",
           check$what, "; got ", agreement)
  }
}

values <- data.frame(pdf = joint$value$pdf, copula_pdf = copula$value)
bad <- names(values)[colSums(!is.finite(as.matrix(values))) > 0]
if (length(bad) > 0) {
  stop("the fitted model must give a finite ", paste(bad, collapse = ", "),
       " at every evaluation point; got non-finite values")
}

pairs <- pair_table(model)
kinds <- table(factor(pairs$kind, c("gam", "constant", "independence")))
conditional <- if (arm == "gamvine-ns") {
  sum(grepl("v", pairs$covariates))
} else {
  0
}
# One line on the fitted model, which the harness logs.
cat(sprintf(
  paste0(
    "%s: %s, d %d, n %d; margins EDF %s; pairs: %d gam, %d constant, ",
    "%d independence, %d with a conditioning variable; copula EDF %.1f; ",
    "transforms clamped %d (training) and %d (test); n.iters %d\n"
  ),
  versions(c("gamCopula", "mgcv", "VineCopula")), arm, d, nrow(train),
  paste(sprintf("%.2f", vapply(margins, function(m) sum(m$edf), 0)),
        collapse = " "),
  kinds[["gam"]], kinds[["constant"]], kinds[["independence"]], conditional,
  sum(pairs$edf), fitted_margins$clamped, joint$value$margin_values$clamped,
  N_ITERS
))

pred <- cbind(
  points[c("x", Y, U)], values,
  margin_time = margin_fit$seconds, vine_time = vine_fit$seconds,
  joint_pdf_time = joint$seconds, copula_pdf_time = copula$seconds
)
write_exact(pred, file.path(dir, "pred.csv"))
write.table(model@Matrix, file.path(dir, "fitted.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
write_exact(pairs, file.path(dir, "pairs.csv"))
