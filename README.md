# Neural Pair-Copulas Constructions (NPCCs)

A Python library for **conditional pair copula and fixed-structure vine density estimation**
built on top of *any* distributional-regression backend.  The package
exposes one outer estimator — `RosenblattBicop` — whose inner
univariate-conditional-predictive-distribution model is **pluggable**
(selected by name via `backend=`).  TabPFN is the default backend, but the
construction only needs a regressor that can produce a conditional
predictive distribution (`pdf`, `cdf`, `icdf`), so classical and
foundation-model regressors slot in on equal footing.

---

## Rosenblatt conditional bivariate copula

Let $(U, V, X)$ denote observations with $U, V \in (0, 1)$ and
$X \in \mathbb{R}^p$.  We want to estimate the conditional copula
density $c(u, v \mid x)$.

### Rosenblatt factorisation

Because $U$ is uniform on the copula scale, the Rosenblatt construction
collapses the bivariate density into a single univariate conditional
density,

$$c(u, v \mid x) = f_{V \mid U, X}(v \mid u, x).$$

So estimating $c$ reduces to estimating $f_{V \mid U, X}$, which is
exactly what a flexible **distributional regressor** provides.  The
features fed to the regressor are simply $W = [u, x]$ (or $[v, x]$ for
the reverse direction).

### Symmetric averaging

A single Rosenblatt direction is ordering-dependent: it satisfies
$\int_0^1 \hat c (u, v \mid x) \, dv = 1$ by construction but generally
not $\int_0^1 \hat c (u, v \mid x) \, du = 1$.  To reduce this
directional bias the estimator always fits both directions and
averages,

$$\hat c(u, v \mid x)
= \tfrac{1}{2} \, \hat f_{V \mid U, X}(v \mid u, x)
+ \tfrac{1}{2} \, \hat f_{U \mid V, X}(u \mid v, x).$$

This still does not impose exact uniform copula margins; if you need
that, enable the optional iterative-proportional-fitting / Sinkhorn
projection via `sinkhorn_iters`.

### Support transforms

Both copula scores live in $(0, 1)$.  The inner estimator can fit the
backend on a transformed target

$$Z = T(Y),$$

and convert the density back via the change-of-variables formula

$$f_Y(y \mid w) = f_Z(T(y) \mid w) \left|\tfrac{dT(y)}{dy}\right|.$$

Supported transforms: `transform="logit"` (default,
$\tfrac{1}{y(1-y)}$), `transform="probit"`
($\tfrac{1}{\phi(\Phi^{-1}(y))}$), and `transform="identity"`.  The
transform machinery lives on the backend-neutral base class, so every
backend gets it for free.

---

## Pluggable backends

The inner conditional-density model is any registered backend, chosen by
name:

| `backend=`             | Underlying model                          | Extra to install   |
| ---------------------- | ----------------------------------------- | ------------------ |
| `"tabpfn-criterion"`   | TabPFN native binned head (default)       | — (core)           |
| `"tabpfn-quantiles"`   | TabPFN quantile output, inverted          | — (core)           |
| `"tabpfn-finetune"`    | TabPFN fine-tuned on a scoring rule       | — (core)           |
| `"ngboost"`            | NGBoost parametric (analytic)             | `npcc[ngboost]`    |
| `"gbm"`                | scikit-learn quantile GBM                 | `npcc[gbm]`        |
| `"catboost"`           | CatBoost MultiQuantile                    | `npcc[catboost]`   |
| `"xgb-quantile"`       | XGBoost quantile trees (see note)         | `npcc[xgboost]`    |
| `"pytabkit-realmlp"`   | PyTabKit RealMLP                          | `npcc[pytabkit]`   |
| `"pytabkit-tabm"`      | PyTabKit TabM                             | `npcc[pytabkit]`   |
| `"tabicl"`             | TabICL foundation model                   | `npcc[tabicl]`     |
| `"tabicl-finetune"`    | TabICL fine-tuned on pinball loss         | `npcc[tabicl]`     |
| `"nori"`               | Synthefy Nori                             | `npcc[nori]`       |

`available_backends()` lists them, `documented_n_range(name)` gives a
backend's supported sample-size range, and `register_backend(name, factory)`
adds your own.  `npcc[backends]` installs every non-TabPFN extra at once.

`"xgb-quantile"` is registered but a poor choice on an unbounded support:
XGBoost quantile trees cannot extrapolate, so the predicted conditional
support collapses to roughly the inner `[0.24, 0.96]` of `(0, 1)` and the
quantile-to-density inversion returns exactly zero outside it.  See the note
in `npcc/core/registry.py`.

### Two base classes

Every backend implements the abstract
`ConditionalMargin` interface — `fit(y, x=w)`, `pdf(y, x=w)`,
`cdf(y, x=w)`, `icdf(alphas, x=w)`, plus the Cartesian-grid fast paths
`pdf_grid(y_grid, x=w)` / `cdf_grid(y_grid, x=w)`. There are two ways to
implement it:

- **Quantile-table backends** subclass `QuantileTableDistribution1D` and
  implement a single hook, `_predict_quantiles(w, alphas) -> (n, K)`.
  Everything else — the chunked, memory-safe quantile-table prediction,
  the monotone re-sort, and the predict-once-per-row pdf/cdf/icdf/grid
  inversion — is inherited.  This is the universal fast path used by
  `tabpfn-quantiles`, `gbm`, and `tabicl`.  Recovery details: pdf via
  $f(y \mid w) = 1 / Q'(\alpha)$ at $\alpha = F(y \mid w)$; cdf/icdf by
  linear interpolation in the sorted quantile table; $Q'$ floored to a
  positive constant for stability.

- **Native-evaluation backends** subclass `ConditionalMargin`
  directly and evaluate the predictive distribution at arbitrary points.
  `tabpfn-criterion` reads TabPFN's `criterion` head
  (`predict(W, output_type="full")` → logits + `pdf`/`cdf`/`icdf`); it is
  faster than, and (currently) as accurate as, the quantile read-out —
  hence the **default**.  `ngboost` uses the analytic SciPy frozen
  distribution (`pred_dist(W).dist`).

### Speed

Grid methods (`pdf_grid` / `cdf_grid`), which drive the Sinkhorn
projection and plotting, predict **at most once per conditioning row** and
evaluate every grid point by interpolation — never one forward pass per
grid cell.  Inference is chunked by `batch_size` (device-aware default:
400 on CPU, 2000 on CUDA) to bound memory.

---

## Public API

`RosenblattBicop` is the main entry point.  It is a `pyvinecopulib`
`BicopBase`, so it inherits `loglik`, `sample`, `plot` and the
`fit` / `select` / `from_data` estimator surface, and it follows the one
argument order those settled on: **the observations, then `controls`, then
keyword-only whatever the object cannot infer**.

Configuration travels as a `FitControlsRosenblattBicop` — including the
backend name, the device, the Sinkhorn iteration count, the `cdf` integration
grid, and the `batch_size` default, which is device-aware (400 on CPU, 2000 on
CUDA).  There are no per-call tuning keywords: `BicopBase` supplies `pdf`,
`cdf`, `hfunc1/2` and `hinv1/2` itself and forwards nothing but `x` to the
leaves this class writes, so every knob is a setting.

| Method | What it returns |
| --- | --- |
| `fit(u, controls=None, *, var_types=None, x=None)` | Fits both Rosenblatt directions on `u` of shape `(n, 2)`.  `x=None` → unconditional fit.  Returns `self`. |
| `from_data(u, controls=None, *, var_types=None, x=None)` | Constructs and fits in one call (inherited). |
| `pdf(u, *, x=None)` | Pointwise $\hat c(u_i, v_i \mid x_i)$ (inherited; this class supplies `_pdf_raw`). |
| `log_pdf(uv, *, x=None)` | $\log$ of `pdf`, floored at the smallest positive float. |
| `pdf_grid(u_grid, v_grid, *, x_row=None)` | Cartesian-grid density `out[i, j] = c(u_grid[i], v_grid[j] | x_row)`.  Available for every backend. |
| `cdf(u, *, x=None)` | Pointwise joint CDF, trapezoidal in $s$ and $t$ over `cdf_n_int` steps (inherited; this class supplies `_cdf_raw`). |
| `cdf_grid(u_grid, v_grid, *, x_row=None, n_int=64)` | Cartesian-grid joint CDF.  Available for every backend. |
| `hfunc1(u, *, x=None)` | $h_1 = \partial C / \partial u = F_{V \mid U, X}(v \mid u, x)$ (conditions on the first argument; matches `pyvinecopulib`). |
| `hfunc2(u, *, x=None)` | $h_2 = \partial C / \partial v = F_{U \mid V, X}(u \mid v, x)$. |
| `hinv1(u, *, x=None)` / `hinv2(u, *, x=None)` | Native inverses, read from the backend's own quantiles rather than root-found. |
| `tau(x_row=None, *, n=1000, seeds=None)` | Kendall's $\tau(x)$ via [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib)'s recipe: `ghalton(n, 2)` + inverse-Rosenblatt + `wdm`. |
| `plot(plot_type="surface", margin_type="unif", xylim=None, grid_size=None, *, x=None)` | Contour or surface plot, inherited from `BicopBase`. |

`npcc.__all__` carries 23 names, and `npcc.core` re-exports all but the two
concrete TabPFN backend leaves — either import path works.  Briefly: the four
estimators (`RosenblattBicop`, `RosenblattVinecop`, `RosenblattVinedist`,
`ConditionalMargin`), the fit controls (`FitControlsRosenblattBicop`,
`FitControlsRosenblattVinecop`, `Transform`), the quantile-table layer
(`QuantileTableConfig`, `QuantileTableDistribution1D`), the registry
(`BackendSpec`, `available_backends`, `create_backend`, `documented_n_range`,
`register_backend`, `validate_backend_kwargs`), the `TabPFNCriterionBackend` /
`TabPFNQuantileBackend` leaves, and a six-member error taxonomy rooted at
`NpccError`.

### Quick start

The public numerical API is **torch-only**: every method takes and returns
`torch.Tensor`.  Inputs are brought onto the estimator's dtype (`float64`) and
device by `_prep`, so a NumPy array or a float32 tensor is accepted and
converted rather than silently carried through at the wrong precision.

```python
from dotenv import load_dotenv
load_dotenv()  # picks up TABPFN_TOKEN from .env

import numpy as np
import pyvinecopulib as pv
import torch

from npcc import FitControlsRosenblattBicop, RosenblattBicop

# Sample from a Clayton bicop (pyvinecopulib works in NumPy)
clayton = pv.Bicop(
    family=pv.BicopFamily.clayton,
    parameters=np.asarray([[3.0]], dtype=np.float64),
)
u = torch.as_tensor(clayton.sample(n=1000, seeds=[2, 2, 4]))

# Fit the Rosenblatt copula (default backend: "tabpfn-criterion")
model = RosenblattBicop().fit(u)

# Pointwise density
uv = torch.tensor([[0.3, 0.4], [0.5, 0.6]], dtype=torch.float64)
print(model.pdf(uv))

# Cartesian-grid density (fast path, any backend)
grid = torch.linspace(0.05, 0.95, 30, dtype=torch.float64)
density = model.pdf_grid(grid, grid)   # shape (30, 30)

# Joint CDF and h-functions (pyvinecopulib convention: h_i conditions on arg i)
C = model.cdf_grid(grid, grid)         # shape (30, 30)
h1 = model.hfunc1(uv)                  # F_{V|U,X}
h2 = model.hfunc2(uv)                  # F_{U|V,X}

# Inherited from BicopBase: native inverse h-functions, sampling, loglik
v = model.hinv1(torch.tensor([[0.3, 0.25], [0.5, 0.75]], dtype=torch.float64))
sampled = model.sample(100, seeds=[42])
log_likelihood = model.loglik(u)

# Kendall's tau via the pyvinecopulib quasi-random recipe.
tau = model.tau()   # Clayton(theta=3) analytic: theta / (theta + 2) = 0.6

# Contour or surface plot, also inherited
model.plot(plot_type="contour", margin_type="norm")
```

Configuration is one object, and it is the second positional argument to
`fit` — so a covariate matrix goes in `x=`, never positionally:

```python
controls = FitControlsRosenblattBicop(backend="tabpfn-quantiles", device="cpu")
model = RosenblattBicop(controls)

# Non-TabPFN backends need the matching extra; backend-specific
# hyperparameters go in backend_kwargs.
RosenblattBicop(FitControlsRosenblattBicop(
    backend="ngboost", backend_kwargs={"n_estimators": 500},
))
RosenblattBicop(FitControlsRosenblattBicop(backend="gbm", backend_kwargs={"max_depth": 3}))

# To pin the TabPFN model version:
from tabpfn.constants import ModelVersion
RosenblattBicop(FitControlsRosenblattBicop(
    backend="tabpfn-criterion",
    backend_kwargs={"model_version": ModelVersion.V3},
))
```

With covariates:

```python
model.fit(uv_train, x=x_train)          # x_train shape (n, p)
model.pdf(uv_query, x=x_query)          # x_query shape (n_query, p)
```

---

## Fixed-structure multivariate vine

`RosenblattVinecop` composes fitted `RosenblattBicop` modules along a
caller-supplied `pyvinecopulib.RVineStructure`. The structure fixes the variable
order, edge layout, and truncation level; this estimator does not perform
automatic structure selection.

The vine is non-simplified. For an edge with conditioned variables
\(a_e, b_e\) and conditioning set \(D_e\), the pair copula receives

\[
   x_e = [u_{D_e}, x],
\]

where \(u_{D_e}\) contains the internal vine conditioning variables and \(x\)
contains optional external covariates. Consequently, the fitted vine density is

\[
   \hat c(u_1, \ldots, u_d \mid x)
   =
   \prod_e
   \hat c_{a_e,b_e;D_e}
   \left(
      u_{a_e\mid D_e},
      u_{b_e\mid D_e}
      \mid
      u_{D_e}, x
   \right).
\]

For tree-zero edges, \(D_e\) is empty, so those pairs receive only the external
covariates. Higher-tree pair copulas receive the conditioning-set values first and the
external covariates last.

A non-simplified vine is **not** built with the inherited `from_data`, whose
plain factory fits a simplified vine along a structure.  Construct the vine
with its conditioning context — which `RosenblattVinecop.__init__` installs —
and then `fit` it with `x`:

```python
import pyvinecopulib as pv
import torch

from npcc import FitControlsRosenblattVinecop, RosenblattVinecop

generator = torch.Generator().manual_seed(0)

# Continuous pseudo-observations and optional external covariates.
u_train = 0.05 + 0.9 * torch.rand((500, 3), generator=generator, dtype=torch.float64)
x_train = torch.randn((500, 2), generator=generator, dtype=torch.float64)

# The order and truncation level are fixed by the supplied structure.
structure = pv.RVineStructure.from_order([1, 2, 3])
controls = FitControlsRosenblattVinecop(backend="tabpfn-criterion", device="cpu")

vine = RosenblattVinecop(None, structure, device=controls.device)
vine.fit(u_train, controls, x=x_train)

u_query = 0.05 + 0.9 * torch.rand((2, 3), generator=generator, dtype=torch.float64)
x_query = torch.randn((2, 2), generator=generator, dtype=torch.float64)

density = vine.pdf(u_query, x=x_query)
independent = vine.rosenblatt(u_query, x=x_query)
recovered = vine.inverse_rosenblatt(independent, x=x_query)

samples = vine.sample(2, x=x_query, seeds=[42])
unconditional_samples = vine.sample(2, seeds=[42])

# With order [1, 2, 3], a one-column conditioning matrix conditions on
# variable 3, the current order tail.
u_cond = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
conditional_samples = vine.sample_conditional(u_cond, x=x_query, seeds=[42])
```

The vine supports continuous fixed structures only; automatic structure
selection and discrete variables are not implemented.  A joint CDF with
external covariates is not available, because it would need a separate Monte
Carlo sample for every covariate row.  For non-simplified vines,
`sample_conditional()` can condition only on variables already forming the
tail of the structure order.

---

## Original-scale vine distribution

`ConditionalMargin` adapts registered distributional-regression backends to
pyvinecopulib's `MarginBase` interface. `RosenblattVinedist` combines fitted
conditional margins with a `RosenblattVinecop` on the resulting pseudo
observations.

For observations \(Y=(Y_1,\ldots,Y_d)\) and optional covariates \(X\), each
margin supplies the conditional probability integral transform

\[
   U_j = F_j(Y_j \mid x),
\]

and the resulting conditional joint density is

\[
   f(y_1,\ldots,y_d \mid x)
   =
   c\!\left(
      F_1(y_1\mid x),\ldots,F_d(y_d\mid x)
      \mid x
   \right)
   \prod_{j=1}^d f_j(y_j\mid x).
\]

```python
import pyvinecopulib as pv
import torch

from npcc import (
   FitControlsRosenblattVinecop,
   QuantileTableConfig,
   RosenblattVinecop,
   RosenblattVinedist,
   create_backend,
)

generator = torch.Generator().manual_seed(42)
y_train = torch.randn((500, 3), generator=generator, dtype=torch.float64)
x_train = torch.randn((500, 2), generator=generator, dtype=torch.float64)

structure = pv.RVineStructure.from_order([1, 2, 3])
table_config = QuantileTableConfig()
controls = FitControlsRosenblattVinecop(
   backend="tabpfn-criterion",
   quantile_table_config=table_config,
   device="cpu",
)
margins = [
   create_backend(
      controls.backend,
      transform="identity",
      quantile_table_config=table_config,
      eps=controls.eps,
      device=controls.device,
      batch_size=controls.batch_size,
      backend_kwargs=controls.backend_kwargs,
   )
   for _ in range(structure.dim)
]
vinecop = RosenblattVinecop(None, structure, device=controls.device)
dist = RosenblattVinedist(vinecop, margins).fit(y_train, controls, x=x_train)

y_query = torch.randn((5, 3), generator=generator, dtype=torch.float64)
x_query = torch.randn((5, 2), generator=generator, dtype=torch.float64)

density = dist.pdf(y_query, x=x_query)
independent = dist.rosenblatt(y_query, x=x_query)
recovered = dist.inverse_rosenblatt(independent, x=x_query)
samples = dist.sample(5, x=x_query, seeds=[42])
```

One `FitControlsRosenblattVinecop` configures the backend family used by the
margins and pair copulas.

The initial implementation supports continuous real-valued margins only.
Every margin uses the identity target transform, while pair copulas default to
the logit transform on the unit interval. A fixed `RVineStructure` is required.
Custom margins, observation weights, automatic structure selection, and
variable names are not currently supported.

---

## Notebooks

Worked demos live under [`notebooks/`](notebooks/): a conditional pair-copula
demo across backends, a fixed-structure vine, an original-scale vine
distribution, and the simulation study.  They need a `TABPFN_TOKEN` (see
below) to run against the real TabPFN model, and `make test-notebooks`
executes them under pytest so a stale demo fails somewhere.

---

## Setup

### Install

```bash
# Pick exactly one PyTorch flavor: cpu, cu126, cu128, cu130, cu132
uv sync --extra cpu

# Optionally add non-TabPFN backends:
uv sync --extra cpu --extra ngboost --extra gbm --extra tabicl
```

The declared dependencies are `pyvinecopulib>=1.0.0`, `tabpfn>=8.0` and
`torch>=2.5`; the flavor extras only pin which PyTorch build is installed.
All three resolve from PyPI, so nothing here builds a C++ extension.

### Authenticate TabPFN (one-time)

`tabpfn` runs locally but authenticates once via a token from the
PriorLabs portal.

1. Go to <https://ux.priorlabs.ai>, log in, accept the `priorlabs-1-1`
   license on the **Licenses** tab, and copy your API key from the
   **Account** tab.
2. Drop it into a `.env` file at the repo root:

   ```
   TABPFN_TOKEN="..."
   ```

3. Call `dotenv.load_dotenv()` before fitting, or `export TABPFN_TOKEN=...`
   in your shell.

The first `fit(...)` downloads the TabPFN-v3 regressor checkpoint from
HuggingFace into the platform cache (Linux default: `~/.cache/tabpfn/`;
override with `TABPFN_MODEL_CACHE_DIR`).  Subsequent runs are offline.
(Only relevant to the TabPFN backends.)

### CPU sample-size cap

TabPFN refuses to fit on more than 1000 samples on CPU by default.  On
larger samples either use a CUDA build, set
`TABPFN_ALLOW_CPU_LARGE_DATASET=1`, or pass
`backend_kwargs={"model_kwargs": {"ignore_pretraining_limits": True}}`.

---

## Commands

```bash
make help             # list the targets
make lint             # ruff, ruff-format, numpydoc, codespell
make check            # lint plus `ty check`
make test             # pytest -n auto
make test-cov         # with a coverage report
make test-notebooks   # execute the notebooks (needs TABPFN_TOKEN)
make hooks            # install the pre-commit hooks
```

`make check` is what CI runs before the test matrix.  All of it is
zero-error: ruff (with `ANN`, `I`, `TC`, `PYI`, `RUF100`, `ERA001`),
numpydoc validation, a codespell prose ban list, and `ty` with its
off-by-default rules enabled.

The suite is hermetic by default: TabPFN is faked via a monkeypatched
regressor, and pluggable-backend behavior is proven end-to-end with a
TabPFN-free in-process backend.  A few tests hit the real models
(`test_real_tabpfn_smoke`, the TabICL smoke) and skip automatically when
their dependency/credentials are absent.
