# AGENTS.md

Normative engineering spec for contributors and coding agents working on this
repository: scope, layers, conventions, and where each rule is written down.

For the user-facing pitch and the worked examples, see
[README.md](README.md). This file does not duplicate it.

## Project

Neural Pair-Copula Constructions (NPCCs) for multivariate conditional density
estimation. Python >= 3.11, managed with `uv`.

The package builds on [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib)
1.0's extension points: `RosenblattBicop` is a `BicopBase`,
`ConditionalMargin` a `MarginBase`, `RosenblattVinecop` a `VinecopBase`, and
`RosenblattVinedist` a `VinedistBase`. **Its conventions are this project's
conventions** wherever the two could differ — argument order, the fitting
idiom, module naming, the typing policy. Where this repository
departs, the departure is named below with its reason.

pyvinecopulib 1.0.0 is not on PyPI yet, so `[tool.uv.sources]` pins a git
revision — currently *ahead* of upstream `main`, at the pull request that
publishes the input pipeline's steps and names
`VinecopBase._invalidate_batched`. That is two steps, in order: the pin moves
to the merge commit once that lands, and the override goes away only once a
published 1.0.0 carries them. The `>=1.0.0` floor in `[project]` is already
the right specifier. The comment beside the pin says which symbols and why.

## Commands

Everything goes through `make`; run `make help` for the list.

```bash
# Install: pick exactly one PyTorch flavor (cpu, cu126, cu128, cu130, cu132),
# optionally with the backend and experiment extras.
uv sync --extra cpu
uv sync --extra cu130 --extra backends --extra experiments

make check            # lint + format-check + numpydoc + codespell, then `ty`
make test             # pytest -n auto
make test-notebooks   # executes the notebooks; needs TABPFN_TOKEN
make hooks            # install the pre-commit hooks
```

`make check` and `make test` are what CI runs, `check` running before `test`. Both
must be zero-error before a pull request.

## The layers, and which way they depend

```text
  npcc/__init__.py                          the public re-export surface
      |
      +--> npcc.experiments   (extra: experiments -- pandas, matplotlib)
      |         |
      v         v
  npcc.core                                 estimators and contracts
      |
      +--> npcc.core.backends               one adapter per third-party model
      |
      v
  pyvinecopulib.core                        the four extension points
```

Four rules, pinned by `tests/test_import_surface.py`:

1. **`import npcc` needs no optional extra.** Every non-TabPFN backend is
   reached through a registry factory that imports its adapter module *inside
   the function body*. That deferral is the whole of the guarantee, so
   `npcc.core.registry` must not import any `npcc.core.backends.*` module at
   module scope.
2. **An adapter module may import its own third-party package eagerly.** The
   module *is* that package's adapter; there is nothing to gain by deferring
   inside it. What must not happen is any other module importing it eagerly.
3. **`npcc.experiments` is above `npcc.core`** and the edge points one way.
   Nothing under `core` may name it.
4. **The two public surfaces agree.** `npcc.core.__all__` is a subset of
   `npcc.__all__`; the difference is exactly the two concrete TabPFN backend
   leaves, which live in `npcc.core.backends`.

`matplotlib`, `pandas` and `sklearn` are **not** treated as optional in that
test, because each is also a transitive hard dependency — `matplotlib` of
pyvinecopulib, `pandas` and `sklearn` of `tabpfn`. Asserting their absence
would pin a fact about someone else's dependency tree.

## Module naming

- **An unprefixed module is an import path.** It holds one public thing, or a
  small coherent set of them: `bicop.py`, `vinecop.py`, `vinedist.py`,
  `margin.py`, `margin_quantile_table.py`, `controls.py`, `errors.py`,
  `registry.py`.
- **An underscore-prefixed module is internal, and is named for what it
  does** — `_placement.py`, `_interp.py`, `_trim.py` — following
  pyvinecopulib's `_<level>_<thing>` scheme.
- **Names inside an underscore-prefixed module carry no second underscore.**
  `_placement.py` exports `resolve_device`, not `_resolve_device`. The
  module's prefix already says "not an import path".
- A module docstring does not restate the file name.

## Code style

- **Formatter**: ruff — line length 80, indent width 2, target Python 3.11.
  `*.ipynb` is included, so the notebooks are linted too.
- **Linter**: ruff with `ANN`, `I`, `TC`, `PYI`, `PGH003`, `RUF022`,
  `RUF045`, `RUF100`, `ERA001`, under `preview`. `ALL` is not
  used: these sets complement `ty` rather than restating the prose rules here.
- **Type checker**: `ty`, with its off-by-default rules enabled. Requires
  `ty>=0.0.73` — an older one warns `unknown-rule` and silently checks less.
- Modern syntax: `X | Y` unions, `list[T]` / `dict[K, V]`, `match` where it
  reads better. `pathlib.Path` over `os.path`.
- **No bare `Any` in a signature.** A signature that must accept anything
  states its reason at the site with `# noqa: ANN401 - <reason>`, and
  `RUF100` fails the build when one goes stale. A body that computes may hold
  a local `Any` and return through `cast(...)`; an `Any` in a *signature*
  erases the type for every caller and is published contract text.
- **Do not suppress diagnostics.** `# noqa` and `# ty: ignore` name the rule
  they suppress and say why. `blanket-ignore-comment` and
  `respect-type-ignore-comments = false` in `[tool.ty.rules]`, plus
  pre-commit's `python-check-blanket-noqa`, enforce the naming half.
- **Comments are documentation, not history.** Aim them at whoever reads the
  code next: the constraint, the invariant, or why a non-obvious choice is
  required. The test — *would this comment still make sense in a file that had
  never had the bug?* If it only reads as a contrast with what the code used
  to do, it belongs in the commit message.
- **No commented-out code.** `ERA001` checks it.
- **No `print` in library code** — use `logging`. `npcc.experiments` is a CLI
  and prints on purpose.

## Estimator API shape

Inherited from pyvinecopulib, and not to be diverged from:

- **One argument order: the observations, then `controls`, then keyword-only
  whatever the object cannot infer.** `fit`, `select` and `from_data` take
  `(data, controls)` positionally; `var_types`, `x`, `weights` and the
  callbacks are keyword-only. A covariate matrix passed positionally binds to
  `controls`, which is why `_bicop_controls` refuses an array in that slot
  with a message naming the order.
- **`fit` returns `self`; `from_data` constructs.** `RosenblattBicop()` must
  stay constructible with no arguments — `BicopBase.from_data` is
  `cls().select(...)`, which is how a vine fits one pair per edge.
- **Covariate layout is upstream's and is not widened at an entry point.**
  `(n,)` is refused because it says nothing about which axis is which;
  `covariate_row(x)` is the one reading the library names, for a single row
  shared across observations. One covariate per observation is a `(n, 1)`
  reshape, and it belongs to the *producer*, which computed `n` and can say
  which axis is which. An estimator only knows it received a covariate. This
  package widened `(n,)` at eight entry points once, and the cost was two
  contracts on one object: accepted on its own methods, refused on the
  inherited ones that call `prepare_covariates` themselves.
- **Configuration travels as a `ControlsLike`** — anything with `to_dict()`.
  A consumer reads the settings it owns and **refuses one it can neither
  honor nor delegate**, rather than dropping it silently. The same rule
  governs `supports_controls` / `supports_weights` / `supports_covariates`:
  a declared `False` means refuse, not ignore.
- **Capability flags are declared, not inferred**, and exist where a consumer
  reads them. Two default the *wrong* way for this package, so leaving either
  at its inherited value fails silently rather than loudly: `RosenblattBicop.supports_covariates` must be `True`, because
  `pair_eval` reads it *before* calling a leaf and a conditional pair is
  otherwise refused outright; and `RosenblattVinecop.supports_weights` must be
  `False`, because `VinecopBase` defaults it to `True` and weights ride in the
  controls, so the default accepts a weighted call and returns the unweighted
  fit. A leaf's signature is not what marks a pair conditional — the flag is.
- **Observation weights ride in the `ControlsLike`**, not in a `fit` argument.
  No `fit` / `select` / `from_data` in the stack takes `weights=`; each part
  refuses what it cannot honor through `supports_weights`. `var_type` and
  `support` travel beside `controls` on a margin's `fit` as *declarations*
  rather than configuration, and `ConditionalMargin` refuses both unless they
  say continuous and unbounded.
- **A per-call evaluation knob travels on the controls too**, because the base
  leaves no other route: `BicopBase`'s `pdf` / `cdf` / `hfunc*` / `hinv*` are
  dispatchers forwarding nothing but `x`, so a subclass keyword would have to
  come from a public override — and an override makes the `_*_raw` leaf beneath
  it unreachable, since every internal caller (`loglik`, the plot, each vine
  cascade) goes through the public member. `RosenblattBicop` therefore writes
  the six leaves and overrides no public member; `batch_size`, `sinkhorn_iters`
  and `cdf_n_int` are settings.
- **A pair copula carries its own `var_types`.** `RosenblattBicop` models no
  atoms, so it refuses a discrete declaration at both doors — `fit`, which the
  vine's engines reach first, and `with_var_types`, which they use to declare
  an edge once it is fitted. Refusing only in the all-continuous case would
  break the inverse-Rosenblatt cascade and the density plot, which call
  `with_var_types()` on every pair.

## Placement, layout, domain

Three separable steps on every input:

| step | where it runs | what it does |
|---|---|---|
| placement | the `_prep` hook | onto this estimator's dtype (`float64`) and device |
| layout | the `_layout` hook | which shapes are admissible |
| domain | `check_uv`, called from the `_prep_args` override | copula arguments into the open unit interval |

All three are hooks here. Upstream composes the first two and then applies its
own module-level `trim` inside `_prep_args`; `RosenblattBicop` **overrides**
`_prep_args` so `check_uv` runs in `trim`'s place. That is the single site
every evaluation member reaches, since `BicopBase` dispatches `pdf`, `cdf`,
`hfunc1/2` and `hinv1/2` through `_atoms`, which calls it first.
`_prepare_grid_inputs` rejects-then-clamps inline rather than calling
`check_uv`, because `check_uv` requires `u` and `v` to have equal shapes and a
grid pair is a cross product.

Three rules that are easy to get wrong:

- **Place every argument, not just the copula ones.** Covariates are placed
  and never clamped — they are arbitrary reals — but they *are* placed, since
  they get concatenated with values that live on the estimator's device.
- **The `_prep` hook is upstream's `TensorPlacementMixin`**, and it must be
  mixed in **ahead** of the canonical base: all four bases already inherit
  `PlacementMixin`, so a mixin placed after one never wins the lookup and
  `_prep` silently becomes the array-API inference again. What
  `TensorPlacement` adds is only the *declaration* — `float64` on `_device`,
  through `_ref_tensor`, which is the mixin's second resolution step. None of
  these estimators is an `nn.Module`, so its first step (a registered tensor)
  finds nothing and its third (an empty CPU `float64`) would put a CUDA
  estimator's inputs on the host.
- **`_set_placement`, not `self._device = ...`**, so every construction path
  records the device the hook reads.
- The hook's `torch.as_tensor` is also what keeps a gradient across a dtype
  change *whatever torch is installed*. The array-API route delegates to
  `torch.asarray`, whose `requires_grad` default changed — silently `False`
  on torch 2.11, `obj.requires_grad` from 2.13.

`check_uv` **departs** from pyvinecopulib's `trim` in both halves, and the
second half is what makes the `_prep_args` override mandatory rather than
stylistic.

- It **rejects** a copula argument at or outside `{0, 1}`, where `trim` clamps
  silently. Every score reaching an npcc estimator comes from a probability
  integral transform, so an exact 0 or 1 is a defect upstream rather than a
  rounding artifact, and clamping would turn it into a plausible number and
  hide it.
- It **clamps at `eps`**, the caller's, where `trim` clamps at the working
  precision — about `1e-10` in `float64`. The inner regressors are fitted on
  `logit(u)`, where those two are some ten units of feature space apart, so
  inheriting `_prep_args` would quietly move every near-boundary answer. That
  is the half nothing else would catch, and `tests/test_bicop.py` pins it on
  the hook rather than through a density, which would only notice it where the
  density varies.

Neither half reaches a pair sitting inside a vine. `VinecopBase._prep_args`
runs upstream's `trim` at the vine's own entry points, before the cascade
builds anything, so the pair only ever sees legal interior values and a
vine distribution reaches the same clamp through the vine it holds. Closing
that gap would mean overriding `_prep_args` at the vine level, which is new
behavior and not a decision this file has made.

**One NumPy boundary.** Everything that hands a tensor to a third-party model
goes through `pyvinecopulib.core.extend.to_numpy`, which detaches and transfers and
adopts no dtype. Two kinds of site stay outside it, and both should: five
transfers in the CatBoost and Nori adapters *name* a dtype while moving
(`to_numpy` would hand the model float64 where it wants float32), and nine
hand off to a CPU torch tensor rather than to NumPy, because that is what the
third-party call takes.

## Docstrings and prose

- **numpydoc**, validated in `make lint`. Sections in the numpydoc order;
  every parameter and return value typed. Private methods are excluded.
- A free-form heading inside a docstring parses as a numpydoc section and then
  fails the order check. Use a **bold run-in heading** instead:
  `**Approach.** ...`.
- **American English**, enforced by codespell's `en-GB_to_en-US` builtin.
- **A banned-word list**, `.codespell-prose.txt`, ported from pyvinecopulib.
  Each entry carries its reason after a comma, which is what makes codespell
  suggest rather than auto-fix: the right replacement depends on the sentence.
  The list is the source of truth and is not repeated here — a rule that
  quotes its own banned words fails itself.
- `tests/test_prose.py` covers what a tokenizer cannot see: multi-word
  phrases, and banned words hiding inside identifiers. To use a banned word
  where it is the right word, wrap the lines in `# codespell:ignore-begin` /
  `-end`.
- **Error messages** read `f"{name} must <requirement>; got {actual}"`, name
  the offending class, and say what to do instead. A missing-extra message
  carries the `pip install npcc[<extra>]` hint.

## Testing

- All new code has tests under `tests/`. One file per topic, flat.
- **Every test's docstring states the defect it exists to prevent.** A test
  that only restates the code's happy path has no reason to exist.
- Tests run in parallel (`-n auto`); no shared mutable state. The
  `register_uniform_backends` fixture pops what it registered on teardown.
- **A test that needs a GPU carries `@pytest.mark.cuda` and a `skipif`, and
  the mark goes on the parametrize *param*, not inside the body** — a mark
  applied from the body lands after collection and does neither. `-m "not
  cuda"` is a declared default in `addopts`, so `make test`, CI and a bare
  `pytest` all agree; `make test-cuda` is the way in. The `skipif` is what
  makes an explicit `-m cuda` safe on a machine without a device.
- Use fixtures; avoid bare `assert` on floats — `pytest.approx` or
  `torch.testing.assert_close`.
- Tests import from the public namespaces, `npcc` or `npcc.core`, except where
  a test's subject is an internal module.
- **The suite is hermetic by default.** TabPFN is faked via a monkeypatched
  regressor, and two closed-form `Uniform(-2, 2)` backends exercise the whole
  registry and vine cascade with no third-party model. A test that reaches a
  real model skips when its dependency or credentials are absent.
- A test that passes a wrong type on purpose says so in a comment and names
  the rule it suppresses.
- **Fix what you find.** A defect uncovered along the way is fixed, not
  annotated or worked around.

## Dependencies

- Runtime: `uv add <pkg>`. Dev/tooling: `uv add --group dev <pkg>`.
- Prefer lower bounds (`>=`); pin exactly only with a reason stated in a
  comment beside the pin.
- PyTorch extras (`cpu`, `cu126`, `cu128`, `cu130`, `cu132`) are mutually
  exclusive — never mix them. `make` exports `UV_NO_SYNC=1` so a bare
  `uv run` cannot silently reinstall a different flavor.
- Do not add backwards-compatibility shims or feature flags unless asked.

## Where each cross-cutting rule is written down

| Deciding | Read |
|---|---|
| whether a module gets an underscore | *Module naming* |
| whether an import may point somewhere | *The layers*, and `tests/test_import_surface.py` |
| where a new argument goes in a signature | *Estimator API shape* |
| what to do with a setting you cannot honor | *Estimator API shape* |
| whether to place, check, or clamp an input | *Placement, layout, domain* |
| how to get a tensor to a third-party model | *Placement, layout, domain* |
| whether an `Any` is allowed | *Code style* |
| how to word an error | *Docstrings and prose* |
| whether a comment should exist | *Code style* |

The rule is stated once, where the link points, and nowhere else. If a
reviewer or a coding agent keeps repeating the same correction, update this
file rather than relying on tribal knowledge — but do not add ephemeral or
machine-local preferences here.
