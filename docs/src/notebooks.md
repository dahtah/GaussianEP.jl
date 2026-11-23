# Notebooks

Worked examples live as executable [Literate.jl](https://github.com/fredrikekre/Literate.jl)
scripts in `docs/notebooks/*.jl`. At build time `docs/make.jl` executes each
of them and renders the output to `docs/src/notebooks/*.md`, which is what
gets published below.

- [GLM logistic regression](notebooks/glm_logistic.md) — logistic regression
  with `glm_model` + `ep`: coefficients vs. truth, `log_ml`, the
  regularisation path (`glm_regpath`), empirical-Bayes autotuning
  (`glm_autotune`), the trapezoid rule (`trap_rule`), and spike-and-slab
  variable selection. Rendered from `docs/notebooks/glm_logistic.jl`.
- [GP classification](notebooks/gp_classification_literate.md) —
  Gaussian-process classification with EP (`EPModel` + `Logit()` sites, RBF
  kernel from KernelFunctions.jl, ~95% test accuracy). Rendered from
  `docs/notebooks/gp_classification_literate.jl`; the same example is also
  available as the standalone script `docs/notebooks/gp_classification.jl`
  (rendered to [notebooks/gp_classification.md](notebooks/gp_classification.md)).

To add your own notebook, drop a `my_example.jl` Literate script into
`docs/notebooks/` and rebuild the docs — it will appear here automatically.
See `docs/README.md` for details.
