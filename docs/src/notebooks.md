# Notebooks

Worked examples live as executable [Literate.jl](https://github.com/fredrikekre/Literate.jl)
scripts in `docs/notebooks/*.jl`. At build time `docs/make.jl` executes each
of them and renders the output to `docs/src/notebooks/*.md`, which is what
gets published below.

- GP classification — Gaussian-process classification with EP, rendered from
  `docs/notebooks/gp_classification.jl` into `notebooks/gp_classification.md`
  (the page appears here once the notebook lands and the docs are rebuilt).

To add your own notebook, drop a `my_example.jl` Literate script into
`docs/notebooks/` and rebuild the docs — it will appear here automatically.
See `docs/README.md` for details.
