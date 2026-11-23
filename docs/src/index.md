# GaussianEP.jl

GaussianEP.jl implements **Expectation Propagation (EP)** for product-structure
models

```math
p(x) \propto \exp\!\left(-\tfrac{1}{2} x' Q_0 x + r_0' x\right) \prod_i f_i(A_i x),
\qquad x \in \mathbb{R}^m,
```

where each factor `f_i` is a (possibly non-Gaussian) likelihood attached to a
low-rank linear projection `A_i x` of the latent variable. Typical uses:

- **generalized linear models** (logistic / probit / Poisson regression) via
  [`glm_model`](@ref),
- **Ising models** and **interacting-particle (gas) models** via
  [`ising_model`](@ref) / [`gas_model`](@ref),
- **sparse and spike-and-slab linear models** via [`sparse_lm`](@ref) /
  [`spike_slab_lm`](@ref),
- any custom product-structure model assembled by hand from
  [`EPModel`](@ref) and [`Site`](@ref).

The EP loop runs on the generic [`EPModel`](@ref) core: build the model, call
[`ep`](@ref) (or construct an `EPApprox` and drive it with
[`GaussianEP.fit!`](@ref)), and read off the posterior moments and the EP
estimate of the log marginal likelihood with [`log_ml`](@ref).

## Quick start

Logistic regression on synthetic data — build a GLM with [`glm_model`](@ref),
run EP with [`ep`](@ref), and evaluate the approximate log marginal likelihood
with [`log_ml`](@ref):

```julia
using GaussianEP, Random, Statistics

Random.seed!(42)
p, n = 4, 200                       # p coefficients, n observations
X = randn(p, n)                     # design matrix, p × n
βtrue = [1.0, -0.5, 0.0, 2.0]
y = rand(n) .< (1 ./ (1 .+ exp.(-(X' * βtrue))))   # Bernoulli responses, logistic link

M = glm_model(X, y, Logit())        # EPModel: prior N(0, τ⁻¹I), one site per observation
G = ep(M)                           # run EP to convergence -> EPApprox

mean(G)                             # posterior mean of the coefficients
log_ml(G)                           # EP estimate of the log marginal likelihood
```

This recovers the planted coefficients (`mean(G) ≈ [0.99, -0.50, 0.23, 1.58]`
for the seed above) and gives `log_ml(G) ≈ -114.11`.

See the [API](api.md) page for the full reference and [Notebooks](notebooks.md)
for worked examples.
