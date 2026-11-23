# # Logistic regression with GaussianEP.jl
#
# Binary logistic regression fitted with expectation propagation (EP) on
# synthetic data. We build the model with `glm_model`, run `ep`, read off the
# EP log marginal likelihood with `log_ml`, then explore the regularisation
# path over the prior precision (`glm_regpath`, `glm_autotune`) and
# spike-and-slab variable selection.
#
# Run this file standalone with `julia --project=docs docs/notebooks/glm_logistic.jl`,
# or build the docs with `julia --project=docs docs/make.jl` — Literate then
# executes this file and renders it to `docs/src/notebooks/glm_logistic.md`.

# %% [markdown]
# ## 1. Synthetic data
#
# `p = 8` coefficients and `n = 300` observations. The design matrix `X` is
# `p × n` (one column per observation, as `glm_model` expects). The true
# coefficient vector is sparse: only 5 of the 8 entries are non-zero.

# %%
using GaussianEP, Random, Statistics, LinearAlgebra

Random.seed!(123)
p, n = 8, 300
X = randn(p, n)                                    # design matrix, p × n
βtrue = [2.0, -1.5, 1.0, 0.0, 0.0, 0.8, -0.6, 0.0] # sparse truth
sigmoid(z) = 1 / (1 + exp(-z))
y = rand(n) .< sigmoid.(X' * βtrue)                # Bernoulli responses, logistic link
println("n = $n observations, p = $p coefficients, $(sum(y)) positives")

# %% [markdown]
# ## 2. Fit the GLM with EP
#
# `glm_model(X, y, Logit())` builds the `EPModel`: a Gaussian prior
# `N(0, τ⁻¹I)` on the coefficients (`τ = 0.01` by default) as the base
# measure, plus one rank-1 `Logit()` site per observation. `ep` runs EP to
# convergence and returns the Gaussian approximation `G`; `log_ml(G)` is the
# EP estimate of the log marginal likelihood.

# %%
M = glm_model(X, y, Logit())
G = ep(M)
println("log_ml(G) = $(round(log_ml(G), digits = 4))")
println("coefficient   estimate   truth")
for j in 1:p
    println("  β[$j]        $(rpad(round(mean(G)[j], digits = 4), 10)) $(βtrue[j])")
end
nothing  # end cell with `nothing` so Literate captures the printed table

# %% [markdown]
# The same fit can be driven step by step: construct the `EPApprox` with
# `EPApprox(M)` and call `fit!` in place. Here we also swap the default
# Gauss–Hermite rule for the trapezoid rule `trap_rule()` — the robust
# quadrature choice for non-smooth / bounded likelihood factors.

# %%
G2 = EPApprox(M)
GaussianEP.fit!(G2; qr = trap_rule())
println("log_ml with trap_rule = $(round(log_ml(G2), digits = 4))")
nothing

# %% [markdown]
# ## 3. Regularisation path over the prior precision τ
#
# `glm_regpath` traces the posterior mean over a log-spaced grid of prior
# precisions `τ ∈ [τ0, τ1]` (traversed from the high-τ end down to τ0, each
# point warm-started from the previous one). It returns the EP log marginal
# likelihood at each point — for hyperparameter selection — plus a
# model-averaged mean `μavg` with weights `∝ exp(logml)`.

# %%
path = glm_regpath(X, y, Logit(), 1e-3, 10.0; nsteps = 12)
println("     τ        log_ml")
for k in eachindex(path.τs)
    println("  $(rpad(round(path.τs[k], digits = 4), 9)) $(round(path.logml[k], digits = 3))")
end
nothing

# %% [markdown]
# ## 4. Empirical-Bayes autotuning of τ
#
# `glm_autotune` runs the regularisation path and refits at the τ that
# maximises the EP log marginal likelihood, returning `(τ_opt, G, path)`.

# %%
(τ_opt, G_opt, path2) = glm_autotune(X, y, Logit(), 1e-3, 10.0; nsteps = 12)
println("τ_opt = $(round(τ_opt, digits = 4)),  log_ml(G_opt) = $(round(log_ml(G_opt), digits = 4))")
println("max |mean(G_opt) − mean(G)| = $(round(maximum(abs.(mean(G_opt) .- mean(G))), digits = 4))")
nothing

# %% [markdown]
# ## 5. Spike-and-slab variable selection
#
# `GaussianEP.spike_slab_glm` adds one `SpikeSlabLik(w, τ2)` prior site per
# coefficient on top of the likelihood sites (mixture prior
# `w δ(β) + (1−w) N(β; 0, τ2)`). After `ep`, the posterior inclusion
# probabilities `GaussianEP.inclusion_probs(G)` (PIPs) and the support
# estimate `GaussianEP.support(G)` (PIP > 0.5) recover the true sparsity
# pattern.

# %%
Mss = GaussianEP.spike_slab_glm(X, y, Logit(), 0.5, 10.0)
Gss = ep(Mss)
pips = GaussianEP.inclusion_probs(Gss)
supp = GaussianEP.support(Gss)
println("PIPs:    $(round.(pips, digits = 3))")
println("support: $supp   (true support: $(findall(!iszero, βtrue)))")
println("log_ml(Gss) = $(round(log_ml(Gss), digits = 4))")
nothing

# %% [markdown]
# ## 6. Plot: coefficient path along the regularisation path
#
# Posterior mean of each coefficient along the path (x-axis: log₁₀ τ), with
# the true non-zero values as dashed horizontal lines.

# %%
using CairoMakie

fig = Figure(size = (760, 420))
ax = Axis(fig[1, 1];
    xlabel = "log₁₀ τ (prior precision)", ylabel = "coefficient",
    title = "EP regularisation path — logistic regression (log_ml = $(round(log_ml(G), digits = 1)))")
for j in 1:p
    lines!(ax, log10.(path.τs), path.μs[j, :]; label = "β[$j]")
end
for j in findall(!iszero, βtrue)
    hlines!(ax, βtrue[j]; color = :black, linestyle = :dash, linewidth = 1)
end
axislegend(ax; position = :rt)
fig
