# # Gaussian-process classification with GaussianEP.jl (Literate notebook)
#
# Binary GP classification with expectation propagation (EP), using the
# generic product-structure interface of GaussianEP.jl (`EPModel` + `ep`).
# This is the Literate version of the standalone script
# `docs/notebooks/gp_classification.jl` (same model, same seed, same ~95%
# test accuracy) — the docs build (`julia --project=docs docs/make.jl`)
# executes this file with Literate and renders it to
# `docs/src/notebooks/gp_classification_literate.md`.
#
# Model: latent function values `f = (f(x_1), …, f(x_n))` with GP prior
# `f ~ N(0, K)`, `K_ij = k(x_i, x_j)` (RBF kernel from KernelFunctions.jl),
# and Bernoulli likelihood with logistic link `p(y_i | f_i) = σ(f_i)` for
# `y_i = 1` (and `σ(-f_i)` for `y_i = 0`). The posterior
#
# ```math
# p(f \mid y) \propto \mathcal{N}(f; 0, K) \prod_i \sigma\bigl((2y_i-1)\, f_i\bigr)
# ```
#
# is non-Gaussian; EP approximates it by a Gaussian `q(f) = N(μ, Σ)`. In
# GaussianEP.jl this is an `EPModel` with
#
# * prior precision `Q0 = (K + εI)⁻¹` as the Gaussian base measure, and
# * one rank-1 site per training point, `A_i = e_i'` with a `Logit()`
#   likelihood.

# %% [markdown]
# ## 1. Setup

# %%
using GaussianEP, KernelFunctions
using LinearAlgebra, Statistics, Random
using CairoMakie

# %% [markdown]
# ## 2. Toy dataset
#
# 1-D inputs in `[-3, 3]`, labels from a threshold at `0` with 10% label
# noise. 60 points in total (`n ≤ 60` so EP runs in seconds): 40 train /
# 20 held-out test.

# %%
Random.seed!(2024)
n = 60
x = sort(6 .* rand(n) .- 3)          # inputs in [-3, 3]
y = (x .> 0) .!= (rand(n) .< 0.10)    # threshold at 0, 10% flipped labels
perm = randperm(n)
itr, ite = perm[1:40], perm[41:60]    # 40 train / 20 test
xtr, ytr = x[itr], y[itr]
xte, yte = x[ite], y[ite]
println("n_train = $(length(itr)), n_test = $(length(ite)); positives: train $(sum(ytr))/40, test $(sum(yte))/20")

# %% [markdown]
# ## 3. Kernel matrix and EP model
#
# RBF (squared-exponential) kernel from KernelFunctions.jl with lengthscale
# `ℓ = 1` and unit signal variance. The GP prior `N(0, K)` becomes the
# Gaussian base measure of the `EPModel` through its precision
# `Q0 = (K + 1e-6 I)⁻¹`; each training point contributes one `Logit()` site
# with `A_i = e_i'` (unit row), so the site factor is `σ((2y_i-1) f_i)`.

# %%
kern = with_lengthscale(SEKernel(), 1.0)     # RBF kernel, lengthscale ℓ = 1
Xtr = reshape(xtr, 1, :)                      # KernelFunctions: observations are columns
K = kernelmatrix(kern, Xtr)                   # n_train × n_train kernel matrix
Q0 = inv(K + 1e-6 * I)                        # prior precision (jittered inverse)
unitrow(i, m) = (e = zeros(m); e[i] = 1.0; reshape(e, 1, m))
sites = [Site(unitrow(i, length(xtr)), ytr[i], Logit()) for i in 1:length(xtr)]
M = EPModel(Q0, sites)                        # p(f) ∝ exp(-½ f'Q0 f) ∏ᵢ σ((2yᵢ-1) fᵢ)
M

# %% [markdown]
# ## 4. Run EP
#
# `ep(M; qr = QuadRule(40))` runs expectation propagation with 40-node
# Gauss–Hermite quadrature for the tilted moments of each `Logit()` site,
# and returns the Gaussian approximation `G` with posterior latent mean
# `mean(G)`, covariance `cov(G)` and EP log marginal likelihood `log_ml(G)`.

# %%
G = ep(M; qr = QuadRule(40))
μ, Σ = mean(G), cov(G)
lml = log_ml(G)
println("log_ml            = $(round(lml, digits = 4))")
println("posterior latent mean μ = $(round.(μ, digits = 3))")
println("posterior latent sd     = $(round.(se(G), digits = 3))")

# %% [markdown]
# ## 5. Predictive distribution at test points
#
# With `q(f) = N(μ, Σ)` over the training latents and prior `N(0, K + εI)`,
# the predictive latent at new inputs is Gaussian with
#
# ```math
# μ* = K_*(K+εI)^{-1} μ, \qquad
# s²* = k_{**} - K_*(K+εI)^{-1}K_*^\top + K_*(K+εI)^{-1} Σ (K+εI)^{-1} K_*^\top .
# ```
#
# The predictive class probability uses the logit approximation
# `p(y* = 1 | x*) ≈ σ( μ* / √(1 + π s²*/8) )`.

# %%
sigmoid(z) = 1 / (1 + exp(-z))
function gp_predict(kern, xnew, xtr, Q0, μ, Σ)
    Xnew = reshape(xnew, 1, :)
    Ks = kernelmatrix(kern, Xnew, Xtr)        # n_new × n_train cross-kernel
    kss = diag(kernelmatrix(kern, Xnew))       # prior variances at new points
    A = Ks * Q0                                # K_* (K + εI)⁻¹
    μs = A * μ
    s2 = kss .- vec(sum(A .* Ks, dims = 2)) .+ vec(sum((A * Σ) .* A, dims = 2))
    return μs, s2
end
μte, s2te = gp_predict(kern, xte, xtr, Q0, μ, Σ)
pte = sigmoid.(μte ./ sqrt.(1 .+ π .* s2te ./ 8))
acc = mean((pte .> 0.5) .== yte)
println("predictive probabilities at the $(length(xte)) test points:")
println("  p = $(round.(pte, digits = 3))")
println("  y = $(Int.(yte))")
println("test accuracy = $(round(acc, digits = 3))  ($(round(Int, 100 * acc))%)")

# %% [markdown]
# ## 6. Plot
#
# EP latent mean ± 2 sd on a fine grid, with the training data (at `±1`,
# coloured by class) and the held-out test points (crosses).

# %%
xg = collect(range(-3.2, 3.2; length = 200))
μg, s2g = gp_predict(kern, xg, xtr, Q0, μ, Σ)
fig = Figure(size = (760, 440))
ax = Axis(fig[1, 1];
    xlabel = "x", ylabel = "latent f",
    title = "GP classification via EP — log_ml = $(round(lml, digits = 2)), test acc = $(round(acc, digits = 2))")
band!(ax, xg, μg .- 2 .* sqrt.(s2g), μg .+ 2 .* sqrt.(s2g); color = (:steelblue, 0.25), label = "latent mean ± 2sd")
lines!(ax, xg, μg; color = :steelblue, linewidth = 2, label = "EP latent mean")
scatter!(ax, xtr[ytr], ones(sum(ytr)); color = :firebrick, label = "train y = 1")
scatter!(ax, xtr[.!ytr], -ones(sum(.!ytr)); color = :royalblue, label = "train y = 0")
scatter!(ax, xte[yte], ones(sum(yte)); color = :firebrick, marker = :xcross, label = "test y = 1")
scatter!(ax, xte[.!yte], -ones(sum(.!yte)); color = :royalblue, marker = :xcross, label = "test y = 0")
axislegend(ax; position = :lt)
save(joinpath(@__DIR__, "gp_classification_literate.png"), fig)
println("figure saved to $(joinpath(@__DIR__, "gp_classification_literate.png"))")
fig
