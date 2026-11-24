# # Sparse GMRF backend (`SparseState`) — 1D latent field with GaussianEP.jl
#
# GaussianEP.jl can run EP on models with a **sparse** prior precision `Q0`
# using the `SparseState` backend: a CHOLMOD LDL' factorization of `Q` is
# cached and refreshed only when `Q` changes, so each site update costs one
# sparse triangular solve instead of a dense Woodbury update. With
# `backend=:auto` (the default), a sparse `Q0` automatically dispatches to
# `SparseState`; `backend=:dense` forces the dense Woodbury backend.
#
# The model here is a 1D Gaussian Markov random field (GMRF) with
# Bernoulli-logistic observations — the sparse analog of GP classification
# (see the `gp_classification_literate.jl` notebook). `gmrf_1d(y, Logit(); ρ)`
# builds the `EPModel` with prior precision `Q0 = ρ·L + I`, where `L` is the
# tridiagonal finite-difference Laplacian (diag 2, off-diag −1), and one
# rank-1 `Logit()` site per grid point (`A_i = e_i'`, sparse).
#
# Run this file standalone with `julia --project=docs docs/notebooks/gmrf_sparse.jl`,
# or build the docs with `julia --project=docs docs/make.jl` — Literate then
# executes this file and renders it to `docs/src/notebooks/gmrf_sparse.md`.

# %% [markdown]
# ## 1. Setup

# %%
using GaussianEP, SparseArrays, LinearAlgebra, Random, Statistics
ENV["GKSwstype"] = "100"   # headless GR rendering for Plots.jl
using Plots
gr()

# %% [markdown]
# ## 2. Toy 1D latent field with binary observations
#
# A smooth latent field `x_true` (random walk, `p = 200` grid points) is
# observed through a Bernoulli-logistic likelihood: `y_i ~ Bernoulli(σ(x_i))`.
# This is exactly the setup of `test/sparse_state.jl`.

# %%
rng = MersenneTwister(7)
p = 200
xtrue = 0.4 .* cumsum(randn(rng, p))           # smooth latent field
y = rand(rng, p) .< 1 ./ (1 .+ exp.(-xtrue))    # Bernoulli-logistic observations
println("p = $p grid points, $(sum(y)) positive observations")
nothing

# %% [markdown]
# ## 3. Build the GMRF model
#
# `gmrf_1d(y, Logit(); ρ=1.0)` returns an `EPModel` whose base measure is the
# Gaussian prior `N(0, Q0⁻¹)` with sparse tridiagonal `Q0 = ρL + I`, and whose
# sites are sparse unit rows `A_i = e_i'` with a `Logit()` likelihood.

# %%
M = gmrf_1d(y, Logit(); ρ=1.0)
println("Q0: $(typeof(M.Q0)), nnz = $(nnz(M.Q0)) (tridiagonal: $(3p-2))")
println("sites: $(length(M.sites)) rank-1 sites, A_i isa $(typeof(M.sites[1].A))")
println("model: $M")
nothing

# %% [markdown]
# ## 4. EP on a small problem: `backend=:auto` / `:sparse` vs `:dense`
#
# On a small `p` all backends are affordable, so we can check that the sparse
# backend agrees with the dense Woodbury backend. `backend=:auto` (default)
# dispatches the sparse `Q0` to `SparseState`.

# %%
Gauto = ep(M; maxcycles=100, tol=1e-8)                          # :auto → SparseState
Gs    = ep(M; backend=:sparse, maxcycles=100, tol=1e-8)
Gd    = ep(M; backend=:dense,  maxcycles=100, tol=1e-8)
println("backend  state type                  log_ml        max|Δμ| vs dense   |Δlog_ml|")
println("  :auto  $(rpad(string(typeof(Gauto.state)), 28)) $(rpad(round(log_ml(Gauto), digits=6), 12))")
println("  :sparse $(rpad(string(typeof(Gs.state)), 28)) $(rpad(round(log_ml(Gs), digits=6), 12)) $(round(maximum(abs.(mean(Gs) .- mean(Gd))), sigdigits=3))            $(round(abs(log_ml(Gs) - log_ml(Gd)), sigdigits=3))")
println("  :dense $(rpad(string(typeof(Gd.state)), 28)) $(rpad(round(log_ml(Gd), digits=6), 12)) (reference)")
println("cor(posterior mean, true field) = $(round(cor(mean(Gs), xtrue), digits=4))")
nothing

# %% [markdown]
# ## 5. Plot: posterior latent field vs. the true field
#
# EP posterior mean `mean(G)` with a ±2 sd credible band (`se(G)`), against
# the true latent field (dashed) and the binary observations (at ±1).

# %%
μ, s = mean(Gs), se(Gs)
plt = plot(1:p, μ;
    ribbon = 2 .* s, fillalpha = 0.25, color = :steelblue,
    label = "EP posterior mean ± 2sd", xlabel = "grid point i", ylabel = "latent x",
    title = "1D GMRF logistic via EP (sparse backend) — log_ml = $(round(log_ml(Gs), digits=2))",
    legend = :topleft, size = (760, 420))
plot!(plt, 1:p, xtrue; color = :black, linestyle = :dash, linewidth = 2, label = "true field")
scatter!(plt, 1:p, 2 .* y .- 1; color = :firebrick, markersize = 2, alpha = 0.4, label = "observations y (±1)")
plt

# %% [markdown]
# ## 6. Large `p`: `parallel=true` on the sparse backend, with timing
#
# At large `p` the dense backend is O(p³) time / O(p²) memory and becomes
# impractical, while the sparse backend stays near-linear in `p` (tridiagonal
# `Q0`). Sequential EP on `SparseState` refactorizes once **per site**
# (exact Gauss–Seidel, same semantics as `:dense`); `parallel=true` switches
# to damped Jacobi updates — one refactorization per sweep plus `p` threaded
# sparse solves — which is the scalable path. Here `p = 3000`:

# %%
rng = MersenneTwister(1)
pbig = 3000
xbig = 0.4 .* cumsum(randn(rng, pbig))
ybig = rand(rng, pbig) .< 1 ./ (1 .+ exp.(-xbig))
Mbig = gmrf_1d(ybig, Logit(); ρ=1.0)
println("p = $pbig, nnz(Q0) = $(nnz(Mbig.Q0))")

tpar = @timed ep(Mbig; backend=:sparse, parallel=true, damp=0.5, maxcycles=100, tol=1e-8)
Gpar = tpar.value
println("parallel=true  (Jacobi, 1 refactor/sweep): $(round(tpar.time, digits=2)) s, log_ml = $(round(log_ml(Gpar), digits=4)), final δ = $(round(Gpar.lastδ, sigdigits=2)), state = $(typeof(Gpar.state))")

tseq = @timed ep(Mbig; backend=:sparse, parallel=false, maxcycles=2, tol=1e-8)
println("sequential (1 refactor/site, 2 cycles only): $(round(tseq.time, digits=2)) s, log_ml = $(round(log_ml(tseq.value), digits=4))")
println("cor(posterior mean, true field) = $(round(cor(mean(Gpar), xbig), digits=4))")
nothing
