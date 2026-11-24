# Cached-marginal refresh mode (augmented-EP scheme) across backends:
# parallel sweeps with per-site marginals (m_i, s_i) cached and refreshed every
# `refresh` cycles; quadrature reweighted from the cached marginal to the
# current cavity (SNIS). SparseState freezes its factor and updates the sparse
# Q per site; dense/cholesky defer the state update and rebuild (Q, r) from the
# sites at each refresh. See sparse_backends.md §7.
using SparseArrays, LinearAlgebra, Random

@testset "refresh mode: 1D GMRF logistic, dense/cholesky backends" begin
    rng = MersenneTwister(7)
    p = 200
    xtrue = 0.4 .* cumsum(randn(rng, p))
    y = rand(rng, p) .< 1 ./ (1 .+ exp.(-xtrue))
    M = gmrf_1d(y, Logit(); ρ=1.0)
    Gd = ep(M; backend=:dense, maxcycles=100, tol=1e-8)
    for backend in (:dense, :cholesky)
        G1 = ep(M; backend=backend, parallel=true, refresh=1, maxcycles=200, tol=1e-8, damp=0.5)
        G8 = ep(M; backend=backend, parallel=true, refresh=8, maxcycles=200, tol=1e-8, damp=0.5)
        G16 = ep(M; backend=backend, parallel=true, refresh=16, maxcycles=300, tol=1e-8, damp=0.5)
        @test abs(log_ml(G1) - log_ml(Gd)) < 1e-4
        @test abs(log_ml(G8) - log_ml(Gd)) < 1e-3
        @test abs(log_ml(G16) - log_ml(Gd)) < 1e-2
        @test maximum(abs.(mean(G1) .- mean(Gd))) < 1e-2
        @test maximum(abs.(mean(G8) .- mean(Gd))) < 1e-2
        @test maximum(abs.(mean(G16) .- mean(Gd))) < 5e-2
        # ESS-triggered refresh (ess_min huge => refresh every cycle) ≈ refresh=1
        Ge = ep(M; backend=backend, parallel=true, refresh=100, ess_min=1e9, maxcycles=200, tol=1e-8, damp=0.5)
        @test abs(log_ml(Ge) - log_ml(G1)) < 1e-3
        @test !G8.ws.cache_on    # cache flag reset after fit!
    end
end

@testset "refresh mode: dense-X GLM (non-unary site rows, deferred apply)" begin
    rng = MersenneTwister(13)
    n, p = 100, 20
    X = randn(rng, n, p)
    β = randn(rng, p)
    y = rand(rng, n) .< 1 ./ (1 .+ exp.(-(X * β)))
    M = glm_model(X, y, Logit())
    Gd = ep(M; backend=:dense, maxcycles=2000, tol=1e-11)
    G1 = ep(M; backend=:dense, parallel=true, refresh=1, maxcycles=2000, tol=1e-11, damp=0.2)
    G8 = ep(M; backend=:dense, parallel=true, refresh=8, maxcycles=2000, tol=1e-11, damp=0.2)
    @test Gd.lastδ < 1e-8 && G1.lastδ < 1e-8        # both converged
    @test abs(log_ml(G1) - log_ml(Gd)) < 1e-6      # refresh=1 ≡ plain Jacobi: same fixed point
    @test abs(log_ml(G8) - log_ml(Gd)) < 0.5       # refresh=8: accepted cache drift (sanity)
    @test maximum(abs.(mean(G1) .- mean(Gd))) < 1e-4
    @test maximum(abs.(mean(G8) .- mean(Gd))) < 0.5
    # cholesky backend on the same GLM
    Gc = ep(M; backend=:cholesky, parallel=true, refresh=4, maxcycles=2000, tol=1e-11, damp=0.2)
    @test abs(log_ml(Gc) - log_ml(Gd)) < 0.5
    @test maximum(abs.(mean(Gc) .- mean(Gd))) < 0.5
end
