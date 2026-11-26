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

@testset "mini-batch (batch=B) mode: stability where full Jacobi fails" begin
    # logistic GLM where full parallel (Jacobi) needs heavy damping but
    # mini-batch at damp=1.0 converges to the sequential fixed point
    rng = MersenneTwister(13)
    n, p = 100, 20
    X = randn(rng, n, p)
    β = randn(rng, p)
    y = rand(rng, n) .< 1 ./ (1 .+ exp.(-(X * β)))
    M = glm_model(X, y, Logit())
    Gd = ep(M; backend=:dense, maxcycles=2000, tol=1e-11)
    @test Gd.lastδ < 1e-8
    for B in (8, 16, 32)
        Gb = ep(M; backend=:dense, batch=B, maxcycles=2000, tol=1e-11, damp=1.0)
        @test Gb.lastδ < 1e-8                       # converges at damp=1.0
        @test abs(log_ml(Gb) - log_ml(Gd)) < 1e-6   # same fixed point as sequential
        @test maximum(abs.(mean(Gb) .- mean(Gd))) < 1e-4
    end
    # B = n recovers full Jacobi; B = 1 recovers sequential
    Gj = ep(M; backend=:dense, batch=n, maxcycles=50, tol=1e-11, damp=0.2)
    Gp = ep(M; backend=:dense, parallel=true, maxcycles=50, tol=1e-11, damp=0.2)
    @test abs(log_ml(Gj) - log_ml(Gp)) < 1e-10
    G1 = ep(M; backend=:dense, batch=1, maxcycles=2000, tol=1e-11, damp=1.0)
    @test abs(log_ml(G1) - log_ml(Gd)) < 1e-10
    @test G1.lastδ < 1e-8
    # sparse GMRF: mini-batch converges and matches dense sequential
    rng = MersenneTwister(7)
    p = 200
    xtrue = 0.4 .* cumsum(randn(rng, p))
    y = rand(rng, p) .< 1 ./ (1 .+ exp.(-xtrue))
    M = gmrf_1d(y, Logit(); ρ=1.0)
    Gd = ep(M; backend=:dense, maxcycles=100, tol=1e-8)
    Gb = ep(M; backend=:sparse, batch=64, maxcycles=100, tol=1e-8, damp=0.5)
    @test Gb.lastδ < 1e-6
    @test abs(log_ml(Gb) - log_ml(Gd)) < 1e-4
end

@testset "rank-B fused apply (DenseState apply_sites!)" begin
    rng = MersenneTwister(21)
    p, B = 30, 12
    Q0 = Matrix(Symmetric(randn(rng, p, p) * 0.1 + 5I))
    r0 = randn(rng, p)
    # fused vs B sequential rank-1 applies: identical up to roundoff
    st1 = GaussianEP.DenseState(Q0, r0)
    st2 = GaussianEP.DenseState(Q0, r0)
    A = randn(rng, p, B)
    δq = -0.05 .* rand(rng, B) .+ 0.02         # downdates, combined K stays PD
    δr = randn(rng, B)
    applied = GaussianEP.apply_sites!(st1, A, δq, δr)
    @test applied !== nothing
    for k in 1:B
        GaussianEP.apply_site!(st2, A[:, k], δq[k], δr[k])
    end
    @test maximum(abs.(st1.Q .- st2.Q)) < 1e-10
    @test maximum(abs.(st1.r .- st2.r)) < 1e-12
    @test maximum(abs.(st1.Σ .- st2.Σ)) < 1e-8
    @test maximum(abs.(st1.μ .- st2.μ)) < 1e-8
    @test abs(st1.logdetQ - st2.logdetQ) < 1e-8
    # non-PD combined update: returns nothing (caller falls back per-site)
    st3 = GaussianEP.DenseState(Q0, r0)
    @test GaussianEP.apply_sites!(st3, A, -100 .* ones(B), δr) === nothing
    # EP level: mini-batch with fused applies converges to the sequential fixed point
    p2, n = 50, 800
    X = randn(rng, p2, n)
    y = rand(rng, n) .> 0.5
    M = glm_model(X, y, Logit(), 0.1)
    Gs = ep(M; backend=:dense, maxcycles=200, tol=1e-11, verbose=false)
    Gb = ep(M; backend=:dense, batch=64, maxcycles=200, tol=1e-11, verbose=false)
    Gp = ep(M; backend=:dense, parallel=true, maxcycles=200, tol=1e-11, verbose=false)
    @test abs(log_ml(Gb) - log_ml(Gs)) < 1e-5
    @test abs(log_ml(Gp) - log_ml(Gs)) < 1e-5
    @test maximum(abs.(mean(Gb) .- mean(Gs))) < 1e-4
    @test maximum(abs.(mean(Gp) .- mean(Gs))) < 1e-4
end
