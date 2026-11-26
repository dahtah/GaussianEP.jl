# SparseState backend (scenario 3B): sparse Q0 + sparse unary sites.
# Test case: 1D GMRF with Q0 = ρL + I (L = finite-difference Laplacian) and
# Bernoulli-logistic observations — the sparse analog of GP classification.
using SparseArrays, LinearAlgebra, Random

@testset "SparseState unit: updates vs dense reference" begin
    rng = MersenneTwister(11)
    p = 60
    # sparse PD Q0: tridiagonal Laplacian + ridge
    L = spdiagm(-1 => -ones(p-1), 0 => 2.0 .* ones(p), 1 => -ones(p-1))
    Q0 = sparse(L + 2.0I)
    r0 = randn(rng, p)
    st = GaussianEP.SparseState(Q0, r0)
    @test st isa GaussianEP.AbstractEPState
    @test !st.stale
    # reference dense computation
    Qd = Matrix(Q0); Σd = inv(Symmetric(Qd)); μd = Σd * r0
    @test maximum(abs.(st.μ .- μd)) < 1e-10
    @test abs(logdet(st) - logdet(Symmetric(Qd))) < 1e-10
    a = sparsevec([3, 17], [1.0, -0.5], p)
    Sa = GaussianEP.Σa(st, a)
    @test maximum(abs.(Vector(Sa) .- Σd * Vector(a))) < 1e-10
    # rank-1 update round-trip: apply then check Q, Σa, logdet, μ
    δq, δr = 0.7, -0.3
    applied = GaussianEP.apply_site!(st, a, δq, δr)
    @test applied == (δq, δr)
    # non-frozen rank-1 apply uses an in-place CHOLMOD updown: the factor stays
    # fresh (stale == false) and matches Q exactly
    @test !st.stale
    Q1 = Q0 + δq .* (a * a'); r1 = r0 .+ δr .* a
    @test nnz(st.Q - Q1) == 0
    @test maximum(abs.(st.r .- r1)) < 1e-14
    Σ1 = inv(Symmetric(Matrix(Q1)))
    @test maximum(abs.(Vector(GaussianEP.Σa(st, a)) .- Σ1 * Vector(a))) < 1e-9
    @test abs(logdet(st) - logdet(Symmetric(Matrix(Q1)))) < 1e-9
    @test maximum(abs.(st.μ .- Σ1 * r1)) < 1e-9
    # PD guard: a downdate with K = 1 + s·δq ≤ 0 is backtracked (halved until
    # K > 0), same semantics as the dense/cholesky backends — never applied
    # with K ≤ 0, so the refactorized factor always stays PD.
    s = dot(a, GaussianEP.Σa(st, a))
    bad = GaussianEP.apply_site!(st, a, -2.0 / s, 0.0)     # K = 1 - 2 < 0
    @test bad !== nothing
    δqb, _ = bad
    @test 1 + s * δqb > 0
    Q2 = Q1 + δqb .* (a * a')
    @test nnz(st.Q - Q2) == 0
    @test isposdef(Symmetric(Matrix(st.Q)))
    # reset! / sync!
    GaussianEP.reset!(st, Q0, r0)
    @test st.stale
    GaussianEP.sync!(st)
    @test !st.stale
    @test maximum(abs.(st.μ .- μd)) < 1e-10
end

@testset "SparseState: 1D GMRF logistic (GP-classification analog)" begin
    rng = MersenneTwister(7)
    p = 200
    xtrue = 0.4 .* cumsum(randn(rng, p))          # smooth latent field
    y = rand(rng, p) .< 1 ./ (1 .+ exp.(-xtrue))  # Bernoulli-logistic observations
    M = gmrf_1d(y, Logit(); ρ=1.0)
    @test M.Q0 isa SparseMatrixCSC
    @test nnz(M.Q0) == 3p - 2                     # tridiagonal
    @test all(s -> s.A isa SparseMatrixCSC, M.sites)

    # default dispatch: sparse Q0 → SparseState with backend=:auto
    Gauto = ep(M; maxcycles=100, tol=1e-8)
    @test Gauto.state isa GaussianEP.SparseState

    Gs = ep(M; backend=:sparse, maxcycles=100, tol=1e-8)
    Gd = ep(M; backend=:dense, maxcycles=100, tol=1e-8)
    Gc = ep(M; backend=:cholesky, maxcycles=100, tol=1e-8)
    @test Gs.state isa GaussianEP.SparseState
    @test maximum(abs.(mean(Gs) .- mean(Gd))) < 1e-8
    @test maximum(abs.(mean(Gs) .- mean(Gc))) < 1e-8
    @test abs(log_ml(Gs) - log_ml(Gd)) < 1e-8
    @test abs(log_ml(Gs) - log_ml(Gc)) < 1e-8
    @test maximum(abs.(cov(Gs) .- cov(Gd))) < 1e-6
    # posterior mean tracks the latent field (better than the prior mean 0)
    @test dot(mean(Gs), xtrue) > 0.5 * dot(xtrue, xtrue) / p * p / 10
    # parallel (batched) path on the sparse backend matches sequential
    Gp = ep(M; backend=:sparse, parallel=true, maxcycles=200, tol=1e-8, damp=0.5)
    @test abs(log_ml(Gp) - log_ml(Gs)) < 1e-4
    @test maximum(abs.(mean(Gp) .- mean(Gs))) < 1e-3
end

@testset "SparseState: 2D grid Ising matches dense" begin
    rng = MersenneTwister(3)
    m = 12                                       # 12x12 grid, p = 144
    p = m * m
    # 2D grid Laplacian + ridge as sparse Q0
    L = spdiagm(-1 => -ones(p-1), 0 => 4.0 .* ones(p), 1 => -ones(p-1))
    # remove wrap-around edges (i, i+1) where i ≡ 0 mod m
    drop = [i for i in 1:p-1 if i % m == 0]
    L[drop, drop .+ 1] .= 0.0
    L[drop .+ 1, drop] .= 0.0
    Q0 = sparse(L + 2.0I)
    M = gmrf_model(Q0, rand(rng, p) .> 0.5, IsingLik())
    Gs = ep(M; backend=:sparse, maxcycles=100, tol=1e-8)
    Gd = ep(M; backend=:dense, maxcycles=100, tol=1e-8)
    @test Gs.state isa GaussianEP.SparseState
    @test maximum(abs.(mean(Gs) .- mean(Gd))) < 1e-8
    @test abs(log_ml(Gs) - log_ml(Gd)) < 1e-8
end

@testset "SparseState: cached-marginal refresh mode (refresh=K, parallel)" begin
    rng = MersenneTwister(7)
    p = 200
    xtrue = 0.4 .* cumsum(randn(rng, p))
    y = rand(rng, p) .< 1 ./ (1 .+ exp.(-xtrue))
    M = gmrf_1d(y, Logit(); ρ=1.0)
    Gd = ep(M; backend=:dense, maxcycles=100, tol=1e-8)
    # refresh=1 (default) must reproduce the plain parallel path
    G1 = ep(M; backend=:sparse, parallel=true, refresh=1, maxcycles=200, tol=1e-8, damp=0.5)
    # larger refresh: stale marginals + reweighted quadrature — looser agreement
    G8 = ep(M; backend=:sparse, parallel=true, refresh=8, maxcycles=200, tol=1e-8, damp=0.5)
    G16 = ep(M; backend=:sparse, parallel=true, refresh=16, maxcycles=300, tol=1e-8, damp=0.5)
    @test abs(log_ml(G1) - log_ml(Gd)) < 1e-4
    @test abs(log_ml(G8) - log_ml(Gd)) < 1e-3
    @test abs(log_ml(G16) - log_ml(Gd)) < 1e-2
    @test maximum(abs.(mean(G8) .- mean(Gd))) < 1e-2
    @test maximum(abs.(mean(G16) .- mean(Gd))) < 5e-2
    # ESS-triggered refresh: ess_min huge => refresh every cycle => same as refresh=1
    Ge = ep(M; backend=:sparse, parallel=true, refresh=100, ess_min=1e9, maxcycles=200, tol=1e-8, damp=0.5)
    @test abs(log_ml(Ge) - log_ml(G1)) < 1e-3   # reweighted vs adapted nodes: same fixed point, different quadrature discretization
    # cache flag is reset after fit!
    @test !G8.ws.cache_on
end

@testset "scenario 3A: sparse Q0 + dense site rows (GLM with sparse prior)" begin
    # Sparse Q0 (diagonal-as-sparse, or GMRF Laplacian) with DENSE design rows
    # X[:,i] — the sparse+low-rank scenario of linear_algebra.md §3A. The direct
    # path (SparseState accumulating dense rank-1 updates) is correct but
    # densifies Q (nnz -> p^2); the (x, eta) expansion is the scalable design.
    rng = MersenneTwister(5)
    p, n = 60, 200
    X = randn(rng, p, n)
    β = 0.5 .* randn(rng, p)
    y = rand(rng, n) .< 1 ./ (1 .+ exp.(-(X' * β)))
    sites = [Site(reshape(X[:, i], 1, p), y[i], Logit()) for i in 1:n]
    L = spdiagm(-1 => -ones(p - 1), 0 => 2.0 .* ones(p), 1 => -ones(p - 1))
    for (name, Q0) in (("diag-sparse", sparse(0.01 * I, p, p)),
                       ("grmf", sparse(L + I)))
        M = EPModel(Q0, sites)
        Gd = ep(M; backend=:dense, maxcycles=100, tol=1e-9, damp=0.5)
        Gs = ep(M; backend=:auto, parallel=false, maxcycles=100, tol=1e-9, damp=0.5)
        @test Gs.state isa GaussianEP.SparseState
        @test abs(log_ml(Gs) - log_ml(Gd)) < 1e-8
        @test maximum(abs.(mean(Gs) .- mean(Gd))) < 1e-6
        # parallel refresh=1 (plain Jacobi semantics) on the GMRF prior
        if name == "grmf"
            Gp = ep(M; backend=:auto, parallel=true, refresh=1, maxcycles=100, tol=1e-9, damp=0.5)
            @test abs(log_ml(Gp) - log_ml(Gd)) < 1e-6
            @test maximum(abs.(mean(Gp) .- mean(Gd))) < 1e-4
        end
    end
end
