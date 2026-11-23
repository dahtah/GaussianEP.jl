# CholeskyState backend (backend=:cholesky): exact rank-1 lowrankupdate!/
# lowrankdowndate! on the Cholesky factor of Q. Must agree with the Woodbury
# DenseState (backend=:dense) on the same models, and the factor must stay PD.

using Random

@testset "CholeskyState unit: update/downdate round-trip and PD guard" begin
    rng = MersenneTwister(5)
    p = 12
    B = randn(rng, p, p)
    Q0 = Matrix(Symmetric(B * B')) + p * I
    r0 = randn(rng, p)
    st = GaussianEP.CholeskyState(Q0, r0)
    Qorig = Matrix(st.C)
    a = randn(rng, p)
    # rank-1 update then downdate -> back to the original factor
    applied = GaussianEP.apply_site!(st, a, 2.0, 0.5)
    @test applied !== nothing
    @test Matrix(st.C) ≈ Qorig + 2.0 * a * a' atol=1e-10
    applied = GaussianEP.apply_site!(st, a, -2.0, -0.5)
    @test applied !== nothing
    @test Matrix(st.C) ≈ Qorig atol=1e-8
    @test isposdef(Symmetric(Matrix(st.C)))
    # downdate that would break PD: backtracked (halved until PD), factor stays PD
    s = dot(a, st.C \ a)
    applied = GaussianEP.apply_site!(st, a, -2.0 / s, 0.0)   # K = 1 - 2 < 0
    @test isposdef(Symmetric(Matrix(st.C)))
    if applied !== nothing
        δq, _ = applied
        @test 1 + s * δq > 0
        @test Matrix(st.C) ≈ Qorig + δq * a * a' atol=1e-8
    end
    # rank-2 update via eigen-split
    A = randn(rng, 2, p)
    δQ = [1.0 0.3; 0.3 0.5]
    δr = [0.1, -0.2]
    Qbefore = Matrix(st.C)
    applied = GaussianEP.apply_site!(st, A, δQ, δr)
    @test applied !== nothing
    @test Matrix(st.C) ≈ Qbefore + A' * δQ * A atol=1e-8
    @test st.r ≈ r0 .+ A' * δr atol=1e-10
    @test st.μ ≈ (Matrix(st.C) \ st.r) atol=1e-8
    @test st.logdetQ ≈ 2 * sum(log, diag(cholesky(Symmetric(Matrix(st.C))).factors)) atol=1e-8
end

@testset "backend=:cholesky matches :dense on EP models" begin
    rng = MersenneTwister(9)
    # 1) logistic GLM (rank-1 quadrature sites)
    p, n = 6, 40
    X = randn(rng, p, n)
    y = rand(rng, n) .> 0.5
    M = glm_model(X, y, Logit(), 0.5)
    Gd = ep(M; qr=QuadRule(20), tol=1e-10, backend=:dense)
    Gc = ep(M; qr=QuadRule(20), tol=1e-10, backend=:cholesky)
    @test Gc.state isa GaussianEP.CholeskyState
    @test mean(Gc) ≈ mean(Gd) atol=1e-8
    @test cov(Gc) ≈ cov(Gd) atol=1e-7
    @test log_ml(Gc) ≈ log_ml(Gd) atol=1e-7
    @test Gc.Q ≈ Gd.Q atol=1e-8          # compat forwarding materialises Q
    @test Gc.Σ ≈ Gd.Σ atol=1e-7

    # 2) sparse LM (Laplace prior sites, quadrature)
    p, n = 5, 30
    X = randn(rng, p, n)
    β = [2.0, 0, -1.5, 0, 0]
    y = X' * β .+ 0.5 .* randn(rng, n)
    M = sparse_lm(X, y, 1.0, 0.25)
    Gd = ep(M; qr=QuadRule(30), tol=1e-10, backend=:dense)
    Gc = ep(M; qr=QuadRule(30), tol=1e-10, backend=:cholesky)
    @test mean(Gc) ≈ mean(Gd) atol=1e-7
    @test log_ml(Gc) ≈ log_ml(Gd) atol=1e-6

    # 3) spike-and-slab LM (analytic sites, exercises downdates)
    M = spike_slab_lm(X, y, 0.8, 10.0, 0.25)
    Gd = ep(M; qr=QuadRule(20), tol=1e-10, backend=:dense)
    Gc = ep(M; qr=QuadRule(20), tol=1e-10, backend=:cholesky)
    @test mean(Gc) ≈ mean(Gd) atol=1e-7
    @test log_ml(Gc) ≈ log_ml(Gd) atol=1e-6
    @test inclusion_probs(Gc) ≈ inclusion_probs(Gd) atol=1e-6

    # 4) rank-2 sites (ConfineLik) — exercises the rank-d eigen-split apply
    m = 4
    lik = ConfineLik(η -> 10 * (sum(abs2, η) - 1)^2)
    A2 = [1.0 0.0 0.3 0.0; 0.0 1.0 0.0 0.2]     # 2×4
    M = EPModel(Matrix(I, m, m), [Site(A2, 0.0, lik), Site([0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0], 0.0, lik)])
    Gd = ep(M; qr=QuadRule(15), tol=1e-9, backend=:dense)
    Gc = ep(M; qr=QuadRule(15), tol=1e-9, backend=:cholesky)
    @test mean(Gc) ≈ mean(Gd) atol=1e-6
    @test cov(Gc) ≈ cov(Gd) atol=1e-5
    @test log_ml(Gc) ≈ log_ml(Gd) atol=1e-5

    # 5) gas model (rank-d pairwise sites)
    Mg = gas_model(4, 2, β=1.0, H=(r)->1/(r+0.5), V=(η)->0.5*sum(abs2, η), τ0=1e-3)
    Gd = ep(Mg; qr=QuadRule(15), tol=1e-9, backend=:dense)
    Gc = ep(Mg; qr=QuadRule(15), tol=1e-9, backend=:cholesky)
    @test mean(Gc) ≈ mean(Gd) atol=1e-6
    @test log_ml(Gc) ≈ log_ml(Gd) atol=1e-6
end

@testset "cholesky backend: the former hang model terminates" begin
    rng = MersenneTwister(42)
    p, n, k, snr = 50, 100, 3, 5.0
    d = nothing
    for r in 1:4
        X = randn(rng, p, n)
        S = randperm(rng, p)[1:k]
        β = zeros(p); β[S] = randn(rng, k) .* 3.0
        σ2 = Statistics.var(X' * β) / snr
        y = X' * β .+ sqrt(σ2) .* randn(rng, n)
        d = (X=X, y=y, σ2=σ2)
    end
    M = spike_slab_lm(d.X, d.y, 0.9, 100.0, d.σ2)
    G = ep(M; qr=QuadRule(20), tol=1e-6, maxcycles=30, backend=:cholesky)
    @test isfinite(log_ml(G))
    @test all(isfinite, mean(G))
    @test isposdef(Symmetric(G.Q))
end
