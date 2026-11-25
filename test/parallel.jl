# Large-n: parallel (Jacobi, damped, threaded) updates should converge to
# the same EP fixed point as sequential updates.
using Random

@testset "parallel_updates" begin
    p, n = 4, 2000
    X = randn(p, n)
    y = rand(n) .> .5
    τ = 0.1
    M = glm_model(X, y, Logit(), τ)
    Gs = ep(M, qr=QuadRule(20), maxcycles=30, tol=1e-6, crit=:strict)
    Gp = ep(M, qr=QuadRule(20), maxcycles=60, tol=1e-6, crit=:strict, parallel=true, damp=0.5)
    @test isapprox(mean(Gs), mean(Gp), atol=1e-2)
    @test isapprox(cov(Gs), cov(Gp), atol=1e-2)
    @test isfinite(log_ml(Gp))
end

# parallel=true is exactly batch=n (full mini-batch): identical sweeps, so
# identical results bit-for-bit on a Gaussian model (exact EP fixed point)
@testset "parallel == batch=n" begin
    rng = MersenneTwister(7)
    p, n = 3, 50
    X = randn(rng, p, n)
    y = randn(rng, n)
    M = glm_model(X, y, GaussLik(), 0.5)
    for backend in (:dense, :cholesky, :sparse)
        Gp = ep(M; backend=backend, parallel=true, maxcycles=100, tol=1e-12, verbose=false)
        Gb = ep(M; backend=backend, batch=n, maxcycles=100, tol=1e-12, verbose=false)
        @test mean(Gp) == mean(Gb)
        @test log_ml(Gp) == log_ml(Gb)
        @test Gp.lastδ == Gb.lastδ
        # batch=n also triggers the cached-marginal mode when eligible
        Gc = ep(M; backend=backend, batch=n, refresh=4, maxcycles=100, tol=1e-12, verbose=false)
        @test isapprox(mean(Gp), mean(Gc), atol=1e-8)
        @test isapprox(log_ml(Gp), log_ml(Gc), atol=1e-8)
    end
end
