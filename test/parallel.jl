# Large-n: parallel (Jacobi, damped, threaded) updates should converge to
# the same EP fixed point as sequential updates.
@testset "parallel_updates" begin
    p, n = 4, 2000
    X = randn(p, n)
    y = rand(n) .> .5
    τ = 0.1
    M = glm_model(X, y, Logit(), τ)
    Gs = ep(M, qr=QuadRule(20), maxcycles=30, tol=1e-6)
    Gp = ep(M, qr=QuadRule(20), maxcycles=60, tol=1e-6, parallel=true, damp=0.5)
    @test isapprox(mean(Gs), mean(Gp), atol=1e-2)
    @test isapprox(cov(Gs), cov(Gp), atol=1e-2)
    @test isfinite(log_ml(Gp))
end
