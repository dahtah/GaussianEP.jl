# Moment rules: Stein (autodiff) vs direct quadrature vs fine-grid truth,
# and tensor quadrature for rank-2 sites.
@testset "stein_moments" begin
    for (lik, y) in ((Logit(), true), (Logit(), false), (PoisLik(), 2.0))
        m, s2 = 0.3, 0.8
        qr = QuadRule(40)
        z1, _, μ1, Σ1 = GaussianEP.tilted_moments(lik, y, [m], [s2;;], qr, stein=false)
        z2, _, μ2, Σ2 = GaussianEP.tilted_moments(lik, y, [m], [s2;;], qr, stein=true)
        qrf = QuadRule(300)
        zf, _, μf, Σf = GaussianEP.tilted_moments(lik, y, [m], [s2;;], qrf, stein=false)
        @test isapprox(z1, zf, rtol=1e-8)
        @test isapprox(μ1[1], μf[1], atol=1e-5)
        @test isapprox(Σ1[1,1], Σf[1,1], atol=1e-5)
        @test isapprox(z2, zf, rtol=1e-8)
        @test isapprox(μ2[1], μf[1], atol=1e-5)
        @test isapprox(Σ2[1,1], Σf[1,1], atol=1e-5)
    end
end

@testset "tensor_moments_rank2" begin
    # non-Gaussian rank-2 factor: exp(-V(η)) with quartic V
    lik = ConfineLik(η -> 0.5*sum(abs2, η) + 0.1*sum(abs2, η)^2)
    m = [0.2, -0.4]
    S = [0.8 0.1; 0.1 0.5]
    zc, _, μc, Σc = GaussianEP.tilted_moments(lik, nothing, m, S, QuadRule(25))
    zf, _, μf, Σf = GaussianEP.tilted_moments(lik, nothing, m, S, QuadRule(60))
    @test isapprox(zc, zf, rtol=1e-6)
    @test isapprox(μc, μf, atol=1e-5)
    @test isapprox(Σc, Matrix(Σf), atol=1e-5)
end
