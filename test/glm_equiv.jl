# The general machinery must reproduce the original GLM path:
# ep(glm_model(X,y,lik,τ)) ≈ ep_glm(X,y,lik,τ=τ)
@testset "glm_equivalence" begin
    p, n = 3, 50
    X = randn(p, n)
    y = rand(n) .> .5
    τ = 0.1
    G1 = ep_glm(X, y, Logit(), τ=τ)
    G2 = ep(glm_model(X, y, Logit(), τ), qr=QuadRule(20))
    @test isapprox(mean(G1), mean(G2), atol=1e-4)
    @test isapprox(cov(G1), cov(G2), atol=1e-4)
    @test isapprox(log_ml(G1), log_ml(G2), atol=1e-3)
end
