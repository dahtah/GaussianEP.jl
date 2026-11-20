# Product distribution: p(x) ∝ N(x; 0, I/τ) ∏_i f_i(x_i) with logistic f_i.
# The posterior factorizes over coordinates; EP is exact per coordinate
# (up to quadrature error), since each cavity is the prior marginal.
_row(i, m) = (e = zeros(m); e[i] = 1.0; reshape(e, 1, m))

@testset "product_distribution" begin
    m = 5
    τ = 1.0
    y = rand(m) .> .5
    M = EPModel(τ*Matrix(I,m,m), [Site(_row(i,m), y[i], Logit()) for i in 1:m])
    G = ep(M, qr=QuadRule(100))
    # ground truth per coordinate by fine 1-D quadrature
    qr = QuadRule(200)
    for i in 1:m
        xq = sqrt(1/τ) .* qr.x
        lw = [GaussianEP.log_dens(Logit(), x, y[i]) + log(w) for (x,w) in zip(xq, qr.w)]
        w = exp.(lw .- GaussianEP.logsumexp(lw))
        μ = dot(w, xq)
        σ2 = dot(w, (xq .- μ).^2)
        @test mean(G)[i] ≈ μ atol=1e-6
        @test cov(G)[i,i] ≈ σ2 atol=1e-6
    end
    C = cov(G)
    @test norm(C - Diagonal(diag(C))) < 1e-8
end
