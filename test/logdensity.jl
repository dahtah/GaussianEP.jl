# logdensity(G, x): log-density of the EP approximation at a point x
# (status.md TODO — convenience function for mode-finding with Optim.jl).
#   q(x) = N(x; μ, Σ) ∏ᵢ fᵢ(Aᵢx)/tᵢ(Aᵢx),  tᵢ(η) = exp(-½ η'Qᵢη + rᵢ'η)
# (a) on a Gaussian target EP is exact, so logdensity(G, x) equals the true
#     log density up to an x-independent constant;
# (b) the ForwardDiff gradient matches finite differences on a logistic GLM;
# (c) at the posterior mean the log-density exceeds that at a far-away point
#     (unimodal model).

using Random, ForwardDiff

@testset "logdensity: exact on a Gaussian target up to a constant" begin
    Random.seed!(11)
    m, n = 4, 6
    τ = 0.5
    Q0 = τ*Matrix{Float64}(I, m, m)
    # rank-2 Gaussian sites (GaussLik) — exercises the rank-d branch
    As = [randn(2, m) for _ in 1:n]
    ys = [randn(2) for _ in 1:n]
    M = EPModel(Q0, [Site(As[i], ys[i], GaussLik()) for i in 1:n])
    G = ep(M; maxcycles=200, tol=1e-12)
    # exact posterior: Q* = Q0 + Σ A'A, r* = Σ A'y (EP is exact on Gaussians)
    Qstar = Q0 + sum(A'A for A in As)
    rstar = sum(A'y for (A, y) in zip(As, ys))
    μex = Qstar \ rstar
    logp(x) = -0.5*dot(x .- μex, Qstar*(x .- μex)) - 0.5*logdet(Symmetric(Qstar)) - 0.5*m*log(2π)
    # at the fixed point tᵢ = fᵢ up to the constant -½‖yᵢ‖² - (d/2)log 2π
    c_ex = sum(-0.5*sum(abs2, y) - log(2π) for y in ys)   # d = 2
    xs = [randn(m) for _ in 1:5]
    cs = [logdensity(G, x) - logp(x) for x in xs]
    @test maximum(abs.(cs .- cs[1])) < 1e-7      # the constant is x-independent
    @test abs(cs[1] - c_ex) < 1e-7                # ... and equals the analytic value

    # rank-1 Gaussian sites (GLik) — same check on the rank-1 branch
    p, k = 3, 5
    X = randn(p, k)
    y1 = randn(k)
    Q0b = τ*Matrix{Float64}(I, p, p)
    M1 = EPModel(Q0b, [Site(reshape(X[:,i], 1, p), y1[i], GaussianEP.GLik()) for i in 1:k])
    G1 = ep(M1; maxcycles=200, tol=1e-12)
    Qs = Q0b + X*X'
    μs = Qs \ (X*y1)
    logp1(x) = -0.5*dot(x .- μs, Qs*(x .- μs)) - 0.5*logdet(Symmetric(Qs)) - 0.5*p*log(2π)
    c1_ex = sum(-0.5*abs2(y1[i]) - 0.5*log(2π) for i in 1:k)   # d = 1
    xs1 = [randn(p) for _ in 1:5]
    cs1 = [logdensity(G1, x) - logp1(x) for x in xs1]
    @test maximum(abs.(cs1 .- cs1[1])) < 1e-7
    @test abs(cs1[1] - c1_ex) < 1e-7
end

@testset "logdensity: ForwardDiff gradient matches finite differences (logistic GLM)" begin
    Random.seed!(22)
    p, n = 5, 40
    X = randn(p, n)
    y = rand(n) .> .5
    G = ep(glm_model(X, y, Logit(), 0.1); qr=QuadRule(20), maxcycles=100, tol=1e-10)
    x0 = randn(p)
    @test logdensity(G, x0) isa Real
    @test isfinite(logdensity(G, x0))
    g = ForwardDiff.gradient(x -> logdensity(G, x), x0)
    h = 1e-6
    gfd = zeros(p)
    for i in 1:p
        e = zeros(p); e[i] = h
        gfd[i] = (logdensity(G, x0 .+ e) - logdensity(G, x0 .- e)) / (2h)
    end
    @test isapprox(g, gfd, rtol=1e-4, atol=1e-5)
    @test maximum(abs.(g .- gfd)) < 1e-5
    # backend consistency: the natural-form Gaussian part must agree across
    # state backends (dense Woodbury vs exact Cholesky factor)
    Gc = ep(glm_model(X, y, Logit(), 0.1); qr=QuadRule(20), maxcycles=100, tol=1e-10, backend=:cholesky)
    @test abs(logdensity(G, x0) - logdensity(Gc, x0)) < 1e-6
end

@testset "logdensity: higher at the posterior mean than far away (unimodal logistic GLM)" begin
    Random.seed!(33)
    p, n = 4, 80
    X = randn(p, n)
    β = randn(p)
    y = (X'β + 0.5*randn(n)) .> 0
    G = ep(glm_model(X, y, Logit(), 0.1); qr=QuadRule(20), maxcycles=100, tol=1e-10)
    μ = mean(G)
    @test isfinite(logdensity(G, μ))
    # q is log-concave (logistic sites, PSD site precisions), hence unimodal;
    # a far-away point must have lower log-density than the posterior mean
    @test logdensity(G, μ) > logdensity(G, μ .+ 25.0)
    @test logdensity(G, μ) > logdensity(G, μ .+ 20.0 .* randn(p))
end
