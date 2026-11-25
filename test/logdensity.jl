# logdensity(M, x): log-density of the MODEL (the target EP approximates) at
# a point x, up to an additive constant:
#   log p(x) = -.5 x'Q0 x + r0'x + Σᵢ log_dens(likᵢ, Aᵢx, yᵢ)
# (a) on a Gaussian target this equals the exact log posterior up to a
#     constant (rank-d and rank-1 sites);
# (b) the ForwardDiff gradient matches finite differences and the analytic
#     gradient on a logistic GLM;
# (c) sparse/diagonal Q0 are not densified and give the same value as dense;
# (d) on a Gaussian target EP is exact, so logdensity(M, x) - log N(x; μ, Σ)
#     is constant in x (model density vs EP approximation agree).

using Random, ForwardDiff, SparseArrays

@testset "logdensity(M,x): exact on a Gaussian target up to a constant" begin
    Random.seed!(11)
    m, n = 4, 6
    τ = 0.5
    Q0 = τ*Matrix{Float64}(I, m, m)
    # rank-2 Gaussian sites (GaussLik) — exercises the rank-d branch
    As = [randn(2, m) for _ in 1:n]
    ys = [randn(2) for _ in 1:n]
    M = EPModel(Q0, [Site(As[i], ys[i], GaussLik()) for i in 1:n])
    # exact posterior: Q* = Q0 + Σ A'A, r* = Σ A'y
    Qstar = Q0 + sum(A'A for A in As)
    rstar = sum(A'y for (A, y) in zip(As, ys))
    μex = Qstar \ rstar
    logp(x) = -0.5*dot(x .- μex, Qstar*(x .- μex)) - 0.5*logdet(Symmetric(Qstar)) - 0.5*m*log(2π)
    xs = [randn(m) for _ in 1:5]
    cs = [logdensity(M, x) - logp(x) for x in xs]
    @test maximum(abs.(cs .- cs[1])) < 1e-10      # the constant is x-independent

    # rank-1 Gaussian sites (GLik) — the rank-1 branch
    p, k = 3, 5
    X = randn(p, k)
    y1 = randn(k)
    Q0b = τ*Matrix{Float64}(I, p, p)
    M1 = EPModel(Q0b, [Site(reshape(X[:,i], 1, p), y1[i], GaussianEP.GLik()) for i in 1:k])
    Qs = Q0b + X*X'
    μs = Qs \ (X*y1)
    logp1(x) = -0.5*dot(x .- μs, Qs*(x .- μs)) - 0.5*logdet(Symmetric(Qs)) - 0.5*p*log(2π)
    xs1 = [randn(p) for _ in 1:5]
    cs1 = [logdensity(M1, x) - logp1(x) for x in xs1]
    @test maximum(abs.(cs1 .- cs1[1])) < 1e-10
end

@testset "logdensity(M,x): gradient matches finite differences and analytic (logistic GLM)" begin
    Random.seed!(22)
    p, n = 5, 40
    X = randn(p, n)
    y = rand(n) .> .5
    τ = 0.1
    M = glm_model(X, y, Logit(), τ)
    x0 = randn(p)
    @test logdensity(M, x0) isa Real
    @test isfinite(logdensity(M, x0))
    g = ForwardDiff.gradient(x -> logdensity(M, x), x0)
    h = 1e-6
    gfd = zeros(p)
    for i in 1:p
        e = zeros(p); e[i] = h
        gfd[i] = (logdensity(M, x0 .+ e) - logdensity(M, x0 .- e)) / (2h)
    end
    @test isapprox(g, gfd, rtol=1e-4, atol=1e-5)
    # analytic: ∇log p = -Q0 x + r0 + Σᵢ Aᵢ' (yᵢ - σ(Aᵢx))   (logit link)
    ga = -τ*x0 .+ X*(y .- GaussianEP.logistic.(X'x0))
    @test maximum(abs.(g .- ga)) < 1e-8
end

@testset "logdensity(M,x): sparse/diagonal Q0 not densified, same value as dense" begin
    Random.seed!(44)
    p, n = 6, 30
    X = randn(p, n)
    y = rand(n) .> .5
    τ = 0.2
    Md = glm_model(X, y, Logit(), τ)
    Ms = EPModel(sparse(τ*Matrix{Float64}(I, p, p)), Md.sites)
    Mdiag = EPModel(Diagonal(τ*ones(p)), Md.sites)
    @test typeof(Ms.Q0) <: SparseMatrixCSC
    x0 = randn(p)
    @test logdensity(Ms, x0) ≈ logdensity(Md, x0) atol=1e-12
    @test logdensity(Mdiag, x0) ≈ logdensity(Md, x0) atol=1e-12
    # r0 ≠ 0 enters linearly
    r0 = randn(p)
    Mr = EPModel(τ*Matrix{Float64}(I, p, p), Md.sites; r0=r0)
    @test logdensity(Mr, x0) - logdensity(Md, x0) ≈ dot(r0, x0) atol=1e-12
end

@testset "logdensity(M,x): agrees with the exact EP approximation on a Gaussian target" begin
    Random.seed!(55)
    p, n = 4, 20
    X = randn(p, n)
    y = randn(n)
    M = glm_model(X, y, GaussianEP.GLik(), 0.5)
    G = ep(M; maxcycles=200, tol=1e-12)     # EP exact on Gaussian targets
    logN(x) = -0.5*dot(x .- mean(G), G.Q*(x .- mean(G))) - 0.5*logdet(Symmetric(G.Q)) - 0.5*p*log(2π)
    xs = [randn(p) for _ in 1:5]
    cs = [logdensity(M, x) - logN(x) for x in xs]
    @test maximum(abs.(cs .- cs[1])) < 1e-8   # model density = EP density up to a constant
end
