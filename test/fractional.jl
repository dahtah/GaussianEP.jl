# Fractional EP (power EP, Minka 2004; Seeger 2008 Sec. 3.5). Fractional EP
# does NOT change the target: the cavity removes only a fraction `power` of
# the site approximation, the tilted moments use f^power, and the site is
# updated by the convex combination Q_i ← (1-power) Q_i + dQ. For a Gaussian
# target the fixed point is the exact posterior for ANY power, and log_ml
# (computed with the true factor against the full cavity) is the exact
# evidence of the unchanged target.

using Random

@testset "fractional EP on a Gaussian target" begin
    Random.seed!(3)
    m = 4
    τ = 0.5
    Q0 = τ*Matrix{Float64}(I, m, m)
    sites = GaussianEP.AbstractSite[]
    Qstar = copy(Q0)
    rstar = zeros(m)
    n2 = 0     # total site dimension (for the evidence constant)
    ysq = 0.0  # Σ ||y_i||² (site constants)
    for _ in 1:3
        A = randn(2, m); y = randn(2)
        push!(sites, Site(A, y, GaussLik()))
        Qstar .+= A'A; rstar .+= A'y
        n2 += 2; ysq += sum(abs2, y)
    end
    X = randn(3, 4); y1 = randn(4)
    for i in 1:4
        A = zeros(1, m); A[1, 1:3] .= X[:,i]
        push!(sites, Site(A, y1[i], GaussianEP.GLik()))
        Qstar .+= A'A; rstar .+= A'y1[i]
        n2 += 1; ysq += abs2(y1[i])
    end
    M = EPModel(Q0, sites)
    μex = Qstar \ rstar
    Σex = inv(Symmetric(Qstar))
    lZu = (Q, r) -> .5*m*log(2π) - .5*logdet(Symmetric(Q)) + .5*dot(r, Q\r)
    logml_ex = lZu(Qstar, rstar) - .5*n2*log(2π) - .5*ysq - lZu(Q0, zeros(m))
    # power > 1 extrapolates, so it needs damping to converge
    for (power, damp) in ((0.3, 1.0), (0.5, 1.0), (1.0, 1.0), (2.0, 0.5), (3.0, 0.3))
        G = ep(M; qr=QuadRule(20), maxcycles=500, tol=1e-12, power=power, damp=damp)
        @test maximum(abs.(mean(G) .- μex)) < 1e-8
        @test maximum(abs.(cov(G) .- Σex)) < 1e-8
        @test abs(log_ml(G) - logml_ex) < 1e-8
    end
end

@testset "fractional EP keeps the target (Laplace sparse LM)" begin
    # log-concave sites (Seeger's setting): fractional EP converges to a
    # fixed point close to standard EP's, approximating the same posterior
    Random.seed!(42)
    p, n, λ, σ2 = 2, 6, 1.0, 1.0
    X = randn(p, n)
    y = X' * randn(p) + sqrt(σ2) * randn(n)
    M = sparse_lm(X, y, λ, σ2)
    G1 = ep(M; qr=QuadRule(40), maxcycles=500, tol=1e-11, power=1.0)
    G05 = ep(M; qr=QuadRule(40), maxcycles=500, tol=1e-11, power=0.5)
    # same target: the two fixed points are close to each other ...
    @test maximum(abs.(mean(G1) .- mean(G05))) < 5e-2
    @test maximum(abs.(cov(G1) .- cov(G05))) < 1e-1
    @test abs(log_ml(G1) - log_ml(G05)) < 5e-2
    # ... and both are close to the exact posterior (grid quadrature)
    L, N = 12.0, 601
    g = collect(range(-L, L; length=N))
    h = g[2] - g[1]
    A = zeros(p, N^p)
    for (k, I) in enumerate(CartesianIndices((1:N, 1:N)))
        A[:,k] .= (g[I[1]], g[I[2]])
    end
    r = y .- X' * A
    ll = -vec(sum(abs2, r; dims=1)) ./ (2σ2)
    lp = vec(-λ * sum(abs, A; dims=1) .+ p * log(λ/2))
    lw1 = ll .+ lp
    m = maximum(lw1)
    lse = m + log(sum(exp.(lw1 .- m)))
    w = exp.(lw1 .- lse)
    μex = vec(A * w)
    Σex = zeros(p, p)
    for k in 1:size(A,2)
        δ = A[:,k] .- μex
        Σex .+= w[k] * (δ * δ')
    end
    @test maximum(abs.(mean(G05) .- μex)) < 5e-2
    @test maximum(abs.(cov(G05) .- Σex)) < 1e-1
end
