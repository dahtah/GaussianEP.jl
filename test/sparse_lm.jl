# Sparse linear model (Seeger 2008): y = X a + ε, ε ~ N(0, σ² I), Laplace
# prior p(a_i) = (λ/2) exp(-λ|a_i|). The non-Gaussian sites are the prior
# factors; the Gaussian likelihood is the base measure Q0, r0.
# For m ≤ 3 coefficients the exact posterior is computed by brute-force
# grid quadrature and compared against EP.

using Random

function _lse(v)
    m = maximum(v)
    m + log(sum(exp.(v .- m)))
end

# exact posterior moments and log p(y) (in the EPModel convention: the base
# measure is exp(-.5 a'Q0 a + r0'a), so the ||y||²/(2σ²) term cancels in the
# ratio) by tensor grid quadrature over [-L,L]^p
function exact_sparse_lm(X, y, λ, σ2; L=12.0, N=201)
    p = size(X, 1)
    grids = ntuple(_ -> collect(range(-L, L; length=N)), p)
    vol = (grids[1][2] - grids[1][1])^p
    npts = N^p
    A = zeros(p, npts)
    for (k, I) in enumerate(CartesianIndices(ntuple(_ -> 1:N, p)))
        A[:,k] .= ntuple(j -> grids[j][I[j]], p)
    end
    r = y .- X' * A                          # n × npts residuals
    ll = -vec(sum(abs2, r; dims=1)) ./ (2σ2) # log likelihood, no constant
    lp = vec(-λ * sum(abs, A; dims=1) .+ p * log(λ/2))
    lw1 = ll .+ lp
    lZ1 = _lse(lw1) + log(vol)
    lZ0 = _lse(ll) + log(vol)
    w = exp.(lw1 .- _lse(lw1))
    μ = vec(A * w)
    Σ = zeros(p, p)
    for k in 1:npts
        δ = A[:,k] .- μ
        Σ .+= w[k] * (δ * δ')
    end
    (μ=μ, Σ=Σ, log_ml=lZ1 - lZ0)
end

@testset "sparse linear model" begin
    # structure: Gaussian likelihood in Q0/r0, one Laplace unary site per coef
    X = [1.0 2.0 0.5; 0.0 1.0 -1.0]
    y = [0.3, -0.2, 1.1]
    λ, σ2 = 1.5, 0.7
    M = sparse_lm(X, y, λ, σ2)
    @test M.Q0 ≈ X*X'/σ2 + 1e-8*Matrix{Float64}(I, 2, 2)
    @test M.r0 ≈ X*y/σ2
    @test length(M.sites) == 2
    @test all(s -> s.lik isa LaplaceLik && s.lik.λ == λ, M.sites)
    @test M.sites[1].A ≈ reshape([1.0, 0.0], 1, 2)
    @test M.sites[2].A ≈ reshape([0.0, 1.0], 1, 2)

    # EP vs brute-force quadrature, m = 1, 2, 3
    Random.seed!(42)
    cases = [(1, 5, 1.0, 1.0, 4001), (2, 6, 1.0, 1.0, 601), (3, 8, 1.0, 0.5, 201),
             (2, 6, 3.0, 0.25, 801), (3, 8, 2.0, 0.25, 201)]
    for (p, n, λ, σ2, N) in cases
        X = randn(p, n)
        y = X' * randn(p) + sqrt(σ2) * randn(n)
        M = sparse_lm(X, y, λ, σ2)
        G = ep(M; qr=QuadRule(40), maxcycles=200, tol=1e-10)
        ex = exact_sparse_lm(X, y, λ, σ2; N=N)
        @test maximum(abs.(mean(G) .- ex.μ)) < 2e-2
        @test maximum(abs.(cov(G) .- ex.Σ)) < 2e-2
        @test abs(log_ml(G) - ex.log_ml) < 5e-2
    end

    # λ → 0: the prior becomes flat, the posterior is Gaussian and EP is exact
    Random.seed!(7)
    p, n = 2, 6
    X = randn(p, n)
    y = X' * randn(p) + randn(n)
    G = ep(sparse_lm(X, y, 1e-6, 1.0); qr=QuadRule(20), maxcycles=100, tol=1e-12)
    Q0 = X*X' + 1e-8*Matrix{Float64}(I, p, p)
    Σg = inv(Q0)
    @test maximum(abs.(mean(G) .- Σg*(X*y))) < 1e-6
    @test maximum(abs.(cov(G) .- Σg)) < 1e-6
end
