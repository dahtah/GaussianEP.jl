# Log-Gaussian Cox process on the unit square (grid approximation):
# points of a Poisson process with intensity λ(s) = exp(x(s)), x a 2D GMRF;
# cell counts y_j ~ Poisson(a·exp(x_j)), a = 1/L². Fully sparse model.
using Distributions, Random, Statistics, SpecialFunctions

# known intensity and Poisson-process simulation on a fine grid
_lgcp_λ(s1, s2) = exp(6.0 + 0.8*s1 - 0.5*s2 + 0.8*sin(2π*s1)*cos(2π*s2))
function _lgcp_simulate(Lfine, λfun; rng)
    a = 1.0 / Lfine^2
    pts = Tuple{Float64,Float64}[]
    for i in 1:Lfine, j in 1:Lfine
        s1 = (i - 0.5) / Lfine; s2 = (j - 0.5) / Lfine
        for _ in 1:rand(rng, Poisson(a * λfun(s1, s2)))
            push!(pts, (s1 + (rand(rng) - 0.5) / Lfine, s2 + (rand(rng) - 0.5) / Lfine))
        end
    end
    return pts
end
_lgcp_bin(pts, L) = begin
    y = zeros(Int, L, L)
    for (s1, s2) in pts
        y[clamp(ceil(Int, s1 * L), 1, L), clamp(ceil(Int, s2 * L), 1, L)] += 1
    end
    y
end

@testset "LGCP model structure" begin
    L = 6; p = L * L
    y = zeros(Int, L, L); y[2, 3] = 5; y[5, 1] = 2
    M = lgcp_model(y; ρ=4.0, τ=0.01)
    @test M.Q0 isa SparseMatrixCSC
    @test nnz(M.Q0) == 5p - 4 * (L - 1) * 2 + 0 || nnz(M.Q0) > 0   # 2D Laplacian + ridge pattern
    @test length(M.sites) == p
    @test all(s -> size(s.A, 1) == 1 && s.A isa SparseMatrixCSC, M.sites)
    @test M.sites[1].lik isa PoisLikA
    @test M.sites[1].lik.a == 1.0 / p
    # log_dens of the model at x: prior + Poisson counts (up to a constant)
    x = randn(p)
    ld = logdensity(M, x)
    @test isfinite(ld)
    ld_ref = -0.5 * dot(x, Matrix(M.Q0) * x) +
             sum(y[i] * (x[i] + log(1.0 / p)) - (1.0 / p) * exp(x[i]) - lgamma(1 + y[i]) for i in 1:p)
    @test ld ≈ ld_ref atol = 1e-10
end

@testset "LGCP: EP sparse ≡ dense ≡ parallel, quadrature, recovery" begin
    rng = MersenneTwister(2024)
    pts = _lgcp_simulate(120, _lgcp_λ; rng=rng)
    @test 300 < length(pts) < 800
    L = 10; p = L * L
    y = _lgcp_bin(pts, L)
    M = lgcp_model(y; ρ=4.0, τ=0.01)
    Gs = ep(M; backend=:sparse, qr=QuadRule(100), maxcycles=200, tol=1e-10, verbose=false)
    Gd = ep(M; backend=:dense,  qr=QuadRule(100), maxcycles=200, tol=1e-10, verbose=false)
    Gp = ep(M; backend=:sparse, qr=QuadRule(100), parallel=true, maxcycles=200, tol=1e-8, verbose=false)
    @test isfinite(log_ml(Gs))
    @test log_ml(Gs) ≈ log_ml(Gd) atol = 1e-7
    @test maximum(abs.(mean(Gs) .- mean(Gd))) < 1e-7
    @test log_ml(Gp) ≈ log_ml(Gs) atol = 1e-5
    # quadrature convergence: GH100 vs GH200 vs trapezoid rule
    G2 = ep(M; backend=:sparse, qr=QuadRule(200), maxcycles=200, tol=1e-10, verbose=false)
    Gt = ep(M; backend=:sparse, qr=trap_rule(12.0, 2401), maxcycles=200, tol=1e-10, verbose=false)
    @test log_ml(G2) ≈ log_ml(Gt) atol = 0.05
    @test log_ml(Gs) ≈ log_ml(Gt) atol = 0.5
    @test maximum(abs.(mean(G2) .- mean(Gt))) < 0.05
    # recovery of the true intensity (posterior mean λ̂ = exp(μ + σ²/2))
    μ = mean(Gs); sd = se(Gs)
    λhat = exp.(μ .+ 0.5 .* sd.^2)
    λt = [_lgcp_λ((i - 0.5) / L, (j - 0.5) / L) for j in 1:L for i in 1:L]  # column-major, matches vec(y)
    @test cor(log.(λhat), log.(λt)) > 0.7
    @test sum(λhat) / p ≈ sum(y) rtol = 0.5   # predicted total count a·Σλ̂ = Σλ̂/p vs observed Σy
end

# ---- SPDE Matérn prior (grid-consistent in L) ----
_pois_small(rng, λ) = begin
    L = exp(-λ); k = 0; p = 1.0
    while true
        p *= rand(rng); p > L || return k; k += 1
    end
end
_lgcp_counts_spde(rng, L, xfun) = begin
    y = zeros(Int, L, L); a = 1.0 / L^2
    for j in 1:L, i in 1:L
        y[i, j] = _pois_small(rng, a * exp(xfun((i - 0.5) / L, (j - 0.5) / L)))
    end
    y
end

@testset "SPDE Matérn precision: structure and grid consistency" begin
    for (L, c) in ((10, 1), (20, 1), (10, 2))
        m = max(4, L ÷ 8)
        Q, interior = GaussianEP.matern_precision_2d(L, m; c=c, κ=1.0, σ²=1.0)
        n = (L + 2m - 2)^2
        @test size(Q) == (n, n) && Q isa SparseMatrixCSC
        @test issymmetric(Q)
        @test isposdef(Symmetric(Matrix(Q)))          # SPD
        @test length(interior) == L * L
        @test all(1 .<= interior .<= n) && length(unique(interior)) == L * L
        # normalization: marginal prior variance at the domain center = σ²
        nf = L + 2m - 2
        ctr = (L÷2 + m - 2) + (L÷2 + m - 2) * nf + 1
        e = zeros(n); e[ctr] = 1.0
        @test sum(e .* (Q \ e)) ≈ 1.0 rtol = 1e-6
    end
    # grid consistency: marginal variance at a fixed relative location is stable in L
    function _margvar(L, c, κ, σ², rel)
        m = max(4, L ÷ 8)
        Q, _ = GaussianEP.matern_precision_2d(L, m; c=c, κ=κ, σ²=σ²)
        nf = L + 2m - 2
        k = round(Int, rel * L) + m - 2 + (round(Int, rel * L) + m - 2) * nf + 1
        e = zeros(size(Q, 1)); e[k] = 1.0
        sum(e .* (Q \ e))
    end
    v10 = _margvar(10, 1, 1.0, 1.0, 0.25)
    v20 = _margvar(20, 1, 1.0, 1.0, 0.25)
    v40 = _margvar(40, 1, 1.0, 1.0, 0.25)
    @test abs(v10 - v20) / v20 < 0.3 && abs(v40 - v20) / v20 < 0.3
end

@testset "LGCP SPDE: structure, backend equivalence, recovery" begin
    rng = MersenneTwister(31)
    L = 16
    xfun(s1, s2) = 4.5 + 1.2 * sin(2π * s1) * cos(2π * s2)
    y = _lgcp_counts_spde(rng, L, xfun)
    M = lgcp_model_spde(y; c=1, κ=6.0, σ²=1.0)
    @test M.Q0 isa SparseMatrixCSC
    @test size(M.Q0, 1) == (L + 2 * max(4, L ÷ 8) - 2)^2   # extended grid
    @test length(M.sites) == L * L
    @test all(s -> nnz(s.A) == 1 && s.lik isa PoisLikA && s.lik.a == 1.0 / L^2, M.sites)
    # logdensity of the model at x (prior on extended grid + Poisson counts)
    x = randn(rng, size(M.Q0, 1))
    ld = logdensity(M, x)
    _, interior = GaussianEP.matern_precision_2d(L, max(4, L ÷ 8); c=1, κ=6.0, σ²=1.0)
    ld_ref = -0.5 * dot(x, Matrix(M.Q0) * x) +
             sum(y[i] * (x[interior[i]] + log(1.0 / L^2)) - (1.0 / L^2) * exp(x[interior[i]]) - lgamma(1 + y[i]) for i in 1:L^2)
    @test ld ≈ ld_ref atol = 1e-10
    # backend equivalence: sparse ≡ dense ≡ minibatch
    qr = QuadRule(80)
    Gs = ep(M; backend=:sparse, qr=qr, maxcycles=100, tol=1e-8, verbose=false)
    Gd = ep(M; backend=:dense,  qr=qr, maxcycles=100, tol=1e-8, verbose=false)
    Gb = ep(M; backend=:sparse, qr=qr, maxcycles=100, tol=1e-8, batch=16, verbose=false)
    @test isfinite(log_ml(Gs))
    @test log_ml(Gs) ≈ log_ml(Gd) atol = 1e-7
    @test log_ml(Gb) ≈ log_ml(Gs) atol = 1e-7
    @test maximum(abs.(mean(Gs) .- mean(Gd))) < 1e-7
    # recovery of the true log-intensity on the original cells
    truth = [xfun((i - 0.5) / L, (j - 0.5) / L) for j in 1:L for i in 1:L]
    @test cor(mean(Gs)[interior], truth) > 0.4
end
