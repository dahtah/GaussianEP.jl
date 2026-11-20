# Analytic tilted moments: ProbitLik and HardSphere vs numerical reference.
using Random, Distributions

# fine-grid trapezoid reference for E[f], E[f η], E[f η²] under η ~ N(m, s²)
function grid_moments(logf, m, s2; L=12.0, N=200001)
    s = sqrt(s2)
    xs = range(m - L * s, m + L * s, N)
    ws = exp.(-0.5 .* ((xs .- m) ./ s) .^ 2) ./ (s * sqrt(2π))
    fs = exp.(logf.(xs))
    tw = [0.5; ones(N - 2); 0.5] .* step(xs)   # trapezoid weights
    z = sum(tw .* ws .* fs)
    μ = sum(tw .* ws .* fs .* xs) / z
    m2 = sum(tw .* ws .* fs .* xs .^ 2) / z
    (z=z, μ=μ, σ2=m2 - μ^2)
end

@testset "analytic probit moments" begin
    for y in (true, false), (m, s2) in ((0.3, 0.8), (-1.2, 2.5), (2.0, 0.1), (0.0, 3.0))
        lik = ProbitLik()
        qr = QuadRule(40)
        za, lza, μa, Σa = GaussianEP.tilted_moments(lik, y, [m], [s2;;], qr)
        zr, μr, σ2r = grid_moments(η -> GaussianEP.log_dens(lik, η, y), m, s2)
        @test isapprox(za, zr, rtol=1e-8)
        @test isapprox(lza, log(zr), atol=1e-8)
        @test isapprox(μa[1], μr, atol=1e-8)
        @test isapprox(Σa[1, 1], σ2r, atol=1e-8)
        # fractional power falls back to quadrature and stays consistent
        zf, _, μf, Σf = GaussianEP.tilted_moments(lik, y, [m], [s2;;], QuadRule(200), power=0.5)
        zr2, μr2, σ2r2 = grid_moments(η -> 0.5 * GaussianEP.log_dens(lik, η, y), m, s2)
        @test isapprox(zf, zr2, rtol=1e-6)
        @test isapprox(μf[1], μr2, atol=1e-5)
    end
end

# reference for a hard constraint: trapezoid over [-R, R] only (smooth
# integrand, no kink inside the integration range)
function grid_moments_sphere(R, m, s2; N=200001)
    s = sqrt(s2)
    xs = range(-R, R, N)
    ws = exp.(-0.5 .* ((xs .- m) ./ s) .^ 2) ./ (s * sqrt(2π))
    tw = [0.5; ones(N - 2); 0.5] .* step(xs)   # trapezoid weights
    z = sum(tw .* ws)
    μ = sum(tw .* ws .* xs) / z
    m2 = sum(tw .* ws .* xs .^ 2) / z
    (z=z, μ=μ, σ2=m2 - μ^2)
end

@testset "analytic hard-sphere moments" begin
    for R in (0.5, 1.0, 2.0), (m, s2) in ((0.3, 0.8), (-1.2, 2.5), (2.0, 0.1), (0.0, 3.0))
        lik = HardSphere(R)
        qr = QuadRule(40)
        za, lza, μa, Σa = GaussianEP.tilted_moments(lik, nothing, [m], [s2;;], qr)
        zr, μr, σ2r = grid_moments_sphere(R, m, s2)
        @test isapprox(za, zr, rtol=1e-8)
        @test isapprox(lza, log(zr), atol=1e-8)
        @test isapprox(μa[1], μr, atol=1e-8)
        @test isapprox(Σa[1, 1], σ2r, atol=1e-8)
        # cavity entirely outside the sphere: no-information fallback
        z0, lz0, μ0, Σ0 = GaussianEP.tilted_moments(lik, nothing, [10.0], [0.01;;], qr)
        @test z0 == 0.0 && lz0 == -Inf && μ0 == [10.0] && Σ0[1, 1] == 0.01
    end
end

@testset "probit GLM vs brute force (d=1)" begin
    Random.seed!(11)
    n = 1
    X = randn(1, n)
    y = vec(X) .> 0
    τ = 0.1
    G = ep(glm_model(X, y, ProbitLik(), τ); qr=QuadRule(20), maxcycles=50, tol=1e-10)
    # brute-force posterior on a fine grid
    xs = range(-8 / sqrt(τ), 8 / sqrt(τ), 200001)
    lp = [GaussianEP.log_dens(ProbitLik(), X[1] * x, y[1]) - 0.5 * τ * x^2 for x in xs]
    w = exp.(lp)   # no max-subtraction: z must be the unnormalised integral
    tw = [0.5; ones(length(xs) - 2); 0.5] .* step(xs)   # trapezoid weights
    z = sum(tw .* w)
    μ = sum(tw .* w .* xs) / z
    σ2 = sum(tw .* w .* (xs .- μ) .^ 2) / z
    @test abs(mean(G)[1] - μ) < 1e-6
    @test abs(cov(G)[1, 1] - σ2) < 1e-6
    # log_ml is log Z - log Z0; the grid z is the unnormalised Z, Z0 = sqrt(2π/τ)
    @test abs(log_ml(G) - (log(z) - 0.5 * log(2π / τ))) < 1e-5
end
