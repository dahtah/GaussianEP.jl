# Quadrature rules beyond Gauss–Hermite: trapezoid rule, adaptive refinement,
# and a GH-vs-trapezoid disagreement diagnostic (status.md: robust quadrature
# for non-differentiable likelihoods; open question 7, Trefethen).

# fine-grid trapezoid reference for (z, μ, σ²) of f(η) under η ~ N(m, s²)
function qref_moments(logf, m, s2; L=14.0, N=400001)
    s = sqrt(s2)
    xs = range(m - L * s, m + L * s, N)
    ws = exp.(-0.5 .* ((xs .- m) ./ s) .^ 2) ./ (s * sqrt(2π))
    fs = exp.(logf.(xs))
    tw = [0.5; ones(N - 2); 0.5] .* step(xs)
    z = sum(tw .* ws .* fs)
    μ = sum(tw .* ws .* fs .* xs) / z
    m2 = sum(tw .* ws .* fs .* xs .^ 2) / z
    (z=z, μ=μ, σ2=m2 - μ^2)
end

@testset "trap_rule basics" begin
    qr = trap_rule(8.0, 401)
    @test qr isa QuadRule
    @test length(qr.x) == 401
    @test sum(qr.w) ≈ 1 atol=1e-14
    @test qr.logw ≈ log.(qr.w)
    @test qr.x[1] == -8.0 && qr.x[end] == 8.0
    @test all(diff(qr.x) .≈ qr.x[2] - qr.x[1])          # uniform nodes
    # spectral accuracy on smooth Gaussian moments: E[1], E[Z], E[Z²], E[Z⁴]
    @test dot(qr.w, ones(401)) ≈ 1 atol=1e-14
    @test abs(dot(qr.w, qr.x)) < 1e-12
    @test dot(qr.w, qr.x .^ 2) ≈ 1 atol=1e-10
    @test dot(qr.w, qr.x .^ 4) ≈ 3 atol=1e-8
    @test_throws ArgumentError trap_rule(8.0, 2)
end

@testset "trap vs Gauss–Hermite on a kinked factor (LaplaceLik)" begin
    lik = LaplaceLik(3.0)
    y = 0.0
    m, s2 = 0.2, 1.5
    ref = qref_moments(η -> GaussianEP.log_dens(lik, η, y), m, s2)
    gh = GaussianEP._tilted_moments_1d(lik, y, m, s2, QuadRule(20), false, 1.0)
    tr = GaussianEP._tilted_moments_1d(lik, y, m, s2, trap_rule(8.0, 1601), false, 1.0)
    err_gh = max(abs(gh.lz - log(ref.z)), abs(gh.μ[1] - ref.μ), abs(gh.Σ[1,1] - ref.σ2))
    err_tr = max(abs(tr.lz - log(ref.z)), abs(tr.μ[1] - ref.μ), abs(tr.Σ[1,1] - ref.σ2))
    @test err_tr < 1e-4                 # O(h²) at the kink, h = 0.01
    @test err_tr < err_gh / 10          # trapezoid clearly better at the kink
end

@testset "trap vs Gauss–Hermite on a sharp bounded factor (steep link)" begin
    lik = GaussianEP.BernLik(η -> 1 / (1 + exp(-30η)))   # near-step invlink
    y = 1.0
    m, s2 = -0.1, 0.7
    ref = qref_moments(η -> GaussianEP.log_dens(lik, η, y), m, s2)
    gh = GaussianEP._tilted_moments_1d(lik, y, m, s2, QuadRule(20), false, 1.0)
    tr = GaussianEP._tilted_moments_1d(lik, y, m, s2, trap_rule(8.0, 1601), false, 1.0)
    err_gh = max(abs(gh.lz - log(ref.z)), abs(gh.μ[1] - ref.μ), abs(gh.Σ[1,1] - ref.σ2))
    err_tr = max(abs(tr.lz - log(ref.z)), abs(tr.μ[1] - ref.μ), abs(tr.Σ[1,1] - ref.σ2))
    @test err_tr < 1e-6
    @test err_tr < err_gh
end

@testset "rank-2 tensor product with trap_rule" begin
    # soft ring constraint in 2-D: V(η) = 10*(|η|²-1)²
    lik = ConfineLik(η -> 10 * (sum(abs2, η) - 1)^2)
    y = 0.0
    m = [0.3, -0.2]; S = [1.2 0.3; 0.3 0.9]
    # reference: fine 2-D grid
    L, N = 6.0, 301
    xs = range(-L, L, N); h = step(xs)
    z = 0.0; μ = zeros(2); m2 = zeros(2, 2)
    C = cholesky(Symmetric(S)); ld = 2 * sum(log, diag(C.L))
    for x1 in xs, x2 in xs
        η = [x1, x2]
        lw = GaussianEP.log_dens(lik, η, y) - 0.5 * ((η - m)' * (S \ (η - m))) - 0.5 * ld - log(2π)
        w = exp(lw) * h^2
        z += w; μ .+= w .* η; m2 .+= w .* (η * η')
    end
    μ ./= z; Σ = m2 ./ z .- μ * μ'
    tr = GaussianEP.tilted_moments(lik, y, m, S, trap_rule(6.0, 201))
    gh = GaussianEP.tilted_moments(lik, y, m, S, QuadRule(30))
    @test abs(tr.μ[1] - μ[1]) < 1e-6 && abs(tr.μ[2] - μ[2]) < 1e-6
    @test abs(tr.Σ[1,1] - Σ[1,1]) < 1e-5 && abs(tr.Σ[2,2] - Σ[2,2]) < 1e-5
    @test abs(tr.Σ[1,2] - Σ[1,2]) < 1e-5
    # GH(30) is visibly off on this non-polynomial factor
    @test maximum(abs.(gh.μ .- μ)) > maximum(abs.(tr.μ .- μ))
end

@testset "moment_disagreement diagnostic" begin
    qr = QuadRule(20)
    # kinked factor: GH and trapezoid disagree noticeably
    d_kink = moment_disagreement(LaplaceLik(3.0), 0.0, 0.2, 1.5, qr)
    @test d_kink > 1e-4
    # Gaussian factor: both rules are (near-)exact, disagreement ~ 0
    d_gauss = moment_disagreement(GaussianEP.GLik(), 1.0, 0.2, 1.5, qr)
    @test d_gauss < 1e-5                # small vs the kinked case above
end

@testset "adaptive_trap_moments_1d" begin
    lik = LaplaceLik(3.0)
    y = 0.0
    m, s2 = 0.2, 1.5
    ref = qref_moments(η -> GaussianEP.log_dens(lik, η, y), m, s2)
    tm = adaptive_trap_moments_1d(lik, y, m, s2)
    # kink limits accuracy to O(h²); refinement to Nmax gives ~1e-6
    @test abs(tm.lz - log(ref.z)) < 1e-5
    @test abs(tm.μ[1] - ref.μ) < 1e-5
    @test abs(tm.Σ[1,1] - ref.σ2) < 1e-5
    # smooth factor: converges immediately and matches the analytic GLik rule
    g = GaussianEP.GLik()
    tm2 = adaptive_trap_moments_1d(g, 1.0, m, s2)
    an = GaussianEP.tilted_moments(g, 1.0, [m], [s2;;], QuadRule(20))
    @test abs(tm2.lz - an.lz) < 1e-10
    @test abs(tm2.μ[1] - an.μ[1]) < 1e-10
end

@testset "ep with a trapezoid rule" begin
    # product model with Laplace (kinked) sites: trap rule plugs into ep.
    # Exact per-coordinate reference: p(x) ∝ exp(-τ x²/2) (λ/2) e^{-λ|x|}.
    m = 4
    τ = 1.0
    λ = 2.0
    lik = LaplaceLik(λ)
    M = EPModel(τ*Matrix(I,m,m), [Site(_row(i,m), 0.0, lik) for i in 1:m])
    Gt = ep(M, qr=trap_rule(8.0, 6401))
    Gg = ep(M, qr=QuadRule(60))
    @test isfinite(log_ml(Gt)) && isfinite(log_ml(Gg))
    lml_ref = 0.0
    for i in 1:m
        # qref integrates against N(0, 1/τ), which IS the base measure here
        ref = qref_moments(η -> GaussianEP.log_dens(lik, η, 0.0), 0.0, 1/τ)
        @test abs(mean(Gt)[i] - ref.μ) < 1e-6
        @test abs(cov(Gt)[i,i] - ref.σ2) < 1e-5   # kink limits to O(h²)
        lml_ref += log(ref.z)   # log z_i - log Z0 per coordinate cancels the ½log(2π/τ)
    end
    @test abs(log_ml(Gt) - lml_ref) < 1e-5
    # GH(60) under-resolves the kink: visibly worse log_ml than the trap rule
    @test abs(log_ml(Gt) - lml_ref) < abs(log_ml(Gg) - lml_ref) / 10
end

@testset "robust=true auto-switches to trapezoid on flagged cavities" begin
    # Laplace (kinked) sites: GH(20) under-resolves, robust mode should fix it
    m = 4
    τ = 1.0
    λ = 2.0
    lik = LaplaceLik(λ)
    M = EPModel(τ*Matrix(I,m,m), [Site(_row(i,m), 0.0, lik) for i in 1:m])
    Gg = ep(M, qr=QuadRule(20))                    # plain GH
    Gr = ep(M, qr=QuadRule(20), robust=true)       # auto-switch when flagged
    lml_ref = 0.0
    for i in 1:m
        ref = qref_moments(η -> GaussianEP.log_dens(lik, η, 0.0), 0.0, 1/τ)
        @test abs(mean(Gr)[i] - ref.μ) < 1e-5
        @test abs(cov(Gr)[i,i] - ref.σ2) < 1e-4
        lml_ref += log(ref.z)
    end
    @test abs(log_ml(Gr) - lml_ref) < 1e-4
    # robust mode must be at least as accurate as plain GH here
    @test abs(log_ml(Gr) - lml_ref) <= abs(log_ml(Gg) - lml_ref) + 1e-12
    # smooth sites: robust mode never triggers, results identical
    Mg = EPModel(τ*Matrix(I,m,m), [Site(_row(i,m), 1.0, GaussianEP.GLik()) for i in 1:m])
    Ga = ep(Mg, qr=QuadRule(20))
    Gb = ep(Mg, qr=QuadRule(20), robust=true)
    @test mean(Ga) ≈ mean(Gb) atol=1e-12
    @test log_ml(Ga) ≈ log_ml(Gb) atol=1e-12
end
