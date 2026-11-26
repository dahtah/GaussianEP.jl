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

# --- Smolyak sparse grids for rank 2-3 (status.md open question 7 remaining
# part; TODO "Sparse grids (Smolyak) for rank 2-3 quadrature"). smolyak_rule
# returns a QuadRule, so it plugs into tilted_moments / ep(...; qr=...) with
# no changes at call sites (the rank-d path in moments.jl uses qr.grid).

@testset "smolyak_rule basics" begin
    # d=1 reduces to the level-`level` 1-D Gauss-Hermite rule
    qr1 = GaussianEP.smolyak_rule(4, 1)
    @test qr1 isa QuadRule
    @test qr1.grid === nothing
    @test length(qr1.x) == 2^3 + 1
    @test sum(qr1.w) ≈ 1 atol=1e-14
    @test qr1.logw ≈ log.(qr1.w)
    # d=2,3: sparse grid stored in qr.grid, signed weights, sum(w)=1
    for (q, d) in ((4,2), (5,2), (4,3), (5,3))
        qr = GaussianEP.smolyak_rule(q, d)
        @test qr isa QuadRule
        @test qr.grid isa Matrix{Float64}
        @test size(qr.grid) == (d, length(qr.w))
        @test qr.logw ≈ log.(abs.(qr.w))
        @test sum(qr.w) ≈ 1 atol=1e-12
        @test any(<(0), qr.w)          # Smolyak combination has signed weights
    end
    # trap base: nested dyadic trapezoid rules on [-L,L], even sparser
    q, d = 5, 2
    qrt = GaussianEP.smolyak_rule(q, d, base=:trap)
    @test length(qrt.w) < length(GaussianEP.smolyak_rule(q, d).w)
    @test sum(qrt.w) ≈ 1 atol=1e-12
    # argument checks
    @test_throws ArgumentError GaussianEP.smolyak_rule(2, 3)            # level < d
    @test_throws ArgumentError GaussianEP.smolyak_rule(3, 2, base=:bogus)
    @test_throws ArgumentError GaussianEP.smolyak_rule(0, 2)
end

@testset "smolyak exactness on polynomials of degree 2*level-1" begin
    nm(k) = isodd(k) ? 0.0 : prod(1:2:k-1)     # E[Z^k] for Z ~ N(0,1)
    for (q, d) in ((4,2), (5,2), (5,3), (6,3))
        qr = GaussianEP.smolyak_rule(q, d)
        Z, w = qr.grid, qr.w
        deg = 2q - 1
        for α in Iterators.product(ntuple(_ -> 0:deg, d)...)
            sum(α) > deg && continue
            val = sum(w[n] * prod(Z[i,n]^α[i] for i in 1:d) for n in 1:size(Z,2))
            @test val ≈ prod(nm(α[i]) for i in 1:d) atol=1e-8
        end
    end
end

@testset "smolyak node count strictly below the full tensor grid" begin
    for (q, d) in ((4,2), (5,2), (6,2), (4,3), (5,3), (6,3))
        qr = GaussianEP.smolyak_rule(q, d)
        k = 2^(q-1) + 1                       # 1-D nodes of the top-level rule
        @test length(qr.w) < k^d              # full tensor grid k^d at level q
    end
end

@testset "smolyak tilted moments vs full tensor grid (rank 2 and 3)" begin
    lik = ConfineLik(η -> 0.5*sum(abs2, η) + 0.1*sum(abs2, η)^2)
    # rank 2: reference = full tensor GH(60) (3600 nodes)
    m = [0.2, -0.4]; S = [0.8 0.1; 0.1 0.5]
    ref = GaussianEP.tilted_moments(lik, nothing, m, S, QuadRule(60))
    qs = GaussianEP.smolyak_rule(6, 2)                    # 203 nodes
    ts = GaussianEP.tilted_moments(lik, nothing, m, S, qs)
    err = max(abs(ts.lz - ref.lz), maximum(abs.(ts.μ .- ref.μ)), maximum(abs.(Matrix(ts.Σ) .- Matrix(ref.Σ))))
    @test err < 1e-2
    @test length(qs.w) < 33^2                              # full tensor at level 6
    # rank 3: reference = full tensor GH(30) (27000 nodes)
    m3 = [0.3, -0.2, 0.1]; S3 = [1.0 0.2 0.1; 0.2 0.8 0.15; 0.1 0.15 0.6]
    ref3 = GaussianEP.tilted_moments(lik, nothing, m3, S3, QuadRule(30))
    qs3 = GaussianEP.smolyak_rule(7, 3)                   # 1113 nodes
    ts3 = GaussianEP.tilted_moments(lik, nothing, m3, S3, qs3)
    err3 = max(abs(ts3.lz - ref3.lz), maximum(abs.(ts3.μ .- ref3.μ)), maximum(abs.(Matrix(ts3.Σ) .- Matrix(ref3.Σ))))
    @test err3 < 3e-2
    @test length(qs3.w) < 129^3                            # full tensor at level 7
    # fractional power (fractional EP) also goes through the sparse path
    tp = GaussianEP.tilted_moments(lik, nothing, m, S, qs, power=0.5)
    tpr = GaussianEP.tilted_moments(lik, nothing, m, S, QuadRule(60), power=0.5)
    @test abs(tp.lz - tpr.lz) < 1e-2
    # a d>=2 sparse rule is rejected on rank-1 sites ...
    @test_throws ArgumentError GaussianEP.tilted_moments(Logit(), true, [0.1], [0.5;;], GaussianEP.smolyak_rule(4, 2))
    # ... but smolyak_rule(level, 1) works on rank-1 sites (it is the 1-D rule)
    t1 = GaussianEP.tilted_moments(Logit(), true, [0.1], [0.5;;], GaussianEP.smolyak_rule(5, 1))
    t1r = GaussianEP.tilted_moments(Logit(), true, [0.1], [0.5;;], QuadRule(17))
    @test t1.lz ≈ t1r.lz atol=1e-12
    @test t1.μ ≈ t1r.μ atol=1e-12
end

@testset "ep with a Smolyak rule on a small rank-2 model" begin
    # two rank-2 sites with a smooth (asymmetric) quartic confinement
    c = [0.5, -0.3]
    lik = ConfineLik(η -> 0.5*sum(abs2, η .- c) + 0.1*sum(abs2, η)^2)
    m = 4
    A2 = [1.0 0.0 0.3 0.0; 0.0 1.0 0.0 0.2]
    M = EPModel(Matrix(I, m, m), [Site(A2, 0.0, lik), Site([0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0], 0.0, lik)])
    qs = GaussianEP.smolyak_rule(8, 2)                     # 992 nodes
    Gs = ep(M, qr=qs, tol=1e-11, maxcycles=200)
    Gf = ep(M, qr=QuadRule(40), tol=1e-11, maxcycles=200)  # 1600 nodes, converged
    @test isfinite(log_ml(Gs))
    @test abs(log_ml(Gs) - log_ml(Gf)) < 5e-3              # logml agreement
    @test maximum(abs.(mean(Gs) - mean(Gf))) < 1e-3
    @test maximum(abs.(cov(Gs) - cov(Gf))) < 5e-3
    # the sparse grid is much smaller than the full tensor grid at level 8
    @test length(qs.w) < (2^7 + 1)^2
end

# --- Smolyak sparse grids on ACTUAL models (status.md TODO "Test the Smolyak
# sparse grid on actual models"; numbers/findings in smolyak_models.md) ---
using Random

@testset "smolyak on a rank-3 logistic model (actual model)" begin
    # two rank-3 sites f(η) = σ(c'η) on m = 6, log_dens = loglogistic(c'η)
    c1 = [0.8, -0.5, 0.3]; c2 = [-0.4, 0.9, 0.6]
    lik1 = ConfineLik(η -> -GaussianEP.loglogistic(dot(c1, η)))
    lik2 = ConfineLik(η -> -GaussianEP.loglogistic(dot(c2, η)))
    A1 = [1.0 0 0 0.3 0 0; 0 1.0 0 0 0.3 0; 0 0 1.0 0 0 0.3]
    A2 = [0.0 0 0 1.0 0 0; 0 0 0 0 1.0 0; 0 0 0 0 0 1.0]
    ML = EPModel(Matrix(I, 6, 6), [Site(A1, 1.0, lik1), Site(A2, 1.0, lik2)])
    Gref = ep(ML, qr=QuadRule(40), tol=1e-10, maxcycles=400, damp=0.7)  # 64000 nodes
    qs = GaussianEP.smolyak_rule(6, 3)                                   # 417 nodes
    Gs = ep(ML, qr=qs, tol=1e-10, maxcycles=400, damp=0.7)
    @test isfinite(log_ml(Gs))
    @test abs(log_ml(Gs) - log_ml(Gref)) < 1e-3
    @test maximum(abs.(mean(Gs) - mean(Gref))) < 1e-3
    @test maximum(abs.(cov(Gs) - cov(Gref))) < 1e-2
    # far fewer nodes than the full tensor grid at the same level (33^3)
    @test length(qs.w) < (2^5 + 1)^3
end

@testset "smolyak d=1 on a 1D GMRF logistic model" begin
    rng = MersenneTwister(42)
    p = 60
    y = rand(rng, p) .> 0.5
    M1 = gmrf_1d(y, Logit(); ρ=1.0)          # rank-1 sites
    Gref = ep(M1, qr=QuadRule(40), tol=1e-10, maxcycles=400)
    qs = GaussianEP.smolyak_rule(5, 1)       # d=1 reduces to GH with 17 nodes
    @test qs.grid === nothing
    @test length(qs.w) == 17
    Gs = ep(M1, qr=qs, tol=1e-10, maxcycles=400)
    @test abs(log_ml(Gs) - log_ml(Gref)) < 1e-9
    @test maximum(abs.(mean(Gs) - mean(Gref))) < 1e-9
end

@testset "smolyak signed-weight fallback (z<=0 -> cavity moments)" begin
    # sharp ring factor: the signed Smolyak sum is non-positive on some
    # cavities; _tilted_quad then falls back to the cavity moments
    lik = ConfineLik(η -> 10 * (sum(abs2, η) - 1)^2)
    qr = GaussianEP.smolyak_rule(4, 2)
    hits = Tuple{Vector{Float64}, Float64}[]
    for m1 in -2.0:0.5:2.0, m2 in -2.0:0.5:2.0, s2 in (0.05, 0.2, 0.5, 1.0, 2.0)
        t = GaussianEP.tilted_moments(lik, 0.0, [m1, m2], s2 .* Matrix(I, 2, 2), qr)
        t.lz == -Inf && push!(hits, ([m1, m2], s2))
    end
    @test length(hits) > 0                    # fallback does trigger on this factor
    m, s2 = hits[1]
    t = GaussianEP.tilted_moments(lik, 0.0, m, s2 .* Matrix(I, 2, 2), qr)
    @test t.lz == -Inf
    @test t.μ == m                            # cavity moments returned
    @test Matrix(t.Σ) == s2 .* Matrix(I, 2, 2)
end
