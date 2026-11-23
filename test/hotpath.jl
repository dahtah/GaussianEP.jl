# Fused rank-1 hot path (perf_gap.md §7): the sequential fit! loop now uses
# _site_update_apply_1d! for rank-1 sites at power=1 — results must be
# identical to the generic site_update/update_site! path, and the crit
# keyword must select the convergence criterion.

using Random

@testset "fused rank-1 hot path matches generic path" begin
    rng = MersenneTwister(7)
    p, n = 6, 60
    X = randn(rng, p, n)
    y = rand(rng, n) .> 0.5
    M = glm_model(X, y, Logit(), 0.5)
    G1 = ep(M; qr=QuadRule(20), tol=1e-10, maxcycles=100)          # hot path
    # force the generic path via power != 1 machinery? no — compare against
    # parallel=false generic by calling site_update/update_site! manually:
    G2 = EPApprox(M; qr=QuadRule(20))
    GaussianEP.fit!(G2; qr=QuadRule(20), tol=1e-10, maxcycles=100, power=1.0 + 0.0)  # same path
    @test mean(G1) ≈ mean(G2) atol=1e-12
    @test log_ml(G1) ≈ log_ml(G2) atol=1e-10
    # generic (non-fused) path: rank-1 but power != 1 goes through site_update;
    # compare power=1 fused vs a manual sequential loop with site_update
    G3 = EPApprox(M; qr=QuadRule(20))
    for c in 1:100
        δ = 0.0
        for i in 1:GaussianEP.nsites(G3)
            dQ, dr, lz = GaussianEP.site_update(G3, i, QuadRule(20))
            δ = max(δ, maximum(abs.(dQ .- G3.Qi[i])), maximum(abs.(dr .- G3.ri[i])))
            GaussianEP.update_site!(G3, i, dQ, dr, lz, 1.0, 1.0)
        end
        GaussianEP.sync!(G3.state)
        δ < 1e-10 && break
    end
    @test mean(G1) ≈ mean(G3) atol=1e-8
    @test log_ml(G1) ≈ log_ml(G3) atol=1e-8
end

@testset "crit keyword" begin
    rng = MersenneTwister(11)
    p, n = 5, 50
    X = randn(rng, p, n)
    y = rand(rng, n) .> 0.5
    M = glm_model(X, y, Logit(), 1.0)
    Gs = ep(M; qr=QuadRule(20), tol=1e-9, crit=:strict)
    Gm = ep(M; qr=QuadRule(20), tol=1e-9, crit=:mean)
    Gx = ep(M; qr=QuadRule(20), tol=1e-9, crit=:max)
    # all criteria converge to the same fixed point here
    @test mean(Gs) ≈ mean(Gm) atol=1e-6
    @test mean(Gs) ≈ mean(Gx) atol=1e-6
    @test log_ml(Gs) ≈ log_ml(Gm) atol=1e-6
    # unknown crit falls back to :strict
    Gb = ep(M; qr=QuadRule(20), tol=1e-9, crit=:bogus)
    @test mean(Gb) ≈ mean(Gs) atol=1e-12
end

@testset "logZ0 cached" begin
    rng = MersenneTwister(3)
    X = randn(rng, 4, 30)
    y = rand(rng, 30) .> 0.5
    M = glm_model(X, y, Logit(), 0.1)
    G = ep(M; qr=QuadRule(20))
    @test G.logZ0 ≈ GaussianEP.safe_log_partition(Matrix(M.Q0), M.r0)
    @test log_ml(G) ≈ GaussianEP.logZ(G.state) - GaussianEP.safe_log_partition(Matrix(M.Q0), M.r0) + sum(G.logz)
end
