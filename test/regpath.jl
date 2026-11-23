using Random, Distributions
# Regularisation path / autotune over the prior precision τ on the new core.
@testset "glm_regpath / glm_autotune" begin
    Random.seed!(3)
    p, n = 4, 60
    X = randn(p, n)
    y = rand(n) .< 0.5

    path = glm_regpath(X, y, Logit(), 0.01, 10.0; nsteps=6, maxcycles=50, tol=1e-8)
    @test length(path.τs) == 6
    @test issorted(path.τs, rev=true)          # path runs from high τ (τ1) down to τ0
    @test path.τs[1] ≈ 10.0 && path.τs[end] ≈ 0.01
    @test all(isfinite, path.logml)
    @test size(path.μs) == (p, 6)
    @test size(path.sds) == (p, 6)
    @test all(path.sdavg .> 0)
    @test length(path.μavg) == p

    # autotune: picks the grid point with max logml, refit matches ep at τ_opt
    at = glm_autotune(X, y, Logit(), 0.01, 10.0; nsteps=6, maxcycles=50, tol=1e-8)
    @test at.τ_opt == path.τs[argmax(path.logml)]
    @test isfinite(log_ml(at.G))
    G = ep(glm_model(X, y, Logit(), at.τ_opt); maxcycles=50, tol=1e-8)
    @test maximum(abs.(mean(G) .- mean(at.G))) < 1e-6
    @test maximum(abs.(cov(G) .- cov(at.G))) < 1e-6

    # warm-started path agrees with the legacy regularisation path (loose tol)
    S = GaussianEP.GLMSites(X, y, Logit(), QuadRule(20))
    τs2, μs2, lz2 = GaussianEP._regpath_tau(S, 0.01, 10.0; nsteps=6)
    @test τs2 ≈ reverse(path.τs)               # legacy path is ascending, ours descending
    @test abs(maximum(path.logml) - maximum(lz2)) < 0.5

    # Poisson likelihood path also runs and stays finite
    yp = Float64.(rand.(Poisson.(exp.(X' * randn(p)))))
    pathp = glm_regpath(X, yp, PoisLik(), 0.1, 10.0; nsteps=4, maxcycles=30, tol=1e-6)
    @test all(isfinite, pathp.logml)
    @test all(isfinite, pathp.μs)

    # pretty-printing: show gives a compact, meaningful summary (no raw arrays)
    s = sprint(show, path)
    @test occursin("tau", s)
    @test occursin("logml", s)
    @test length(s) < 1000                      # compact, not an array dump
    @test sprint(show, "text/plain", path) == s
    sa = sprint(show, at)
    @test occursin("tau_opt", sa)
    @test occursin("logml", sa)
    @test length(sa) < 1000
end
