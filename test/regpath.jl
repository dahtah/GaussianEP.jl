using Random, Distributions, Logging
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

    # set_tau!: in-place τ change on a unit-base model matches a fresh model
    # built directly at that τ (mean, cov, log_ml), on both backends
    for backend in (:dense, :cholesky)
        M1 = glm_model(X, y, Logit(), 1.0)          # unscaled base Q0 = I
        G1 = EPApprox(M1; backend=backend)
        GaussianEP.fit!(G1; maxcycles=50, tol=1e-8)
        GaussianEP.set_tau!(G1, 0.5)
        @test G1.state.τ == 0.5
        GaussianEP.fit!(G1; maxcycles=50, tol=1e-8)
        G2 = ep(glm_model(X, y, Logit(), 0.5); maxcycles=50, tol=1e-8)
        @test maximum(abs.(mean(G1) .- mean(G2))) < 1e-6
        @test maximum(abs.(cov(G1) .- cov(G2))) < 1e-6
        @test abs(log_ml(G1) - log_ml(G2)) < 1e-6
        # rebuild! respects the state's τ
        GaussianEP.rebuild!(G1)
        @test G1.state.τ == 0.5
        @test abs(log_ml(G1) - log_ml(G2)) < 1e-6
    end

    # regpath on the cholesky backend stays finite and matches dense logml
    pathc = glm_regpath(X, y, Logit(), 0.01, 10.0; nsteps=6, maxcycles=50, tol=1e-8,
                        backend=:cholesky)
    @test all(isfinite, pathc.logml)
    @test maximum(abs.(pathc.logml .- path.logml)) < 1e-6
end

@testset "glm_regpath / _regpath_tau forward fit! kwargs (batch, refresh, ess_min, crit, verbose)" begin
    Random.seed!(11)
    p, n = 4, 60
    X = randn(p, n)
    y = rand(n) .< 0.5

    # (a) batch=8 path runs and matches the batch=0 path at the fixed point
    path0 = glm_regpath(X, y, Logit(), 0.01, 10.0; nsteps=4, maxcycles=100, tol=1e-10, verbose=false)
    pathb = glm_regpath(X, y, Logit(), 0.01, 10.0; nsteps=4, maxcycles=100, tol=1e-10, batch=8, verbose=false)
    @test maximum(abs.(pathb.logml .- path0.logml)) < 1e-6
    @test maximum(abs.(pathb.μs .- path0.μs)) < 1e-6

    # (b) the kwargs are actually forwarded to fit!: verbose=false silences the
    # per-fit info summary; verbose=true emits exactly one per grid point, and
    # batch / crit show up in the summary's mode / crit fields
    lg = Test.TestLogger()
    Logging.with_logger(lg) do
        glm_regpath(X, y, Logit(), 0.01, 10.0; nsteps=3, maxcycles=50, tol=1e-8, verbose=false)
    end
    @test isempty([l for l in lg.logs if l.level >= Logging.Info])

    lg = Test.TestLogger()
    Logging.with_logger(lg) do
        glm_regpath(X, y, Logit(), 0.01, 10.0; nsteps=3, maxcycles=50, tol=1e-8, batch=8, verbose=true)
    end
    infos = [l for l in lg.logs if l.level >= Logging.Info]
    @test length(infos) == 3
    @test all(l -> occursin("mode=batch=8", String(l.message)), infos)

    lg = Test.TestLogger()
    Logging.with_logger(lg) do
        glm_regpath(X, y, Logit(), 0.01, 10.0; nsteps=2, maxcycles=50, tol=1e-8, crit=:max, verbose=true)
    end
    infos = [l for l in lg.logs if l.level >= Logging.Info]
    @test length(infos) == 2
    @test all(l -> occursin("crit=max", String(l.message)), infos)

    # (c) crit / refresh / ess_min / verbose accepted everywhere, incl. the
    # legacy-signature _regpath_tau and glm_autotune
    pathc = glm_regpath(X, y, Logit(), 0.01, 10.0; nsteps=3, maxcycles=50, tol=1e-8,
                        refresh=2, ess_min=1.0, crit=:max, verbose=false)
    @test all(isfinite, pathc.logml)
    S = GaussianEP.GLMSites(X, y, Logit(), QuadRule(20))
    τs3, μs3, lz3 = GaussianEP._regpath_tau(S, 0.01, 10.0; nsteps=3, batch=8, crit=:mean, verbose=false)
    @test issorted(τs3)
    @test all(isfinite, lz3)
    @test size(μs3) == (p, 3)
    at = glm_autotune(X, y, Logit(), 0.01, 10.0; nsteps=3, maxcycles=30, tol=1e-6, batch=8, verbose=false)
    @test isfinite(log_ml(at.G))
end
