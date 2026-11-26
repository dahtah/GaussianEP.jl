# QoL improvements: compact show methods for the state backends (src/state.jl)
# and the verbose convergence summary of ep/fit! (src/ep.jl).
using Logging, Random

@testset "qol: show methods for the state backends" begin
    rng = MersenneTwister(11)
    X = randn(rng, 4, 12)
    y = rand(rng, 12) .< 0.5
    M = glm_model(X, y, Logit())
    for backend in (:dense, :cholesky, :sparse)
        G = EPApprox(M; backend=backend)
        st = G.state
        T = string(nameof(typeof(st)))          # DenseState / CholeskyState / SparseState
        # 2-arg show: type name + dim, no error
        io = IOBuffer()
        show(io, st)
        s = String(take!(io))
        @test occursin(T, s)
        @test occursin("dim=$(GaussianEP.dim(st))", s)
        # text/plain show: no error
        show(IOBuffer(), MIME"text/plain"(), st)
        # cheap: warm up, then a loose allocation bound (tiny state, p = 4)
        show(IOBuffer(), st)
        @test (@allocated show(IOBuffer(), st)) < 100_000
        # EPApprox show includes the state line
        show(io, G)
        sG = String(take!(io))
        @test occursin("EPApprox", sG)
        @test occursin(T, sG)
        @test occursin("dim=$(GaussianEP.dim(st))", sG)
    end
    # SparseState show must be getproperty-safe: marking the factor stale must
    # survive show (getproperty on :μ/:logdetQ would call ensure_fresh! and
    # refactorize, clearing the flag)
    Gs = EPApprox(M; backend=:sparse)
    sts = Gs.state
    sts.stale = true
    s = sprint(show, sts)
    @test getfield(sts, :stale) == true
    @test occursin("nnz(Q)=$(nnz(getfield(sts, :Q)))", s)
    @test occursin("stale=true", s)
    @test occursin("frozen=false", s)
    # CholeskyState show must not materialize Q (O(p²)) or Σ = C⁻¹ (O(p³)):
    # a moderately sized state shows with a small allocation footprint
    p = 200
    A = sprand(rng, p, p, 0.05); Q0 = Matrix(A * A' + 5I)
    stc = GaussianEP.CholeskyState(Q0, randn(rng, p))
    show(IOBuffer(), stc)                       # warm up
    @test (@allocated show(IOBuffer(), stc)) < 50_000
    @test occursin("dim=$p", sprint(show, stc))
end

@testset "qol: verbose convergence reporting in ep/fit!" begin
    rng = MersenneTwister(12)
    X = randn(rng, 4, 10)
    y = randn(rng, 10)                            # Gaussian sites: EP converges exactly
    M = glm_model(X, y, GaussLik())
    infos(lg) = [l for l in lg.logs if l.level >= Logging.Info]

    # default (verbose=true): exactly one info line with cycles/δ/tol/crit/mode
    lg = Test.TestLogger()
    G = Logging.with_logger(lg) do
        ep(M; maxcycles=100, tol=1e-10)
    end
    li = infos(lg)
    @test G.lastδ < 1e-10
    @test length(li) == 1
    msg = String(li[1].message)
    @test occursin("cycles=", msg)
    @test occursin("δ=$(round(G.lastδ, sigdigits=3))", msg)   # printed at 3 sig digits
    @test occursin("tol=$(1e-10)", msg)
    @test occursin("crit=rel", msg)
    @test occursin("mode=sequential", msg)
    @test occursin("converged=true", msg)

    # sweep mode appears in the summary line
    for (kw, pat) in (((parallel=true,), "parallel Jacobi"),
                      ((parallel=true, refresh=4), "cache refresh=4"),
                      ((batch=2,), "batch=2"))
        lg = Test.TestLogger()
        Logging.with_logger(lg) do
            ep(M; maxcycles=100, tol=1e-10, kw...)
        end
        @test any(l -> occursin(pat, String(l.message)), infos(lg))
    end

    # verbose=false: no info logs at all when converged
    lg = Test.TestLogger()
    G2 = EPApprox(M)
    Logging.with_logger(lg) do
        fit!(G2; maxcycles=100, tol=1e-10, verbose=false)
    end
    @test G2.lastδ < 1e-10
    @test isempty(infos(lg))

    # verbose=false: the non-convergence warning is still printed
    lg = Test.TestLogger()
    G3 = EPApprox(M)
    Logging.with_logger(lg) do
        fit!(G3; maxcycles=1, tol=1e-300, verbose=false)
    end
    li = infos(lg)
    @test length(li) == 1
    @test occursin("did not reach tolerance", String(li[1].message))
end
