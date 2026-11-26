# Gas model. With quadratic H and V the target is Gaussian and EP must be
# (nearly) exact; with a non-quadratic repulsion we smoke-test, and we check
# the symmetry-breaking behaviour: one pinned particle leaves the model
# reflection-symmetric (EP mean ≈ 0), pinning a second particle at a chosen
# position (hyperparameter) lets EP center on a local minimum (mean ≠ 0).
@testset "gas_quadratic" begin
    np, d = 3, 2
    β = 0.5
    τ0 = 1e-3
    pin2 = [1.0, 0.5]
    M = gas_model(np, d, β=β, H=(r)->r^2, V=(η)->0.5*sum(abs2, η), τ0=τ0, pin2=pin2)
    G = ep(M, qr=QuadRule(12), maxcycles=200, tol=1e-12)
    # exact posterior: exp(-β||Ax - c||²) contributes 2β A'A to Q and 2β A'c to r;
    # exp(-.5||Ax||²) contributes A'A
    m = GaussianEP.dim(M)
    Qstar = τ0*Matrix(I, m, m)
    rstar = zeros(m)
    for s in M.sites
        if s.lik isa RepulseLik
            c = s.lik.center === nothing ? zeros(d) : s.lik.center
            Qstar .+= 2*s.lik.β * (s.A' * s.A)
            rstar .+= s.A' * (2*s.lik.β * c)
        else
            Qstar .+= s.A' * s.A
        end
    end
    @test isapprox(mean(G), Qstar\rstar, atol=1e-4)
    @test isapprox(cov(G), inv(Qstar), atol=1e-4)
end

@testset "gas_smoke" begin
    M = gas_model(4, 2, β=1.0, H=(r)->1/(r+0.5), V=(η)->0.5*sum(abs2, η), τ0=1e-3)
    G = ep(M, qr=QuadRule(10), maxcycles=50, tol=1e-6, damp=0.7)
    @test all(isfinite, mean(G))
    @test isfinite(log_ml(G))
    @test minimum(eigvals(Symmetric(cov(G)))) > 0
end

# Accuracy check: three particles (two pinned, one free) — the exact
# posterior over the free particle is low-dimensional, computable by fine
# quadrature. EP should match it well at moderate β.
@testset "gas_three_particles_accuracy" begin
    β = 1.0
    H = (r)->exp(-r)
    κ = 0.5
    V = (x)->0.5*κ*x^2
    τ0 = 1e-3
    # --- d=1 ---
    c = 1.0
    M = gas_model(3, 1, β=β, H=H, V=V, τ0=τ0, pin2=[c])
    G = ep(M, qr=QuadRule(40), maxcycles=500, tol=1e-12, damp=0.8)
    n, L = 8000, 12.0
    xs = range(-L, L, n)
    lw = [-0.5*τ0*x^2 - β*(H(abs(x)) + H(abs(x-c))) - V(x) for x in xs]
    mx = maximum(lw)
    w = exp.(lw .- mx); w ./= sum(w)
    μ_ex = sum(w .* xs)
    σ2_ex = sum(w .* (xs .- μ_ex).^2)
    @test abs(mean(G)[1] - μ_ex) < 0.05
    @test abs(cov(G)[1,1] - σ2_ex) < 0.15
    # --- d=2 ---
    pin2v = [1.0, 0.5]
    V2 = (η)->0.5*κ*sum(abs2, η)
    M2 = gas_model(3, 2, β=β, H=H, V=V2, τ0=τ0, pin2=pin2v)
    G2 = ep(M2, qr=QuadRule(25), maxcycles=500, tol=1e-12, damp=0.8)
    n2, L2 = 300, 8.0
    xs2 = range(-L2, L2, n2)
    lws = [-0.5*τ0*sum(abs2,[x,y]) - β*(H(norm([x,y])) + H(norm([x,y].-pin2v))) - V2([x,y]) for x in xs2, y in xs2]
    mx2 = maximum(lws)
    w2 = exp.(lws .- mx2); w2 ./= sum(w2)
    μ2_ex = zeros(2); Σ2_ex = zeros(2,2)
    for (i,x) in enumerate(xs2), (j,y) in enumerate(xs2)
        μ2_ex .+= w2[i,j] .* [x,y]
        Σ2_ex .+= w2[i,j] .* ([x,y]*[x,y]')
    end
    Σ2_ex .-= μ2_ex*μ2_ex'
    @test norm(mean(G2) .- μ2_ex) < 0.02
    @test norm(cov(G2) .- Σ2_ex) < 0.05
end

# log_ml must be exact for the quadratic (Gaussian) gas model, including
# the constant -β||pin2||² from the centered pair factor
@testset "gas_quadratic_logml" begin
    np, d = 3, 2
    β = 0.5
    τ0 = 1e-3
    pin2 = [1.0, 0.5]
    M = gas_model(np, d, β=β, H=(r)->r^2, V=(η)->0.5*sum(abs2, η), τ0=τ0, pin2=pin2)
    G = ep(M, qr=QuadRule(20), maxcycles=300, tol=1e-13)
    m = GaussianEP.dim(M)
    Qstar = τ0*Matrix(I, m, m)
    rstar = zeros(m)
    for s in M.sites
        if s.lik isa RepulseLik
            c = s.lik.center === nothing ? zeros(d) : s.lik.center
            Qstar .+= 2*s.lik.β*(s.A'*s.A)
            rstar .+= s.A'*(2*s.lik.β*c)
        else
            Qstar .+= s.A'*s.A
        end
    end
    C = cholesky(Symmetric(Qstar))
    exact_logZ = 0.5*(m*log(2π) - logdet(C) + rstar'*(C\rstar))
    lZ0 = GaussianEP.safe_log_partition(Matrix(M.Q0), M.r0)
    exact_logp = exact_logZ - lZ0 - β*sum(abs2, pin2)
    @test abs(log_ml(G) - exact_logp) < 1e-6
end

@testset "gas_symmetry_breaking" begin
    np, d = 4, 1
    β = 1.0
    H = (r)->exp(-r)
    V = (x)->0.5*x^2
    τ0 = 1e-3
    # energy of the pinned model (x1 = 0, x2 = 1), free particles x3, x4
    function energy(x)
        xx = [0.0, 1.0, x...]
        e = 0.0
        for i in 1:np, j in (i+1):np
            e += β*H(abs(xx[i]-xx[j]))
        end
        for i in 3:np
            e += V(xx[i])
        end
        e
    end
    # only particle 1 pinned: reflection symmetry ⟹ EP mean ≈ 0
    M1 = gas_model(np, d, β=β, H=H, V=V, τ0=τ0, pin2=nothing)
    G1 = ep(M1, qr=QuadRule(30), maxcycles=500, tol=1e-12, damp=0.8)
    @test maximum(abs.(mean(G1))) < 1e-2
    # particles 1 and 2 pinned: symmetry broken ⟹ EP centers on a local
    # basin with nonzero mean and lower energy than the symmetric point
    M2 = gas_model(np, d, β=β, H=H, V=V, τ0=τ0, pin2=[1.0])
    G2 = ep(M2, qr=QuadRule(30), maxcycles=500, tol=1e-12, damp=0.8)
    @test maximum(abs.(mean(G2))) > 0.05
    @test energy(mean(G2)) < energy(zeros(np-2))
end

@testset "gas log-repulsion (beta-ensemble, H=-log|r|): robust, no crash" begin
    rng = MersenneTwister(9)
    Hlog = r -> -log(abs(r) + 1e-300)
    pin2 = randn(rng, 2)
    V = (η) -> 0.5*sum(abs2, η)
    M = gas_model(4, 2; β=1.0, H=Hlog, V=V, τ0=1e-3, pin2=pin2)
    # standard EP converges (all damp values, several seeds)
    G = ep(M; qr=QuadRule(30), maxcycles=300, tol=1e-8, damp=0.5, verbose=false)
    @test G.lastδ < 1e-6
    @test isfinite(log_ml(G))
    @test all(isfinite, mean(G))
    # over-relaxed power must not crash on drifted states (regression:
    # logdet DomainError, LU SingularException, cholesky PosDefException)
    G2 = ep(M; qr=QuadRule(30), maxcycles=50, tol=1e-8, power=2.0, verbose=false)
    @test all(isfinite, mean(G2))
    # kinked factor |η|^β at β=2: trapezoid rule converges where tensor GH struggles
    M2 = gas_model(4, 2; β=2.0, H=Hlog, V=V, τ0=1e-3, pin2=pin2)
    Gt = ep(M2; qr=trap_rule(8.0, 401), maxcycles=300, tol=1e-8, damp=0.5, verbose=false)
    @test Gt.lastδ < 1e-6
    @test isfinite(log_ml(Gt))
end
