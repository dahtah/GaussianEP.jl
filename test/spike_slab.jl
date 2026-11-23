# Spike-and-slab linear model: y = X a + ε, ε ~ N(0, σ² I), mixture prior
# p(a_i) = w δ(a_i) + (1-w) N(a_i; 0, τ2). The non-Gaussian sites are the
# prior mixture factors (analytic tilted moments in moments.jl).
# For m ≤ 3 coefficients the exact posterior is a mixture of 2^m Gaussians
# (one per active subset), computable in closed form.

using Random

function _lse(v)
    m = maximum(v)
    m + log(sum(exp.(v .- m)))
end

function _subsets(p)
    out = Vector{Int}[]
    for mask in 0:(2^p - 1)
        push!(out, [i for i in 1:p if (mask >> (i-1)) & 1 == 1])
    end
    out
end

# exact posterior moments and log evidence (EPModel convention:
# log ∫ b(a)∏f da − log ∫ b da with b(a) = exp(-.5 a'Q0 a + r0'a)).
# Component S (active set): a_S ~ N(μ_S, P_S^{-1}) with
# P_S = X_S'X_S/σ² + I/τ², μ_S = P_S^{-1} X_S'y/σ², and weight
# ∝ w^{p-|S|} (1-w)^{|S|} N(y; 0, σ²I + τ² X_S X_S').
function exact_spike_slab(X, y, w, τ2, σ2; τ0=1e-8)
    p, n = size(X, 1), size(X, 2)
    Q0 = X*X'/σ2 + τ0*Matrix{Float64}(I, p, p)
    r0 = X*y/σ2
    logZ0 = .5*p*log(2π) - .5*logdet(Symmetric(Q0)) + .5*dot(r0, Q0\r0)
    lw = Float64[]
    μs = Vector{Vector{Float64}}(undef, 2^p)
    Σs = Vector{Matrix{Float64}}(undef, 2^p)
    for (k, S) in enumerate(_subsets(p))
        XS = X[S, :]'
        C = σ2*Matrix{Float64}(I, n, n) + τ2*XS*XS'
        lwm = (p-length(S) == 0 ? 0.0 : (p-length(S))*log(w)) +
              (length(S) == 0 ? 0.0 : length(S)*log1p(-w)) -
              .5*n*log(2π) - .5*logdet(Symmetric(C)) - .5*dot(y, C\y)
        push!(lw, lwm)
        if isempty(S)
            μs[k] = zeros(p); Σs[k] = zeros(p, p)
        else
            P = XS'*XS/σ2 + Matrix{Float64}(I, length(S), length(S))/τ2
            μS = P \ (XS'*y/σ2)
            μ = zeros(p); μ[S] .= μS
            Σ = zeros(p, p); Σ[S, S] .= inv(Symmetric(P))
            μs[k] = μ; Σs[k] = Σ
        end
    end
    lZ = _lse(lw)
    W = exp.(lw .- lZ)
    μ = sum(W[k]*μs[k] for k in eachindex(W))
    E2 = sum(W[k]*(Σs[k] + μs[k]*μs[k]') for k in eachindex(W))
    Σ = E2 - μ*μ'
    logI1 = dot(y, y)/(2σ2) + .5*n*log(2π*σ2) + lZ
    (μ=μ, Σ=Σ, log_ml=logI1 - logZ0)
end

@testset "spike-and-slab linear model" begin
    # structure: Gaussian likelihood in Q0/r0, one SpikeSlabLik unary site per coef
    X = [1.0 2.0 0.5; 0.0 1.0 -1.0]
    y = [0.3, -0.2, 1.1]
    w, τ2, σ2 = 0.4, 2.0, 0.7
    M = spike_slab_lm(X, y, w, τ2, σ2)
    @test M.Q0 ≈ X*X'/σ2 + 1e-8*Matrix{Float64}(I, 2, 2)
    @test M.r0 ≈ X*y/σ2
    @test length(M.sites) == 2
    @test all(s -> s.lik isa SpikeSlabLik && s.lik.w == w && s.lik.τ2 == τ2, M.sites)
    @test M.sites[1].A ≈ reshape([1.0, 0.0], 1, 2)
    @test M.sites[2].A ≈ reshape([0.0, 1.0], 1, 2)

    # EP vs exact mixture, m = 1, 2, 3
    Random.seed!(42)
    cases = [(1, 5, 0.5, 1.0, 1.0), (1, 5, 0.3, 10.0, 0.5),
             (2, 6, 0.5, 1.0, 1.0), (2, 6, 0.3, 10.0, 0.5),
             (3, 8, 0.5, 1.0, 1.0), (3, 8, 0.3, 10.0, 0.5), (3, 8, 0.7, 4.0, 1.0)]
    for (p, n, w, τ2, σ2) in cases
        X = randn(p, n)
        y = X' * randn(p) + sqrt(σ2) * randn(n)
        G = ep(spike_slab_lm(X, y, w, τ2, σ2); qr=QuadRule(20), maxcycles=200, tol=1e-10)
        ex = exact_spike_slab(X, y, w, τ2, σ2)
        @test maximum(abs.(mean(G) .- ex.μ)) < 2e-2
        @test maximum(abs.(cov(G) .- ex.Σ)) < 2e-2
        @test abs(log_ml(G) - ex.log_ml) < 5e-2
    end

    # w = 0: pure Gaussian slab, the model is exactly Gaussian and EP is exact
    Random.seed!(7)
    p, n = 2, 6
    X = randn(p, n)
    y = X' * randn(p) + randn(n)
    τ2 = 2.0
    G = ep(spike_slab_lm(X, y, 0.0, τ2, 1.0); qr=QuadRule(20), maxcycles=100, tol=1e-12)
    Q = X*X' + 1e-8*Matrix{Float64}(I, p, p) + Matrix{Float64}(I, p, p)/τ2
    r = X*y
    Σg = inv(Q)
    ex0 = exact_spike_slab(X, y, 0.0, τ2, 1.0)
    @test maximum(abs.(mean(G) .- Σg*r)) < 1e-8
    @test maximum(abs.(cov(G) .- Σg)) < 1e-8
    @test abs(log_ml(G) - ex0.log_ml) < 1e-6

    # w = 1: pure spike, the posterior is a point mass at 0
    G1 = ep(spike_slab_lm(X, y, 1.0, τ2, 1.0); qr=QuadRule(20), maxcycles=200, tol=1e-10)
    ex1 = exact_spike_slab(X, y, 1.0, τ2, 1.0)
    @test maximum(abs.(mean(G1))) < 1e-8
    @test maximum(abs.(cov(G1))) < 1e-8
    @test abs(log_ml(G1) - ex1.log_ml) < 1e-6
end

@testset "non-PD drift: cavity loop terminates, sync! robust (regression)" begin
    # exact model that used to hang the cavity halving loop (Si < 0 after
    # Woodbury drift => t_min < 0 => no termination) and then threw in sync!
    rng = MersenneTwister(42)
    p, n, k, snr = 50, 100, 3, 5.0
    d = nothing
    for r in 1:4
        X = randn(rng, p, n)
        S = randperm(rng, p)[1:k]
        β = zeros(p); β[S] = randn(rng, k) .* 3.0
        σ2 = Statistics.var(X' * β) / snr
        y = X' * β .+ sqrt(σ2) .* randn(rng, n)
        d = (X=X, y=y, σ2=σ2)
    end
    M = spike_slab_lm(d.X, d.y, 0.9, 100.0, d.σ2)
    G = ep(M; qr=QuadRule(20), tol=1e-6, maxcycles=30)   # must terminate
    @test isfinite(log_ml(G))
    @test all(isfinite, mean(G))
end
