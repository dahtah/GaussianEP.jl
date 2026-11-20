# General EP approximation for an EPModel.
# Natural parameters: Q = Q0 + Σ_i A_i' Qi A_i,  r = r0 + Σ_i A_i' ri,
# with Qi a d×d site precision and ri a d-vector. The Gaussian state
# (Q, r, Σ = Q⁻¹, μ = Σr, logdet Q) lives in an AbstractEPState backend
# (src/state.jl): DenseState keeps a full Q and maintains Σ, μ, logdetQ by
# rank-d Woodbury updates — no per-cycle re-inversion, online log_ml.

"""
    EPApprox(M::EPModel; qr=QuadRule(20), stein=false, power=1.0)

EP approximation of an `EPModel`: natural parameters `Q = Q0 + Σᵢ Aᵢ'QᵢAᵢ`,
`r = r0 + Σᵢ Aᵢ'rᵢ` with per-site precisions `Qi`, plus the Gaussian state
`Σ = Q⁻¹`, `μ = Σr` held in `G.state` (an `AbstractEPState`, `DenseState` for
a full `Q`). Constructed empty; run `fit!` (or `ep`) to converge.
"""
struct EPApprox{T,M<:EPModel}
    model :: M
    state :: AbstractEPState
    Qi :: Vector{Matrix{T}}
    ri :: Vector{Vector{T}}
    logz :: Vector{T}
end

# compat: G.Q, G.Σ, G.μ, G.r forward to the state (read-only use)
function Base.getproperty(G::EPApprox, s::Symbol)
    if s === :Q || s === :Σ || s === :μ || s === :r
        return getfield(getfield(G, :state), s)
    end
    return getfield(G, s)
end

Statistics.mean(G::EPApprox) = mean(G.state)
Statistics.cov(G::EPApprox) = cov(G.state)
se(G::EPApprox) = sqrt.(diag(cov(G)))
dim(G::EPApprox) = dim(G.state)
nsites(G::EPApprox) = nsites(G.model)

function Base.show(io::IO, G::EPApprox)
    print(io, "EPApprox in dim $(dim(G)) with $(nsites(G)) sites, state $(typeof(G.state))")
end

# project a symmetric matrix onto the PSD cone
function psd_project(S::Symmetric)
    F = eigen(S)
    Symmetric(F.vectors * Diagonal(clamp.(F.values, 0, Inf)) * F.vectors')
end

# inverse of a symmetric matrix, clamped to stay PD
function pd_inv(S::Symmetric)
    F = eigen(S)
    Symmetric(F.vectors * Diagonal(1 ./ clamp.(F.values, 1e-12, Inf)) * F.vectors')
end

# log ∫ exp(-.5 η'Λη + λ'η) dη, robust to indefinite Λ (bookkeeping only)
function safe_log_partition(Λ::AbstractMatrix, λ::AbstractVector)
    d = length(λ)
    F = eigen(Symmetric(Λ))
    v = clamp.(F.values, 1e-300, Inf)
    .5*(d*log(2π) - sum(log, v) + sum((F.vectors'λ).^2 ./ v))
end

# Initialise with one parallel EP step at a FIXED UNIT cavity (m=0, S=I) per
# site — exactly like the legacy GLMApprox (src/gaussian_posterior.jl:44,
# hybrid_moments(S,i,0,1)) — projecting each site's natural-parameter change
# onto the PSD cone so Q stays PD.
#
# NB: initialising from the PRIOR cavity N(A*μ0, A*Σ0*A') diverges with a weak
# prior (Σ0 = τ⁻¹I, τ small): the cavity variance ‖A‖²/τ is so large that
# Gauss–Hermite nodes are too coarsely spaced to resolve the tilted density
# (e.g. Poisson: peak at log y, width 1/√y); the quadrature collapses onto a
# single node, the tilted variance underflows, and the 1e-12 clamp turned it
# into site precision Qp = 1e12, poisoning Q and diverging EP (status.md open
# question 8; root cause in pois_stability.md). The unit cavity keeps the init
# nodes within ±7.6·√d with spacing ~0.7·√d. The init is only a starting
# point — the EP fixed point is unchanged.
function EPApprox(M::EPModel; qr=QuadRule(20), T=Float64, stein::Bool=false, power::Float64=1.0)
    n = nsites(M)
    st = EPState(M.Q0, M.r0)
    Qi = Vector{Matrix{T}}(undef, n)
    ri = Vector{Vector{T}}(undef, n)
    for i in 1:n
        s = M.sites[i]
        A = s.A
        d = size(A, 1)
        S = Symmetric(Matrix{T}(I, d, d))   # fixed unit cavity ...
        mv = zeros(T, d)                    # ... at m = 0
        z, lzt, μt, Σt = tilted_moments(s.lik, s.y, mv, S, qr, stein=stein, power=power)
        Λc = pd_inv(S)
        λc = Λc * mv
        Qp = pd_inv(Symmetric(Σt))
        rp = Qp * μt
        # site natural params are tilted minus cavity (absolute, not incremental)
        Qi[i] = psd_project(Symmetric(Qp - Λc))
        ri[i] = rp - λc
        if d == 1
            applied = apply_site!(st, @view(A[1, :]), Qi[i][1, 1], ri[i][1])
            if applied !== nothing             # backtracked init: keep what was applied
                Qi[i][1, 1], ri[i][1] = applied
            end
        else
            applied = apply_site!(st, A, Qi[i], ri[i])
            if applied !== nothing
                Qi[i], ri[i] = applied
            end
        end
    end
    sync!(st)
    EPApprox{T,typeof(M)}(M, st, Qi, ri, zeros(T, n))
end

# cavity parameters of η_i = A_i x, in moment form — delegated to the state
# backend. Fractional EP (Seeger 2008, Sec. 3.5): the cavity removes only a
# fraction `power` of the site approximation, Q\\i ∝ Q / t̃_i^power; the removed
# fraction t is shrunk adaptively until the cavity precision S⁻¹ - t·Qi is PD.
function cavity(G::EPApprox, i, power::Float64=1.0)
    A = G.model.sites[i].A
    if size(A, 1) == 1                        # rank-1 fast path: scalar cavity
        return cavity(G.state, @view(A[1, :]), G.Qi[i][1, 1], G.ri[i][1], power)
    end
    cavity(G.state, A, G.Qi[i], G.ri[i], power)
end

# Normalize the tilted-moments return to a concrete type: the generic
# quadrature method mixes rank paths behind a runtime Val, so its inferred
# type has Any fields (and rank>1 paths return Σ::Symmetric). This barrier
# keeps dQ/dr/lz concrete in the EP loop.
function _concrete_tm(tm)::NamedTuple{(:z, :lz, :μ, :Σ), Tuple{Float64, Float64, Vector{Float64}, Matrix{Float64}}}
    (z=Float64(tm.z), lz=Float64(tm.lz), μ=Vector{Float64}(tm.μ), Σ=Matrix{Float64}(tm.Σ))
end

# scalar log ∫ exp(-.5 q η² + r η) dη  (rank-1 safe_log_partition)
_logpart1(q::Real, r::Real) = 0.5*(log(2π) - log(q) + r^2/q)

# rank-1 fast path: scalar cavity arithmetic, no small-matrix allocations
function _site_update_1d(G::EPApprox, i, qr::QuadRule, stein::Bool, power::Float64)
    s = G.model.sites[i]
    cv = cavity(G, i, power)
    m1 = cv.m[1]; s2 = cv.S[1, 1]
    # NB: go through the tilted_moments dispatcher so analytic rules
    # (GLik, IsingLik, SpikeSlabLik, GaussLik) are used when available
    z, lzt, μt, Σt = tilted_moments(s.lik, s.y, [m1], [s2;;], qr, stein=stein, power=power)
    Qp = 1 / Σt[1, 1]; rp = Qp * μt[1]
    Λc = 1 / s2; λc = Λc * m1
    dQ = [Qp - Λc;;]; dr = [rp - λc]
    if power == 1.0
        lz = lzt + _logpart1(Λc, λc) - _logpart1(Qp, rp)
    else
        cv1 = cavity(G, i, 1.0)
        _, lzt1, μt1, Σt1 = tilted_moments(s.lik, s.y, cv1.m, cv1.S, qr, stein=stein, power=1.0)
        Qp1 = 1 / Σt1[1, 1]; rp1 = Qp1 * μt1[1]
        Λc1 = 1 / cv1.S[1, 1]; λc1 = Λc1 * cv1.m[1]
        lz = lzt1 + _logpart1(Λc1, λc1) - _logpart1(Qp1, rp1)
    end
    isfinite(lz) || (lz = 0.0)
    (dQ=dQ, dr=dr, lz=lz)
end

function site_update(G::EPApprox, i, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    if size(G.model.sites[i].A, 1) == 1
        return _site_update_1d(G, i, qr, stein, power)
    end
    s = G.model.sites[i]
    cv = cavity(G, i, power)
    z, lzt, μt, Σt = _concrete_tm(tilted_moments(s.lik, s.y, cv.m, cv.S, qr, stein=stein, power=power))
    Qp = pd_inv(Symmetric(Σt))
    rp = Qp * μt
    Λc = pd_inv(Symmetric(cv.S))
    λc = Λc * cv.m
    dQ = Qp - Λc
    dr = rp - λc
    # evidence term: always computed with the true factor against the full
    # cavity (power=1), so log_ml estimates the evidence of the unchanged
    # target even under fractional EP
    if power == 1.0
        lz = lzt + safe_log_partition(Λc, λc) - safe_log_partition(Qp, rp)
    else
        cv1 = cavity(G, i, 1.0)
        _, lzt1, μt1, Σt1 = _concrete_tm(tilted_moments(s.lik, s.y, cv1.m, cv1.S, qr, stein=stein, power=1.0))
        Qp1 = pd_inv(Symmetric(Σt1))
        rp1 = Qp1 * μt1
        Λc1 = pd_inv(Symmetric(cv1.S))
        λc1 = Λc1 * cv1.m
        lz = lzt1 + safe_log_partition(Λc1, λc1) - safe_log_partition(Qp1, rp1)
    end
    # robust evidence term (pois_stability.md §5d): if the quadrature fell back
    # (lzt = -Inf) or the safe_log_partition difference cancels catastrophically,
    # don't write -Inf/NaN into logz — fall back to 0 (no-op site update)
    isfinite(lz) || (lz = 0.0)
    (dQ=dQ, dr=dr, lz=lz)
end

function update_site!(G::EPApprox, i, dQ, dr, lz, α=1.0, power::Float64=1.0)
    A = G.model.sites[i].A
    if size(A, 1) == 1                        # rank-1 fast path: scalar ger! update
        δq = α * (dQ[1, 1] - power * G.Qi[i][1, 1])
        δr = α * (dr[1] - power * G.ri[i][1])
        applied = apply_site!(G.state, @view(A[1, :]), δq, δr)
        if applied === nothing
            G.logz[i] = lz
            return
        end
        δqa, δra = applied
        G.Qi[i][1, 1] += δqa
        G.ri[i][1] += δra
        G.logz[i] = lz
        return
    end
    # dQ, dr are the proposed new absolute site natural params (tilted minus
    # fractional cavity). Fractional EP (Seeger 2008, Sec. 3.5) updates the
    # site by the convex combination  Q_i ← (1-power) Q_i + dQ,  i.e. the
    # applied change is δ = α (dQ - power·Q_i); the target is unchanged —
    # for Gaussian factors the fixed point is the exact factor for any power.
    δQ = α .* (dQ .- power .* G.Qi[i])
    δr = α .* (dr .- power .* G.ri[i])
    # rank-d Woodbury update of the state (PD backtracking inside apply_site!);
    # returns the applied δ, or nothing if the update was skipped
    applied = apply_site!(G.state, A, δQ, δr)
    if applied === nothing
        G.logz[i] = lz
        return
    end
    δQa, δra = applied
    G.Qi[i] .+= δQa
    G.ri[i] .+= δra
    G.logz[i] = lz
    return
end

function fit!(G::EPApprox; qr=QuadRule(20), maxcycles=50, tol=1e-8, damp=1.0, stein::Bool=false, parallel::Bool=false, power::Float64=1.0, resync::Int=10)
    m = copy(mean(G))
    δ = Inf
    n = nsites(G)
    for c in 1:maxcycles
        # Woodbury updates drift: resync Σ = Q⁻¹, μ = Σr, logdetQ from Q every
        # `resync` cycles (and once at the end) instead of re-inverting every
        # cycle — the PD backtracking in apply_site! is only valid when Σ ≈ Q⁻¹
        (c % resync == 0) && sync!(G.state)
        δsite = 0.0
        if parallel
            # Jacobi-style: all site updates computed against the same state
            # (embarrassingly parallel), then applied with damping
            ups = Vector{NamedTuple{(:dQ, :dr, :lz)}}(undef, n)
            Threads.@threads for i in 1:n
                ups[i] = site_update(G, i, qr, stein=stein, power=power)
            end
            for i in 1:n
                δsite = max(δsite, maximum(abs.(ups[i].dQ .- power .* G.Qi[i])), maximum(abs.(ups[i].dr .- power .* G.ri[i])))
                update_site!(G, i, ups[i].dQ, ups[i].dr, ups[i].lz, damp, power)
            end
        else
            for i in 1:n
                dQ, dr, lz = site_update(G, i, qr, stein=stein, power=power)
                δsite = max(δsite, maximum(abs.(dQ .- power .* G.Qi[i])), maximum(abs.(dr .- power .* G.ri[i])))
                update_site!(G, i, dQ, dr, lz, damp, power)
            end
        end
        # convergence on both the mean and the site natural parameters
        # (the mean alone can be stationary in symmetric models)
        δ = max(maximum(abs.(mean(G) .- m)), δsite)
        δ < tol && break
        m .= mean(G)
    end
    sync!(G.state)
    δ < tol || @info "EP did not reach tolerance (δ=$δ); consider damping or more quadrature nodes"
    return G
end

"""
    ep(M::EPModel; qr=QuadRule(20), maxcycles=50, tol=1e-8, damp=1.0,
       stein=false, parallel=false, power=1.0, resync=10)

Run expectation propagation on `M` until convergence and return the
`EPApprox`. `damp` ∈ (0,1] damps the site updates, `parallel=true` uses
damped Jacobi (parallel) updates, `stein=true` uses Stein/autodiff moments
for rank-1 sites, `power=α` is fractional EP (target unchanged), `resync`
is how often (in cycles) the state is resynced from `Q` (Woodbury drift).
"""
function ep(M::EPModel; qr=QuadRule(20), maxcycles=50, tol=1e-8, damp=1.0, stein::Bool=false, parallel::Bool=false, power::Float64=1.0, resync::Int=10)
    G = EPApprox(M, qr=qr, stein=stein, power=power)
    fit!(G, qr=qr, maxcycles=maxcycles, tol=tol, damp=damp, stein=stein, parallel=parallel, power=power, resync=resync)
end

"""
    log_ml(G::EPApprox)

EP estimate of the log marginal likelihood of the model, in the convention
`log Z - log Z0` where `Z = ∫ exp(logpost_unnorm(x)) dx` and `Z0` is the
partition function of the base measure `exp(-.5 x'Q0 x + r0'x)`.

O(p) thanks to the online state bookkeeping: `logZ(Q, r)` uses the maintained
`logdet Q` and `r'Σr = r'μ` instead of fresh O(p³) factorizations.
"""
function log_ml(G::EPApprox)
    M = G.model
    logZ(G.state) - safe_log_partition(Matrix(M.Q0), M.r0) + sum(G.logz)
end
