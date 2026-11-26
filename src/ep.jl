# General EP approximation for an EPModel.
# Natural parameters: Q = Q0 + Σ_i A_i' Qi A_i,  r = r0 + Σ_i A_i' ri,
# with Qi a d×d site precision and ri a d-vector. The Gaussian state
# (Q, r, Σ = Q⁻¹, μ = Σr, logdet Q) lives in an AbstractEPState backend
# (src/state.jl): DenseState keeps a full Q and maintains Σ, μ, logdetQ by
# rank-d Woodbury updates — no per-cycle re-inversion, online log_ml.

# Preallocated workspace threaded through `fit!` and the fused rank-1 hot
# path: the per-site `Sa = Σa` matvec result and the Δμ convergence buffer
# are reused across sites and cycles, so the sequential hot path allocates
# ~nothing per site (perf_alloc.md §3).
mutable struct EPWorkspace{T}
    Sa :: Vector{T}    # buffer for Sa = Σa (one matvec per site, shared cavity/apply)
    dμ :: Vector{T}    # buffer for Δμ in the convergence check
    marg_m :: Vector{T}   # cached per-site marginals m_i = a_i'μ (refresh mode)
    marg_s :: Vector{T}   # cached per-site marginal variances s_i = a_i'Σa_i
    marg_ess :: Vector{T} # per-site ESS of the reweighted quadrature (drift detector)
    sd :: Vector{T}       # posterior std devs sqrt(diag Σ), for crit=:rel (refreshed at syncs)
    cache_on :: Bool      # true while the marginal cache is in use
    ups :: Vector{NamedTuple{(:δq, :δr, :lz, :δs, :sa2), Tuple{T,T,T,T,T}}}  # per-site proposals (threaded cache sweep)
end

"""
    EPApprox(M::EPModel; qr=QuadRule(20), stein=false, power=1.0)

EP approximation of an `EPModel`: natural parameters `Q = Q0 + Σᵢ Aᵢ'QᵢAᵢ`,
`r = r0 + Σᵢ Aᵢ'rᵢ` with per-site precisions `Qi`, plus the Gaussian state
`Σ = Q⁻¹`, `μ = Σr` held in `G.state` (an `AbstractEPState`, `DenseState` for
a full `Q`). Constructed empty; run `fit!` (or `ep`) to converge.
"""
mutable struct EPApprox{T,M<:EPModel}
    model :: M
    state :: AbstractEPState
    Qi :: Vector{Matrix{T}}
    ri :: Vector{Vector{T}}
    logz :: Vector{T}
    logZ0 :: T    # log partition of the base measure, cached at construction
    lastδ :: T    # final convergence residual of the last fit! call
    ws :: EPWorkspace{T}   # preallocated buffers for the fit! hot path (perf_alloc.md)
end

# compat: G.Q, G.Σ, G.μ, G.r forward to the state (read-only use)
function Base.getproperty(G::EPApprox, s::Symbol)
    if s === :Q || s === :Σ || s === :μ || s === :r
        return getproperty(getfield(G, :state), s)
    end
    return getfield(G, s)
end

Statistics.mean(G::EPApprox) = mean(G.state)
Statistics.cov(G::EPApprox) = cov(G.state)
"""
    se(G::EPApprox)

Standard errors of the posterior mean: `sqrt.(diag(cov(G)))` (new-core
`EPApprox` path).
"""
se(G::EPApprox) = sqrt.(diag(cov(G)))
dim(G::EPApprox) = dim(G.state)
nsites(G::EPApprox) = nsites(G.model)

function Base.show(io::IO, G::EPApprox)
    print(io, "EPApprox in dim $(dim(G)) with $(nsites(G)) sites, lastδ=$(G.lastδ), state: ")
    show(io, G.state)   # compact one-line show of the state backend (src/state.jl)
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
function EPApprox(M::EPModel; qr=QuadRule(20), T=Float64, stein::Bool=false, power::Float64=1.0, backend::Symbol=:auto)
    n = nsites(M)
    st = EPState(M.Q0, M.r0; backend=backend)
    Qi = Vector{Matrix{T}}(undef, n)
    ri = Vector{Vector{T}}(undef, n)
    # function barrier: inside _init_sites! the state is concretely typed
    # (dispatch on the runtime type), so apply_site!/Σa are static and the
    # per-site init does not box through the abstract AbstractEPState field
    # (perf_alloc.md §3)
    ws = EPWorkspace{T}(zeros(T, dim(st)), zeros(T, dim(st)), zeros(T, n), zeros(T, n), zeros(T, n), ones(T, dim(st)), false, [(δq=zero(T), δr=zero(T), lz=zero(T), δs=zero(T), sa2=zero(T)) for _ in 1:n])
    # SparseState: freeze the factor for the whole init — one refactorization
    # instead of one per site (each rank-1 update marks the factor stale, and
    # ensure_fresh! would refactor per site otherwise: O(n) refactors, the
    # dominant cost of ep() at large p). Init updates are PSD-projected
    # (qi >= 0), so the frozen 4-arg apply skips the per-site solve entirely.
    if st isa SparseState
        sync!(st)
        st.jacobi = true
        freeze!(st)
    end
    _init_sites!(st, M, qr, stein, power, Qi, ri, ws, T)
    if st isa SparseState
        thaw!(st)                    # folds all init updates in: one refactor
        st.jacobi = false
    end
    sync!(st)
    logZ0 = T(safe_log_partition(M.Q0, M.r0))   # cached: log_ml is O(p) (perf_gap.md §7.4); sparse Q0 via CHOLMOD, not densified
    EPApprox{T,typeof(M)}(M, st, Qi, ri, zeros(T, n), logZ0, T(Inf), ws)
end

function _init_sites!(st::ST, M::EPModel, qr::QuadRule, stein::Bool, power::Float64, Qi, ri, ws::EPWorkspace, ::Type{T}) where {ST<:AbstractEPState, T}
    for i in 1:nsites(M)
        s = M.sites[i]
        A = s.A
        d = size(A, 1)
        if d == 1
            # scalar init at the fixed unit cavity (m=0, S=I) — algebraically
            # identical to the generic path below (Λc = pd_inv(I) = 1, λc = 0,
            # Qp = pd_inv(Σt) = 1/clamp(σ2,1e-12,Inf), psd_project of the 1×1
            # Qp-Λc = clamp(Qp-1, 0, Inf)) but without per-site small matrices
            # and 1×1 eigen decompositions (perf_alloc.md §2)
            _, μt, σ2t = _tilted_moments_scalar(s.lik, s.y, zero(T), one(T), qr, stein, power)
            Qp = 1 / max(σ2t, 1e-12)
            rp = Qp * μt
            qi = max(Qp - 1, zero(T))
            ri_i = rp
            Qi[i] = fill(qi, 1, 1)
            ri[i] = [ri_i]
            a = _rowvec(A)
            if st isa SparseState
                # frozen during init: qi >= 0 (PSD-projected) hits the
                # up-site fast path — no per-site solve (see EPApprox ctor)
                applied = apply_site!(st, a, qi, ri_i)
            else
                Sa = _Σa_into!(ws.Sa, st, a)      # buffered matvec (perf_alloc.md §3)
                applied = apply_site!(st, a, qi, ri_i, Sa, dot(a, Sa))
            end
            if applied !== nothing             # backtracked init: keep what was applied
                Qi[i][1, 1], ri[i][1] = applied
            end
        else
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
            applied = apply_site!(st, A, Qi[i], ri[i])
            if applied !== nothing
                Qi[i], ri[i] = applied
            end
        end
    end
    return nothing
end

# row vector of a rank-1 site: a view for dense A, a SparseVector for sparse A
# (SparseState / GMRF models — views of sparse rows are slow SubArrays)
_rowvec(A::AbstractMatrix) = @view(A[1, :])
_rowvec(A::AbstractSparseMatrix) = A[1, :]

# cavity parameters of η_i = A_i x, in moment form — delegated to the state
# backend. Fractional EP (Seeger 2008, Sec. 3.5): the cavity removes only a
# fraction `power` of the site approximation, Q\\i ∝ Q / t̃_i^power; the removed
# fraction t is shrunk adaptively until the cavity precision S⁻¹ - t·Qi is PD.
function cavity(G::EPApprox, i, power::Float64=1.0)
    A = G.model.sites[i].A
    if size(A, 1) == 1                        # rank-1 fast path: scalar cavity
        return cavity(G.state, _rowvec(A), G.Qi[i][1, 1], G.ri[i][1], power)
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

# likelihoods with an analytic tilted-moment rule (never need quadrature,
# so the GH-vs-trap disagreement check does not apply to them)
_has_analytic_rule(::GLik, power) = true
_has_analytic_rule(::IsingLik, power) = true
_has_analytic_rule(::SpikeSlabLik, power) = true
_has_analytic_rule(::GaussLik, power) = true
_has_analytic_rule(::HardSphere, power) = true
_has_analytic_rule(::ProbitLik, power) = power == 1.0
_has_analytic_rule(lik, power) = false

# rank-1 fast path: scalar cavity arithmetic, no small-matrix allocations.
# robust=true: if the GH rule and the trapezoid rule disagree on the tilted
# moments by more than robust_tol (under-resolved cavity: kinked/bounded/
# strongly informative factor), recompute with the trapezoid rule.
function _site_update_1d(G::EPApprox, i, qr::QuadRule, stein::Bool, power::Float64, robust::Bool, robust_tol::Float64)
    s = G.model.sites[i]
    cv = cavity(G, i, power)
    m1 = cv.m[1]; s2 = cv.S[1, 1]
    # NB: go through the tilted_moments dispatcher so analytic rules
    # (GLik, IsingLik, SpikeSlabLik, GaussLik) are used when available
    z, lzt, μt, Σt = tilted_moments(s.lik, s.y, [m1], [s2;;], qr, stein=stein, power=power)
    if robust && !_has_analytic_rule(s.lik, power) && moment_disagreement(s.lik, s.y, m1, s2, qr) > robust_tol
        z, lzt, μt, Σt = tilted_moments(s.lik, s.y, [m1], [s2;;], trap_rule(8.0, 3201), stein=stein, power=power)
    end
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

# Σa into a preallocated buffer (hot path): DenseState has an explicit Σ,
# CholeskyState solves with the factor; the generic fallback copies Σa(st, a).
# Returns the buffer.
_Σa_into!(buf::Vector{Float64}, st::DenseState, a::AbstractVector) = mul!(buf, st.Σ, a)
_Σa_into!(buf::Vector{Float64}, st::DenseState, a::AbstractSparseVector) = copyto!(buf, st.Σ * a)
function _Σa_into!(buf::Vector{Float64}, st::CholeskyState, a::AbstractVector)
    LinearAlgebra.ldiv!(buf, st.C, a)
    return buf
end
_Σa_into!(buf::Vector{Float64}, st::CholeskyState, a::AbstractSparseVector) = copyto!(buf, st.C \ a)
_Σa_into!(buf::Vector{Float64}, st::AbstractEPState, a::AbstractVector) = copyto!(buf, Σa(st, a))
# SparseState: one CHOLMOD solve into the buffer (refreshes the factor if stale)
function _Σa_into!(buf::Vector{Float64}, st::SparseState, a::AbstractVector)
    ensure_fresh!(st)
    return copyto!(buf, st.F \ a)
end

# Fused rank-1 site update + apply (hot path, power == 1): scalar cavity
# inline, ONE Σ*a matvec shared between cavity and apply (into the workspace
# buffer G.ws.Sa), scalar tilted moments, no small-matrix allocations
# (perf_gap.md §7.1/§7.3; perf_alloc.md §3). `a` is the site's row view, passed
# in by fit! (precomputed once per fit! when all sites are rank-1).
# Returns the proposed site-param change max(|dQ-Qi|, |dr-ri|) for the
# convergence check.
function _site_update_apply_1d!(G::EPApprox, i, a::AV, qr::QuadRule, stein::Bool, damp::Float64, robust::Bool, robust_tol::Float64) where AV<:AbstractVector
    # function barrier: G.state is abstractly typed (AbstractEPState field), so
    # the body is dispatched on the concrete state type — inside _hot1d! every
    # state access is statically typed and nothing boxes (perf_alloc.md §3)
    _hot1d!(G, i, a, G.state, qr, stein, damp, robust, robust_tol)
end

# state-passing variant: when the caller already holds the state in a
# concretely-typed local (fit! → _fit_impl! barrier), the call below is a
# static dispatch and allocates nothing (perf_alloc.md §3)
function _site_update_apply_1d!(G::EPApprox, i, a::AV, st::ST, qr::QuadRule, stein::Bool, damp::Float64, robust::Bool, robust_tol::Float64, use_cache::Bool=false) where {AV<:AbstractVector, ST<:AbstractEPState}
    _hot1d!(G, i, a, st, qr, stein, damp, robust, robust_tol, use_cache)
end

function _hot1d!(G::EPApprox, i, a::AV, st::ST, qr::QuadRule, stein::Bool, damp::Float64, robust::Bool, robust_tol::Float64, use_cache::Bool=false, want_ess::Bool=false) where {AV<:AbstractVector, ST<:AbstractEPState}
    δq, δr, lz, δs, sa2 = _hot1d_compute!(G, i, a, st, qr, stein, damp, robust, robust_tol, use_cache, want_ess)
    _hot1d_apply!(G, i, a, st, δq, δr, lz, sa2, use_cache)
    return δs
end

# compute part of the rank-1 hot path: read-only w.r.t. the state and the
# marginal cache (safe to thread in cache mode); returns the damped proposal
function _hot1d_compute!(G::EPApprox, i, a::AV, st::ST, qr::QuadRule, stein::Bool, damp::Float64, robust::Bool, robust_tol::Float64, use_cache::Bool=false, want_ess::Bool=false) where {AV<:AbstractVector, ST<:AbstractEPState}
    s = G.model.sites[i]
    ws = G.ws
    if use_cache
        # cached-marginal mode (refresh > 0): the per-site marginal (m_i, s_i)
        # was computed at the last refresh — no solve per site, and the
        # quadrature reweights from the cached marginal to the current cavity
        sa2 = ws.marg_s[i]
        ma = ws.marg_m[i]
        Sa = ws.Sa                    # unused by the 6-arg apply in this mode
        if !(sa2 > 0) || !isfinite(sa2)
            G.logz[i] = 0.0
            return (zero(sa2), zero(sa2), zero(sa2), zero(sa2), zero(sa2))
        end
    else
    Sa = _Σa_into!(ws.Sa, st, a)
    sa2 = dot(a, Sa)                 # cavity marginal variance s = a'Σa
    ma = dot(a, st.μ)                # cavity mean m = a'μ
    if !(sa2 > 0) || !isfinite(sa2)
        # Σ lost PD-ness in direction a (Woodbury drift): resync from Q once,
        # skip the site if still bad — never enter the cavity loop with Si ≤ 0
        # (t_min would be negative and the halving loop would not terminate)
        sync!(st)
        Sa = _Σa_into!(G.ws.Sa, st, a)
        sa2 = dot(a, Sa)
        ma = dot(a, st.μ)
        if !(sa2 > 0) || !isfinite(sa2)
            G.logz[i] = 0.0
            return (zero(sa2), zero(sa2), zero(sa2), zero(sa2), zero(sa2))
        end
    end
    end
    qi = G.Qi[i][1, 1]; ri = G.ri[i][1]
    Si = 1 / sa2
    t_min = qi > 0 ? max(0.99 * Si / qi, 0.0) : 0.0
    t = 1.0
    Λc = Si - qi                     # power = 1: remove the full site
    while Λc <= 0 && t > t_min
        t /= 2
        Λc = Si - t * qi
    end
    λc = Si * ma - t * ri
    s2 = 1 / Λc
    m1 = s2 * λc
    # Quadrature nodes are remapped to the cavity (m1, s2) — free for GH rules
    # and strictly more accurate than reweighting fixed nodes at the cached
    # marginal g when the cavity is much narrower than g (dense-X GLM: SNIS
    # reweighting from g shifted the EP fixed point by O(1) in logml). The
    # cache saves the solve/matvec per site, not the node remapping.
    lzt, μt, σ2t = _tilted_moments_scalar(s.lik, s.y, m1, s2, qr, stein, 1.0)
    if use_cache
        # diagnostic only (ess_min > 0): ESS of the SNIS reweighting c/g at the
        # cached marginal — low ESS flags a strong-site / stale-cache regime
        ws.marg_ess[i] = want_ess ? _tilted_moments_scalar_rw(s.lik, s.y, m1, s2, qr, ma, sa2, 1.0)[4] : NaN
    end
    if robust && !_has_analytic_rule(s.lik, 1.0) && moment_disagreement(s.lik, s.y, m1, s2, qr) > robust_tol
        lzt, μt, σ2t = _tilted_moments_scalar(s.lik, s.y, m1, s2, trap_rule(8.0, 3201), stein, 1.0)
    end
    Qp = 1 / σ2t; rp = Qp * μt
    dq = Qp - Λc; dr = rp - λc
    lz = lzt + _logpart1(Λc, λc) - _logpart1(Qp, rp)
    isfinite(lz) || (lz = 0.0)
    δq = damp * (dq - qi)
    δr = damp * (dr - ri)
    return (δq, δr, lz, max(abs(dq - qi), abs(dr - ri)), sa2)
end

# apply a proposal computed by _hot1d_compute! (serial; per-site state writes).
# In cache mode on the dense/cholesky backends the apply is DEFERRED: only the
# site params (G.Qi, G.ri) change and (Q, r) are rebuilt from the sites at the
# next refresh — the PD pre-check runs against the cached s (O(1), same
# approx_check semantics as the sparse frozen factor).
function _hot1d_apply!(G::EPApprox, i, a::AV, st::ST, δq, δr, lz, sa2, use_cache::Bool=false) where {AV<:AbstractVector, ST<:AbstractEPState}
    if use_cache && !(st isa SparseState)
        isfinite(δq) || (δq = 0.0)
        isfinite(δr) || (δr = 0.0)
        if sa2 > 0 && isfinite(sa2)
            K = 1 + sa2*δq
            while (K <= 0 || abs(δr) > 1e12) && (abs(δq) > 1e-14 || abs(δr) > 1e12)
                δq /= 2; δr /= 2
                K = 1 + sa2*δq
            end
            (K > 0 && abs(δr) <= 1e12) || (δq = 0.0; δr = 0.0)
        else
            δq = 0.0; δr = 0.0
        end
        G.Qi[i][1, 1] += δq
        G.ri[i][1] += δr
        G.logz[i] = lz
        return
    end
    applied = apply_site!(st, a, δq, δr, G.ws.Sa, sa2)
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

function site_update(G::EPApprox, i, qr::QuadRule; stein::Bool=false, power::Float64=1.0, robust::Bool=false, robust_tol::Float64=1e-6)
    if size(G.model.sites[i].A, 1) == 1
        return _site_update_1d(G, i, qr, stein, power, robust, robust_tol)
    end
    s = G.model.sites[i]
    cv = cavity(G, i, power)
    z, lzt, μt, Σt = _concrete_tm(tilted_moments(s.lik, s.y, cv.m, cv.S, qr, stein=stein, power=power))
    if !isfinite(lzt) || !all(isfinite, μt) || !all(isfinite, Σt) || !all(isfinite, cv.m) || !all(isfinite, cv.S)
        # degenerate quadrature / drifted state: no-op site update (dQ = dr = 0,
        # lz = 0) instead of poisoning Q with garbage natural parameters
        d = size(s.A, 1)
        return (dQ=zeros(d, d), dr=zeros(d), lz=0.0)
    end
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

# Apply a batch of rank-1 proposals (ups[k] for sites b0..b1) to the state.
# DenseState + uniform rank-1 sites + blen ≥ 8: ONE fused rank-B Woodbury
# apply (apply_sites!, O(p²) traffic on Σ/Q instead of O(B p²) for B gers),
# with fallback to per-site update_site! (PD backtracking) when the combined
# K is not PD. Other backends: per-site applies (CholeskyState's
# lowrankupdate! is rank-1 only; SparseState refactors once per batch already).
function _apply_batch!(G::EPApprox, st::DenseState, avs, b0::Int, b1::Int, ups, damp::Float64, power::Float64)
    blen = b1 - b0 + 1
    if avs !== nothing && blen >= 8
        p = dim(st)
        A = Matrix{Float64}(undef, p, blen)          # columns = site rows a_i
        δq = Vector{Float64}(undef, blen)
        δr = Vector{Float64}(undef, blen)
        for k in 1:blen
            i = b0 + k - 1
            A[:, k] .= avs[i]
            δq[k] = damp * (ups[k].dQ[1, 1] - power * G.Qi[i][1, 1])
            δr[k] = damp * (ups[k].dr[1] - power * G.ri[i][1])
        end
        applied = apply_sites!(st, A, δq, δr)
        if applied !== nothing
            δqa, δra = applied
            for k in 1:blen
                i = b0 + k - 1
                G.Qi[i][1, 1] += δqa[k]
                G.ri[i][1] += δra[k]
                G.logz[i] = ups[k].lz
            end
            return
        end
    end
    for k in 1:blen
        i = b0 + k - 1
        update_site!(G, i, ups[k].dQ, ups[k].dr, ups[k].lz, damp, power)
    end
    return
end

function _apply_batch!(G::EPApprox, st::AbstractEPState, avs, b0::Int, b1::Int, ups, damp::Float64, power::Float64)
    for k in 1:(b1 - b0 + 1)
        i = b0 + k - 1
        update_site!(G, i, ups[k].dQ, ups[k].dr, ups[k].lz, damp, power)
    end
    return
end

function update_site!(G::EPApprox, i, dQ, dr, lz, α=1.0, power::Float64=1.0)
    A = G.model.sites[i].A
    if size(A, 1) == 1                        # rank-1 fast path: scalar ger! update
        δq = α * (dQ[1, 1] - power * G.Qi[i][1, 1])
        δr = α * (dr[1] - power * G.ri[i][1])
        applied = apply_site!(G.state, _rowvec(A), δq, δr)
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

"""
    fit!(G::EPApprox; qr=QuadRule(20), maxcycles=50, tol=1e-6, damp=1.0,
         stein=false, parallel=false, power=1.0, resync=10, robust=false,
         robust_tol=1e-6, verbose=true)

Run expectation propagation on `G` in place until convergence (or
`maxcycles`), and return `G`. Same keyword semantics as [`ep`](@ref):
`damp` ∈ (0,1] damps the site updates, `parallel=true` uses damped Jacobi
(parallel) updates — equivalent to `batch=n` (full mini-batch), `stein=true`
uses Stein/autodiff moments for rank-1
sites, `power=α` is fractional EP (target unchanged), `resync` is how often
(in cycles) the state is resynced from `Q` (Woodbury drift), `robust=true`
auto-switches a site to the trapezoid rule when `moment_disagreement` flags
its cavity. Convergence is measured on both the mean and the site natural
parameters. `verbose=true` (default) prints a one-line convergence summary
(cycles run, final δ, tol, crit, sweep mode) after the loop; the
non-convergence warning is always printed, regardless of `verbose`.
"""
function fit!(G::EPApprox; qr=QuadRule(20), maxcycles=50, tol=1e-6, damp=1.0, stein::Bool=false, parallel::Bool=false, power::Float64=1.0, resync::Int=10, robust::Bool=false, robust_tol::Float64=1e-6, crit::Symbol=:rel, refresh::Int=1, ess_min::Float64=0.0, batch::Int=0, verbose::Bool=true)
    # function barrier on the concrete state type: ONE dynamic dispatch per
    # fit! call instead of one per site — inside _fit_impl! the state local is
    # statically typed, so the hot path allocates nothing per site
    # (perf_alloc.md §3)
    return _fit_impl!(G, G.state; qr=qr, maxcycles=maxcycles, tol=tol, damp=damp, stein=stein, parallel=parallel, power=power, resync=resync, robust=robust, robust_tol=robust_tol, crit=crit, refresh=refresh, ess_min=ess_min, batch=batch, verbose=verbose)
end

# Fill the per-site marginal cache (m_i, s_i) = (a_i'μ, a_i'Σa_i) from the
# current (fresh, unfrozen) factor — n solves, threaded with per-thread buffers.
function _fill_marg_cache!(ws::EPWorkspace{T}, st::AbstractEPState, avs) where T
    n = length(avs)
    p = dim(st)
    nth = Threads.nthreads()
    bufs = [zeros(T, p) for _ in 1:nth]
    Threads.@threads for i in 1:n
        b = bufs[mod1(Threads.threadid(), nth)]   # threadid can exceed nthreads()
        Sa = _Σa_into!(b, st, avs[i])
        ws.marg_s[i] = dot(avs[i], Sa)
        ws.marg_m[i] = dot(avs[i], st.μ)
    end
    return ws
end

# Rebuild the state's (Q, r) from the model base measure and the current site
# natural parameters: Q = τ·Q0 + Σᵢ qᵢ aᵢaᵢ', r = r0 + Σᵢ rᵢ aᵢ. Used by the
# cached-marginal mode on the dense/cholesky backends, where the sweep applies
# are deferred (only G.Qi/G.ri change) — the state is rebuilt and refactorized
# once per refresh. O(n·p²) per refresh, threadable fill follows.
function _rebuild_state_from_sites!(st::DenseState, G::EPApprox, avs)
    M = G.model
    copyto!(st.Q, st.τ .* M.Q0)
    copyto!(st.r, M.r0)
    for i in eachindex(avs)
        qi = G.Qi[i][1, 1]; ri = G.ri[i][1]
        (qi == 0 && ri == 0) && continue
        a = avs[i]
        if a isa AbstractSparseVector
            nz = nonzeros(a); idx = rowvals(a)
            if length(nz) == 1                    # unary site: O(1) diagonal update
                j = idx[1]
                st.Q[j, j] += qi * nz[1]^2
                st.r[j] += ri * nz[1]
            else
                ad = Vector(a)
                BLAS.ger!(qi, ad, ad, st.Q)
                st.r .+= ri .* ad
            end
        else
            BLAS.ger!(qi, a, a, st.Q)
            st.r .+= ri .* a
        end
    end
    sync!(st)
    return st
end

function _rebuild_state_from_sites!(st::CholeskyState, G::EPApprox, avs)
    M = G.model
    Q = Matrix{Float64}(st.τ .* M.Q0)
    r = Vector{Float64}(M.r0)
    for i in eachindex(avs)
        qi = G.Qi[i][1, 1]; ri = G.ri[i][1]
        (qi == 0 && ri == 0) && continue
        a = avs[i]
        if a isa AbstractSparseVector
            nz = nonzeros(a); idx = rowvals(a)
            if length(nz) == 1
                j = idx[1]
                Q[j, j] += qi * nz[1]^2
                r[j] += ri * nz[1]
            else
                ad = Vector(a)
                BLAS.ger!(qi, ad, ad, Q)
                r .+= ri .* ad
            end
        else
            BLAS.ger!(qi, a, a, Q)
            r .+= ri .* a
        end
    end
    # PD fallback: escalate a diagonal ridge if the accumulated sites lost PD
    # (per-site K checks make this rare; same spirit as _sparse_ldlt's shift)
    sh = 0.0
    scale = max(1.0, maximum(abs, diag(Q)))
    while true
        try
            reset!(st, sh > 0 ? Q + sh*I : Q, r)
            break
        catch
            sh = sh == 0.0 ? 1e-10*scale : 10*sh
            sh > 2*length(r)*scale && error("CholeskyState: could not factor rebuilt Q even with ridge")
        end
    end
    return st
end

# ===========================================================================
# fit! driver: normalize the sweep mode, then dispatch to one of two loops —
#   _fit_sweep_loop!   sequential (batch=0) or mini-batch (batch>0) sweeps
#   _fit_cached_loop!  EXPERIMENTAL cached-marginal loop (full sweeps only,
#                      refresh/ess_min; no measured gains yet — keep separate)
# The per-cycle state maintenance is dispatched on the state type
# (_pre_sweep!, _batch_pre!/_batch_post!, _fold_sweep!) so the loops contain
# no `st isa SparseState` conditionals.
# ===========================================================================

# Precompute the site row views once per fit! when all sites are rank-1 with a
# uniform A type (concrete eltype) — avoids a ~96 B view allocation per site
# per cycle in the hot path (perf_alloc.md §3). Falls back to per-site views.
function _site_row_views(G::EPApprox)
    all(s -> size(s.A, 1) == 1, G.model.sites) || return nothing
    cand = [_rowvec(s.A) for s in G.model.sites]
    return isconcretetype(eltype(cand)) ? cand : nothing
end

# Sweep-mode normalization — a single block-sweep engine parameterized by the
# batch size B: batch=1 is exactly sequential Gauss-Seidel (routed to the fused
# sequential hot path), parallel=true is exactly full Jacobi, i.e. B = n.
function _normalize_sweep(batch::Int, parallel::Bool, n::Int)
    batch == 1 && return (0, false)
    (parallel && batch == 0) && (batch = n)
    return (batch, batch >= n)
end

_cache_eligible(st::AbstractEPState) =
    st isa SparseState || st isa DenseState || st isa CholeskyState

# --- per-cycle state maintenance, dispatched on the state type -------------
# Generic (non-cache) backends: resync Σ = Q⁻¹, μ = Σr, logdetQ from Q every
# `resync` cycles (Woodbury drift; the PD backtracking in apply_site! is only
# valid when Σ ≈ Q⁻¹).
function _pre_sweep!(st::AbstractEPState, G, ws, avs, use_cache::Bool, refresh_now::Bool, c::Int, resync::Int)
    (!use_cache && c % resync == 0) && sync!(st)
    return nothing
end

# SparseState, cached mode: refactor once per refresh and freeze the factor
# for the whole sweep (one refactorization + n solves per refresh instead of
# n refactorizations); with refresh > 1 the factor and marginals stay frozen
# for `refresh` sweeps (drift accepted, quadrature reweights exactly).
function _pre_sweep!(st::SparseState, G, ws, avs, use_cache::Bool, refresh_now::Bool, c::Int, resync::Int)
    if use_cache
        if refresh_now || !getfield(st, :frozen)
            thaw!(st); sync!(st)               # fresh factor (refactor if stale)
            _fill_marg_cache!(ws, st, avs)     # n solves, once per refresh
            st.jacobi = true
            st.approx_check = true             # downdates use the cached s
            freeze!(st)
        end
    elseif c % resync == 0
        setfield!(st, :stale, true)   # force exact refactor from Q (updown drift control)
        sync!(st)
    end
    return nothing
end

# Dense/Cholesky, cached mode: DEFER the state update — rebuild (Q, r) from the
# sites and refactor once per refresh, then refill the marginal cache.
function _pre_sweep!(st::Union{DenseState,CholeskyState}, G, ws, avs, use_cache::Bool, refresh_now::Bool, c::Int, resync::Int)
    if use_cache
        if refresh_now
            _rebuild_state_from_sites!(st, G, avs)
            _fill_marg_cache!(ws, st, avs)
        end
    elseif c % resync == 0
        sync!(st)
    end
    return nothing
end

# per-batch hooks in the mini-batch loop (SparseState refactors once per batch:
# freeze for the batch, thaw after)
_batch_pre!(st::AbstractEPState) = nothing
function _batch_pre!(st::SparseState)
    thaw!(st); sync!(st)
    st.jacobi = true; st.approx_check = true
    freeze!(st)
    return nothing
end
_batch_post!(st::AbstractEPState) = nothing
function _batch_post!(st::SparseState)
    thaw!(st); st.jacobi = false; st.approx_check = false
    return nothing
end
_post_sweep!(st::AbstractEPState) = nothing
_post_sweep!(st::SparseState) = sync!(st)   # fold the sweep in for the δ check

# end-of-cycle fold-in for a frozen SparseState factor, so the convergence
# check sees the current μ (cached mode: only at refresh cycles — between
# refreshes st.μ is stale so Δμ = 0 and δ = δsite)
_fold_sweep!(st::AbstractEPState, use_cache::Bool, refresh_now::Bool) = nothing
function _fold_sweep!(st::SparseState, use_cache::Bool, refresh_now::Bool)
    if getfield(st, :frozen) && (!use_cache || refresh_now)
        thaw!(st); st.jacobi = false; st.approx_check = false
        if use_cache
            freeze!(st)                       # re-freeze for the next cached sweep
            st.jacobi = true; st.approx_check = true
        end
    end
    return nothing
end

# --- the three sweep bodies (return δsite, the max site-param change) ------

# sequential Gauss-Seidel: fused scalar hot path for rank-1 sites at power=1
# (st is concretely typed by the caller → static dispatch, 0 B/site)
function _sweep_sequential!(G::EPApprox, st::AbstractEPState, avs, qr::QuadRule, stein::Bool, damp::Float64, power::Float64, robust::Bool, robust_tol::Float64)
    δsite = 0.0
    for i in 1:nsites(G)
        if size(G.model.sites[i].A, 1) == 1 && power == 1.0
            a = avs === nothing ? _rowvec(G.model.sites[i].A) : avs[i]
            δsite = max(δsite, _site_update_apply_1d!(G, i, a, st, qr, stein, damp, robust, robust_tol))
        else
            dQ, dr, lz = site_update(G, i, qr, stein=stein, power=power, robust=robust, robust_tol=robust_tol)
            δsite = max(δsite, maximum(abs.(dQ .- power .* G.Qi[i])), maximum(abs.(dr .- power .* G.ri[i])))
            update_site!(G, i, dQ, dr, lz, damp, power)
        end
    end
    return δsite
end

# mini-batch (block Gauss-Seidel): sites split into batches of size
# B = min(batch, n); each batch's proposals are computed in parallel against
# the CURRENT state and applied serially, so later batches see earlier
# batches' updates. B = n recovers full Jacobi, B = 1 sequential. Fixes the
# stability of full Jacobi on GLMs (minibatch.md): B ~ sqrt(n)..n/8 at damp=1
# converges where Jacobi does not, to the same fixed point.
function _sweep_batch!(G::EPApprox, st::AbstractEPState, avs, batch::Int, qr::QuadRule, stein::Bool, damp::Float64, power::Float64, robust::Bool, robust_tol::Float64)
    n = nsites(G)
    B = min(batch, n)
    δsite = 0.0
    ups = Vector{NamedTuple{(:dQ, :dr, :lz)}}(undef, B)
    for b0 in 1:B:n
        b1 = min(b0 + B - 1, n)
        _batch_pre!(st)
        blen = b1 - b0 + 1
        if blen >= 2 * Threads.nthreads()
            Threads.@threads for k in 1:blen
                ups[k] = site_update(G, b0 + k - 1, qr, stein=stein, power=power, robust=robust, robust_tol=robust_tol)
            end
        else
            # small batch: serial compute, threading overhead dominates
            for k in 1:blen
                ups[k] = site_update(G, b0 + k - 1, qr, stein=stein, power=power, robust=robust, robust_tol=robust_tol)
            end
        end
        for k in 1:blen
            i = b0 + k - 1
            δsite = max(δsite, maximum(abs.(ups[k].dQ .- power .* G.Qi[i])), maximum(abs.(ups[k].dr .- power .* G.ri[i])))
        end
        _apply_batch!(G, st, avs, b0, b1, ups, damp, power)   # fused rank-B on DenseState
        _batch_post!(st)
    end
    _post_sweep!(st)
    return δsite
end

# EXPERIMENTAL cached-marginal sweep: frozen-factor sweep over the cached
# marginals — per-site work is read-only (cavity from the cache + reweighted
# quadrature, no solves), so the compute phase is threaded; the applies
# (O(1) diagonal updates for unary sites) run serially
function _sweep_cached!(G::EPApprox, st::AbstractEPState, ws::EPWorkspace, avs, qr::QuadRule, stein::Bool, damp::Float64, robust::Bool, robust_tol::Float64, want_ess::Bool)
    n = nsites(G)
    ups = ws.ups
    Threads.@threads for i in 1:n
        ups[i] = _hot1d_compute!(G, i, avs[i], st, qr, stein, damp, robust, robust_tol, true, want_ess)
    end
    δsite = 0.0
    for i in 1:n
        u = ups[i]
        δsite = max(δsite, u[4])
        _hot1d_apply!(G, i, avs[i], st, u[1], u[2], u[3], u[5], true)
    end
    return δsite
end

# --- convergence criterion ---------------------------------------------------
# :strict = max|Δμ| AND max site-param change (the mean alone can be
# stationary in symmetric models); :max = max|Δμ|; :mean = mean|Δμ| (legacy
# ep_glm criterion, perf_gap.md §1); :rel = max_i |Δμ_i| / posterior sd_i
# (scale-free). In cached mode st.μ is stale between refreshes (Δμ = 0), so
# the site-param change is the only convergence signal there.
function _convergence_δ(crit::Symbol, dμ::AbstractVector, ws::EPWorkspace, δsite::Float64, use_cache::Bool)
    if crit === :mean
        return sum(abs, dμ) / length(dμ)
    elseif crit === :max
        return maximum(abs, dμ)
    elseif crit === :rel
        δμ = 0.0
        @inbounds for i in eachindex(dμ)
            δμ = max(δμ, abs(dμ[i]) / ws.sd[i])
        end
        return use_cache ? max(δμ, δsite) : δμ
    else
        return max(maximum(abs, dμ), δsite)
    end
end

# --- the two loops ------------------------------------------------------------

# main loop: sequential (batch=0) or mini-batch (batch>0) sweeps
function _fit_sweep_loop!(G::EPApprox, st::AbstractEPState, avs, batch::Int; qr::QuadRule, maxcycles::Int, tol::Float64, damp::Float64, stein::Bool, power::Float64, robust::Bool, robust_tol::Float64, crit::Symbol, resync::Int)
    ws = G.ws
    m = copy(st.μ)
    δ = Inf
    ncycles = 0
    n = nsites(G)
    for c in 1:maxcycles
        _pre_sweep!(st, G, ws, avs, false, false, c, resync)
        crit === :rel && _post_sd!(ws.sd, st)
        δsite = batch > 0 ?
            _sweep_batch!(G, st, avs, batch, qr, stein, damp, power, robust, robust_tol) :
            _sweep_sequential!(G, st, avs, qr, stein, damp, power, robust, robust_tol)
        _fold_sweep!(st, false, false)
        dμ = ws.dμ                       # Δμ into the workspace buffer (no alloc)
        dμ .= st.μ .- m
        δ = _convergence_δ(crit, dμ, ws, δsite, false)
        ncycles = c
        δ < tol && break
        m .= st.μ
    end
    return δ, ncycles
end

# EXPERIMENTAL cached-marginal loop (full sweeps only): the per-site marginals
# (m_i, s_i) are cached and refreshed every `refresh` cycles (or early when the
# min quadrature ESS drops below `ess_min`) — between refreshes the per-site
# work is O(1) + quadrature with NO solves/matvecs. No measured gains yet;
# kept as a separate loop on purpose.
function _fit_cached_loop!(G::EPApprox, st::AbstractEPState, avs; qr::QuadRule, maxcycles::Int, tol::Float64, damp::Float64, stein::Bool, power::Float64, robust::Bool, robust_tol::Float64, crit::Symbol, refresh::Int, ess_min::Float64, resync::Int)
    ws = G.ws
    m = copy(st.μ)
    δ = Inf
    ncycles = 0
    force_refresh = true
    for c in 1:maxcycles
        refresh_now = force_refresh || (c - 1) % refresh == 0
        _pre_sweep!(st, G, ws, avs, true, refresh_now, c, resync)
        crit === :rel && _post_sd!(ws.sd, st)   # sparse: no-op while stale/frozen
        δsite = _sweep_cached!(G, st, ws, avs, qr, stein, damp, robust, robust_tol, ess_min > 0)
        _fold_sweep!(st, true, refresh_now)
        dμ = ws.dμ
        dμ .= st.μ .- m
        δ = _convergence_δ(crit, dμ, ws, δsite, true)
        force_refresh = ess_min > 0 && minimum(ws.marg_ess) < ess_min
        ncycles = c
        δ < tol && break
        m .= st.μ
    end
    return δ, ncycles
end

# --- driver -------------------------------------------------------------------

function _fit_impl!(G::EPApprox, st::ST; qr=QuadRule(20), maxcycles=50, tol=1e-6, damp=1.0, stein::Bool=false, parallel::Bool=false, power::Float64=1.0, resync::Int=10, robust::Bool=false, robust_tol::Float64=1e-6, crit::Symbol=:rel, refresh::Int=1, ess_min::Float64=0.0, batch::Int=0, verbose::Bool=true) where {ST<:AbstractEPState}
    ws = G.ws
    n = nsites(G)
    avs = _site_row_views(G)
    batch, parallel = _normalize_sweep(batch, parallel, n)
    # cached-marginal mode is a property of full sweeps (B = n): only then are
    # all proposals computed against the same state, so the marginals can be
    # cached and refreshed
    use_cache = parallel && _cache_eligible(st) && avs !== nothing && power == 1.0
    ws.cache_on = use_cache
    if crit === :rel
        # posterior sd scale for the relative criterion; ws.sd starts at ones
        # so a failed refresh degrades gracefully to the absolute criterion
        _post_sd!(ws.sd, st) || (sync!(st); _post_sd!(ws.sd, st))
    end
    if use_cache
        δ, ncycles = _fit_cached_loop!(G, st, avs; qr=qr, maxcycles=maxcycles, tol=tol, damp=damp, stein=stein, power=power, robust=robust, robust_tol=robust_tol, crit=crit, refresh=refresh, ess_min=ess_min, resync=resync)
    else
        δ, ncycles = _fit_sweep_loop!(G, st, avs, batch; qr=qr, maxcycles=maxcycles, tol=tol, damp=damp, stein=stein, power=power, robust=robust, robust_tol=robust_tol, crit=crit, resync=resync)
    end
    if use_cache && !(st isa SparseState)
        _rebuild_state_from_sites!(st, G, avs)   # fold the last sweeps into (Q, r)
    end
    st isa SparseState && (st.frozen = false; st.jacobi = false; st.approx_check = false)
    ws.cache_on = false
    sync!(st)
    G.lastδ = δ
    # one-line convergence summary (gated on `verbose`); the non-convergence
    # warning below is ALWAYS printed, independent of `verbose`
    mode = parallel ? (use_cache ? "parallel Jacobi (batch=$n), cache refresh=$refresh" :
                                   "parallel Jacobi (batch=$n)") :
           batch > 0 ? "batch=$batch" : "sequential"
    if verbose
        @info "EP fit: cycles=$ncycles, δ=$(round(G.lastδ, sigdigits=3)), tol=$tol, crit=$crit, mode=$mode, converged=$(δ < tol)"
    end
    δ < tol || @info "EP did not reach tolerance (δ=$δ); consider damping or more quadrature nodes"
    return G
end

"""
    ep(M::EPModel; qr=QuadRule(20), maxcycles=50, tol=1e-6, damp=1.0,
       stein=false, parallel=false, power=1.0, resync=10, robust=false,
       robust_tol=1e-6, crit=:rel)

Run expectation propagation on `M` until convergence and return the
`EPApprox`. `damp` ∈ (0,1] damps the site updates, `parallel=true` uses
damped Jacobi (parallel) updates — equivalent to `batch=n`, `stein=true`
uses Stein/autodiff moments
for rank-1 sites, `power=α` is fractional EP (target unchanged), `resync`
is how often (in cycles) the state is resynced from `Q` (Woodbury drift).
`robust=true` re-computes rank-1 tilted moments with the trapezoid rule
(`trap_rule`) whenever `moment_disagreement` against `qr` exceeds
`robust_tol` — the robust integration mode for kinked/bounded factors
(status.md; `test/quadrature.jl`). `crit` selects the convergence criterion:
`:rel` (default) = maxᵢ |Δμᵢ|/sqrt(Σᵢᵢ) (scale-free), `:strict` = max|Δμ| and max site-param change, `:max` = max|Δμ|,
`:mean` = mean|Δμ| (the legacy `ep_glm` criterion, perf_gap.md §1),
`:rel` = maxᵢ |Δμᵢ| / sqrt(Σᵢᵢ) — scale-free: the mean has stabilised to a
fraction `tol` of the posterior std dev (the posterior sd is refreshed from
the state at each sync; O(p) dense, O(p²) Cholesky recurrence, O(p·nnz L) at
sync points for sparse). Note the mean alone can be stationary in symmetric
models — use `:strict` there.
`backend=:dense` (default) uses the Woodbury `DenseState`; `backend=:cholesky`
uses `CholeskyState` — exact rank-1 `lowrankupdate!`/`lowrankdowndate!` on the
Cholesky factor of Q, no Woodbury drift (`src/state.jl`).
`batch=B` (default 0 = off) uses mini-batch (block Gauss-Seidel) sweeps: sites
are split into batches of size `B`, each batch's proposals are computed in
parallel against the current state and applied serially before the next batch
(B = n is full Jacobi — the same as `parallel=true`, B = 1 is sequential).
Mini-batch converges reliably on GLMs where full Jacobi does not (minibatch.md);
`refresh`/`ess_min` (cached-marginal mode) apply only to full sweeps
(`batch = n` / `parallel=true`) and are disabled when `0 < batch < n`.
`verbose=true` (default) prints a one-line
convergence summary (cycles run, final δ, tol, crit, sweep mode) after the
loop; `verbose=false` silences it (the non-convergence warning is always
printed).
"""
function ep(M::EPModel; qr=QuadRule(20), maxcycles=50, tol=1e-6, damp=1.0, stein::Bool=false, parallel::Bool=false, power::Float64=1.0, resync::Int=10, robust::Bool=false, robust_tol::Float64=1e-6, crit::Symbol=:rel, backend::Symbol=:auto, refresh::Int=1, ess_min::Float64=0.0, batch::Int=0, verbose::Bool=true)
    G = EPApprox(M, qr=qr, stein=stein, power=power, backend=backend)
    fit!(G, qr=qr, maxcycles=maxcycles, tol=tol, damp=damp, stein=stein, parallel=parallel, power=power, resync=resync, robust=robust, robust_tol=robust_tol, crit=crit, refresh=refresh, ess_min=ess_min, batch=batch, verbose=verbose)
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
    logZ(G.state) - G.logZ0 + sum(G.logz)
end
