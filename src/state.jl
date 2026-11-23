# Posterior state for the EP core, behind a small interface so that different
# linear-algebra backends can be swapped in (see linear_algebra.md).
#
# The state carries the natural parameters (Q, r) of the Gaussian approximation
# q(x) ∝ exp(-.5 x'Qx + r'x) together with the derived quantities the EP loop
# needs: Σ = Q⁻¹, μ = Σr, and logdet Q (for the online log marginal likelihood).
#
# Interface (implement per backend):
#   marginal(st, A)              -> (m, S)   moments of η = A x under q
#   cavity(st, A, Qi, ri, power) -> (m, S)   cavity moments (site removed, fraction `power`)
#   apply_site!(st, A, δQ, δr)   -> Bool     rank-d update: Q += A'δQ A, r += A'δr,
#                                            Woodbury update of Σ, μ, logdetQ; false if skipped
#   sync!(st)                             resync Σ, μ, logdetQ from Q (drift control)
#   Σa(st, a)                            Σ * a  (matvec for DenseState, triangular
#                                        solves for CholeskyState)
#   reset!(st, Q, r)                     replace (Q, r) wholesale (warm starts)
#   mean(st), cov(st), logdet(st), quadform(st)   quadform = r'Σr = r'μ
#   logZ(st)                              log ∫ exp(-.5 x'Qx + r'x) dx
#
# DenseState implements the interface for a full (dense) Q via rank-d Woodbury
# updates — the "Q is a full matrix" case. The constructor EPState(Q0, r0) is the
# dispatch hook for the sparse case: today sparse/diagonal Q0 fall back to the
# dense state (correct, just not yet exploiting sparsity); later add
#   EPState(Q0::SparseMatrixCSC, r0) = SparseState(Q0, r0)   (linear_algebra.md §3B)
# and a sparse+low-rank backend for sparse Q0 with dense sites (§3A), without
# touching the EP loop.

abstract type AbstractEPState end

mutable struct DenseState{T} <: AbstractEPState
    Q :: Matrix{T}
    Σ :: Matrix{T}
    μ :: Vector{T}
    r :: Vector{T}
    logdetQ :: T
end

# --- constructor dispatch (the sparse-case hook) ---
# backend=:dense (default) -> DenseState (Woodbury), :cholesky -> CholeskyState
# (exact rank-1 lowrankupdate!/lowrankdowndate! on the factor, no drift).
EPState(Q0::AbstractMatrix, r0::AbstractVector; backend::Symbol=:dense) =
    backend === :cholesky ? CholeskyState(Q0, r0) : DenseState(Q0, r0)
# TODO sparse backends (linear_algebra.md §3): keep working today via the dense
# fallback, swap in SparseState / sparse+low-rank state later.
EPState(Q0::SparseMatrixCSC, r0::AbstractVector; backend::Symbol=:dense) = DenseState(Q0, r0)
EPState(Q0::Diagonal, r0::AbstractVector; backend::Symbol=:dense) = DenseState(Q0, r0)

# Σa(st, a) = Σ * a — the one matvec the EP hot path needs per site
Σa(st::DenseState, a::AbstractVector) = st.Σ * a

function DenseState(Q0::AbstractMatrix, r0::AbstractVector)
    Q = Matrix{Float64}(Q0)
    r = Vector{Float64}(r0)
    C = cholesky(Symmetric(Q))
    Σ = inv(Symmetric(Q))
    μ = Σ * r
    logdetQ = 2 * sum(log, diag(C.L))   # full logdet of Q
    DenseState{Float64}(Q, Σ, μ, r, logdetQ)
end

# --- accessors ---
Statistics.mean(st::DenseState) = copy(st.μ)
Statistics.cov(st::DenseState) = st.Σ
LinearAlgebra.logdet(st::DenseState) = st.logdetQ
quadform(st::DenseState) = dot(st.r, st.μ)          # r'Σr = r'μ
logZ(st::DenseState) = 0.5*(length(st.r)*log(2π) - st.logdetQ + quadform(st))
dim(st::DenseState) = size(st.Q, 1)

# inverse of a small (d×d, d ≤ 3) symmetric PD matrix, Cholesky-based with an
# eigenvalue-clamp fallback (keeps the cavity finite if the shrink floor is hit)
function _small_pd_inv(S::Symmetric)
    try
        C = cholesky(S)
        return C \ (C' \ Matrix{Float64}(I, size(S, 1)))
    catch
        F = eigen(S)
        return Symmetric(F.vectors * Diagonal(1 ./ clamp.(F.values, 1e-12, Inf)) * F.vectors')
    end
end

# --- interface: marginal and cavity ---
# NB: method parameters are typed (`A::AM where AM<:AbstractMatrix`, ...) so the
# bodies infer concretely — an `::AbstractMatrix` parameter makes every BLAS
# call inside dispatch dynamically (perf_gap.md §3).

marginal(st::DenseState, A::AM) where AM<:AbstractMatrix = (A * st.μ, Symmetric(A * st.Σ * A'))

# cavity moments of η = A x with a fraction `power` of the site (Qi, ri) removed:
#   Λc = S⁻¹ - t·Qi ,  λc = S⁻¹m - t·ri ,  shrink t until Λc is PD
function cavity(st::DenseState, A::AM, Qi::QM, ri::RV, power::Float64=1.0) where {AM<:AbstractMatrix, QM<:AbstractMatrix, RV<:AbstractVector}
    S = Symmetric(A * st.Σ * A')
    m = A * st.μ
    Si = _small_pd_inv(S)
    QiS = Symmetric(Qi)
    λmaxQi = maximum(eigvals(QiS))
    # adaptive floor: any t ≤ λmin(S⁻¹)/λmax(Qi) keeps Λc = S⁻¹ - t·Qi PD by
    # construction; halve from `power` down to that floor (pois_stability.md §5b)
    t_min = λmaxQi > 0 ? 0.99 * minimum(eigvals(Si)) / λmaxQi : 0.0
    t = power
    Λc = Symmetric(Si - t*QiS)
    t_min = max(t_min, 0.0)          # terminate even if S was not PD (t_min < 0)
    while !isposdef(Λc) && t > t_min
        t /= 2
        Λc = Symmetric(Si - t*QiS)
    end
    λc = Si * m - t * ri
    Sc = _small_pd_inv(Λc)
    (m = Sc * λc, S = Sc)
end

# scalar (rank-1) cavity: a is a p-vector, qi/ri the scalar site params.
# Same math as the matrix cavity above, without any small-matrix overhead.
function cavity(st::DenseState, a::AV, qi::Real, ri::Real, power::Float64=1.0) where AV<:AbstractVector
    Sa = st.Σ * a
    s = dot(a, Sa)
    m = dot(a, st.μ)
    if !(s > 0) || !isfinite(s)      # Σ not PD in direction a: resync once
        sync!(st)
        Sa = st.Σ * a
        s = dot(a, Sa)
        m = dot(a, st.μ)
        !(s > 0) && (s = 1e-12)      # last resort: huge cavity variance
    end
    Si = 1 / s
    t_min = qi > 0 ? max(0.99 * Si / qi, 0.0) : 0.0
    t = power
    Λc = Si - t * qi
    while Λc <= 0 && t > t_min
        t /= 2
        Λc = Si - t * qi
    end
    λc = Si * m - t * ri
    Sc = 1 / Λc
    (m = [Sc * λc], S = reshape([Sc], 1, 1))
end

# --- interface: rank-d site update (Woodbury) ---

# Q += A'δQ A  ⟹  Σ -= ΣA' δQ (I + AΣA'δQ)⁻¹ AΣ ,  logdet Q += logdet(I + AΣA'δQ)
# (determinant lemma — exact when Σ = Q⁻¹, approximate under Woodbury drift, which
# is why sync! exists). PD backtracking: halve (δQ, δr) until K = I + AΣA'δQ is PD.
function apply_site!(st::DenseState, A::AM, δQ::DQM, δr::DRV; rmax::Float64=1e12) where {AM<:AbstractMatrix, DQM<:AbstractMatrix, DRV<:AbstractVector}
    SA = st.Σ * A'                       # p×d
    SAA = A * SA                         # d×d
    δQ = copy(δQ); δr = copy(δr)
    # non-finite proposals are a no-op for that component
    all(isfinite, δQ) || fill!(δQ, zero(eltype(δQ)))
    all(isfinite, δr) || fill!(δr, zero(eltype(δr)))
    K = I + SAA * δQ
    # Q + A'δQ A stays PD iff K = I + AΣA'δQ is PD (determinant lemma); backtrack
    # δ if not. The backtracking also guards δr (a garbage cavity can produce
    # δr ~ 1e27 with δQ ≈ 0, K trivially PD): halve both until K is PD and
    # |δr| ≤ rmax.
    while (!isposdef(Symmetric(K)) || maximum(abs.(δr)) > rmax) &&
          (maximum(abs.(δQ)) > 1e-14 || maximum(abs.(δr)) > rmax)
        δQ ./= 2
        δr ./= 2
        K = I + SAA * δQ
    end
    (isposdef(Symmetric(K)) && maximum(abs.(δr)) <= rmax) || return nothing  # skip
    st.Q .+= A' * δQ * A
    st.r .+= A' * δr
    st.logdetQ += LinearAlgebra.logdet(Matrix(K))   # determinant lemma; K is d×d and PD here
    st.Σ .-= SA * (δQ / K) * SA'
    # NB: no `st.Σ .= Symmetric(st.Σ)` here — the Woodbury update above is
    # symmetric up to roundoff (SA*(δQ/K)*SA' is a symmetric product), and the
    # full p² copy cost ~20 KB and ~5 µs per site update in the EP hot loop
    # (perf_gap.md §3); sync! every `resync` cycles controls the drift.
    st.μ .= st.Σ * st.r
    return (δQ, δr)                      # the applied change
end

# rank-1 fast path (GLM sites, A = a'): scalar cavity arithmetic + BLAS.ger!,
# same math as the generic path with d=1 but without forming d×d matrices
function apply_site!(st::DenseState, a::AV, δq::Real, δr::Real; rmax::Float64=1e12) where AV<:AbstractVector
    Sa = st.Σ * a
    apply_site!(st, a, δq, δr, Sa, dot(a, Sa); rmax=rmax)
end

# rank-1 apply with the matvec Sa = Σa and s = a'Σa precomputed (shared with
# the cavity computation in the fused hot path — saves one O(p²) matvec per
# site, perf_gap.md §7.3)
function apply_site!(st::DenseState, a::AV, δq::Real, δr::Real, Sa::Vector{Float64}, s::Float64; rmax::Float64=1e12) where AV<:AbstractVector
    isfinite(δq) || (δq = 0.0)
    isfinite(δr) || (δr = 0.0)
    if !(s > 0) || !isfinite(s)
        return nothing               # cannot validate PD-ness of the update
    end
    K = 1 + s*δq
    while (K <= 0 || abs(δr) > rmax) && (abs(δq) > 1e-14 || abs(δr) > rmax)
        δq /= 2
        δr /= 2
        K = 1 + s*δq
    end
    (K > 0 && abs(δr) <= rmax) || return nothing
    rSa = dot(Sa, st.r)                  # a'Σr = a'μ, needed for the O(p) μ update
    BLAS.ger!(δq, a, a, st.Q)            # Q += δq a a'
    st.r .+= δr .* a
    st.logdetQ += log(K)
    c = δq / K
    BLAS.ger!(-c, Sa, Sa, st.Σ)          # Σ -= (δq/K) (Σa)(Σa)'
    # O(p) incremental μ update instead of the O(p²) matvec μ = Σr
    # (algebraically identical: μ⁺ = (Σ - c Sa Sa')(r + δr a)
    #  = μ + (δr - c(Sa'r + δr s)) Sa — same formula as legacy update_approx!,
    #  perf_gap.md §3). ger! keeps Σ exactly symmetric, so no Symmetric copy.
    st.μ .+= (δr - c*(rSa + δr*s)) .* Sa
    return (δq, δr)
end

# --- interface: replace (Q, r) wholesale (warm starts across hyperparameters) ---
function reset!(st::DenseState, Q::AbstractMatrix, r::AbstractVector)
    st.Q .= Q
    st.r .= r
    sync!(st)
    return st
end

# --- interface: resync from Q (Woodbury drift control) ---
function sync!(st::DenseState)
    C = try
        cholesky(Symmetric(st.Q))
    catch
        # Q drifted non-PD between syncs (extreme site precisions): project
        # onto the PSD cone and continue instead of throwing
        F = eigen(Symmetric(st.Q))
        st.Q .= Symmetric(F.vectors * Diagonal(clamp.(F.values, 1e-12, Inf)) * F.vectors')
        cholesky(Symmetric(st.Q))
    end
    st.Σ .= inv(Symmetric(st.Q))
    st.μ .= st.Σ * st.r
    st.logdetQ = 2 * sum(log, diag(C.L))
    return st
end

# diagnostic: how far Σ has drifted from Q⁻¹
function drift(st::DenseState)
    return maximum(abs.(st.Σ .- inv(Symmetric(st.Q))))
end

# ---------------------------------------------------------------------------
# CholeskyState: the Cholesky factor of Q is the source of truth (no Woodbury
# drift). Rank-1 site updates use LinearAlgebra.lowrankupdate!/lowrankdowndate!
# (Givens-based, O(p²), in-place on the factor); rank-d updates are applied as d
# sequential rank-1 updates along the eigenvectors of δQ. The PD condition
# K = 1 + s·δq > 0 is checked BEFORE downdating — it is exactly the condition
# under which lowrankdowndate! cannot throw, so the factor is never corrupted.
# Cavities and Σa use triangular solves (C \ a = Q⁻¹a). No resync needed in
# exact arithmetic; sync! only re-factors to clean roundoff (linear_algebra.md §2).
# ---------------------------------------------------------------------------
mutable struct CholeskyState{T} <: AbstractEPState
    C :: Cholesky{T, Matrix{T}}   # Q = C.L * C.L' (or U'U)
    r :: Vector{T}
    μ :: Vector{T}
    logdetQ :: T
end

function CholeskyState(Q0::AbstractMatrix, r0::AbstractVector)
    Q = Matrix{Float64}(Q0)
    r = Vector{Float64}(r0)
    C = cholesky(Symmetric(Q))
    μ = C \ r
    logdetQ = 2 * sum(log, diag(C.factors))
    CholeskyState{Float64}(C, r, μ, logdetQ)
end

# compat: G.Q / G.Σ forwarding (EPApprox getproperty calls getproperty on the
# state) — materialised on demand, O(p³), only for read-only external use
function Base.getproperty(st::CholeskyState, s::Symbol)
    if s === :Q
        return Matrix(getfield(st, :C))
    elseif s === :Σ
        return inv(getfield(st, :C))
    end
    return getfield(st, s)
end

Statistics.mean(st::CholeskyState) = copy(st.μ)
Statistics.cov(st::CholeskyState) = inv(st.C)
LinearAlgebra.logdet(st::CholeskyState) = st.logdetQ
quadform(st::CholeskyState) = dot(st.r, st.μ)
logZ(st::CholeskyState) = 0.5*(length(st.r)*log(2π) - st.logdetQ + quadform(st))
dim(st::CholeskyState) = length(st.r)
Σa(st::CholeskyState, a::AbstractVector) = st.C \ a    # Q⁻¹a, two triangular solves
drift(st::CholeskyState) = 0.0                          # factor is exact by construction

function reset!(st::CholeskyState, Q::AbstractMatrix, r::AbstractVector)
    st.C = cholesky(Symmetric(Matrix{Float64}(Q)))
    st.r .= r
    LinearAlgebra.ldiv!(st.μ, st.C, st.r)
    st.logdetQ = 2 * sum(log, diag(st.C.factors))
    return st
end

function sync!(st::CholeskyState)
    st.C = cholesky(Symmetric(Matrix(st.C)))   # re-factor: clean roundoff only
    LinearAlgebra.ldiv!(st.μ, st.C, st.r)
    st.logdetQ = 2 * sum(log, diag(st.C.factors))
    return st
end

marginal(st::CholeskyState, A::AM) where AM<:AbstractMatrix =
    (A * st.μ, Symmetric(A * (st.C \ A')))

function cavity(st::CholeskyState, a::AV, qi::Real, ri::Real, power::Float64=1.0) where AV<:AbstractVector
    Sa = Σa(st, a)
    s = dot(a, Sa)
    m = dot(a, st.μ)
    if !(s > 0) || !isfinite(s)      # factor roundoff: re-factor once
        sync!(st)
        Sa = Σa(st, a); s = dot(a, Sa); m = dot(a, st.μ)
        !(s > 0) && (s = 1e-12)
    end
    Si = 1 / s
    t_min = qi > 0 ? max(0.99 * Si / qi, 0.0) : 0.0
    t = power
    Λc = Si - t * qi
    while Λc <= 0 && t > t_min
        t /= 2
        Λc = Si - t * qi
    end
    λc = Si * m - t * ri
    Sc = 1 / Λc
    (m = [Sc * λc], S = reshape([Sc], 1, 1))
end

function cavity(st::CholeskyState, A::AM, Qi::QM, ri::RV, power::Float64=1.0) where {AM<:AbstractMatrix, QM<:AbstractMatrix, RV<:AbstractVector}
    S = Symmetric(A * (st.C \ A'))
    m = A * st.μ
    Si = _small_pd_inv(S)
    QiS = Symmetric(Qi)
    λmaxQi = maximum(eigvals(QiS))
    t_min = max(λmaxQi > 0 ? 0.99 * minimum(eigvals(Si)) / λmaxQi : 0.0, 0.0)
    t = power
    Λc = Symmetric(Si - t*QiS)
    while !isposdef(Λc) && t > t_min
        t /= 2
        Λc = Symmetric(Si - t*QiS)
    end
    λc = Si * m - t * ri
    Sc = _small_pd_inv(Λc)
    (m = Sc * λc, S = Sc)
end

# rank-1 apply: exact lowrankupdate!/lowrankdowndate! on the factor
function apply_site!(st::CholeskyState, a::AV, δq::Real, δr::Real; rmax::Float64=1e12) where AV<:AbstractVector
    Sa = Σa(st, a)
    apply_site!(st, a, δq, δr, Sa, dot(a, Sa); rmax=rmax)
end

function apply_site!(st::CholeskyState, a::AV, δq::Real, δr::Real, Sa::Vector{Float64}, s::Float64; rmax::Float64=1e12) where AV<:AbstractVector
    isfinite(δq) || (δq = 0.0)
    isfinite(δr) || (δr = 0.0)
    if !(s > 0) || !isfinite(s)
        return nothing
    end
    K = 1 + s*δq
    while (K <= 0 || abs(δr) > rmax) && (abs(δq) > 1e-14 || abs(δr) > rmax)
        δq /= 2
        δr /= 2
        K = 1 + s*δq
    end
    (K > 0 && abs(δr) <= rmax) || return nothing
    # K > 0 is exactly the PD condition for the downdate, so the factor update
    # below cannot throw (lowrankdowndate! throws PosDefException iff v'Σv ≥ 1)
    if δq > 0
        LinearAlgebra.lowrankupdate!(st.C, sqrt(δq) .* a)
    elseif δq < 0
        LinearAlgebra.lowrankdowndate!(st.C, sqrt(-δq) .* a)
    end
    st.r .+= δr .* a
    LinearAlgebra.ldiv!(st.μ, st.C, st.r)
    st.logdetQ = 2 * sum(log, diag(st.C.factors))
    return (δq, δr)
end

# rank-d apply: eigen-split δQ = V Λ V' and apply each component as a rank-1
# update (λ > 0) or downdate (λ < 0, PD pre-checked per component; on failure
# restore the factor from a backup and halve δ)
function apply_site!(st::CholeskyState, A::AM, δQ::DQM, δr::DRV; rmax::Float64=1e12) where {AM<:AbstractMatrix, DQM<:AbstractMatrix, DRV<:AbstractVector}
    SA = st.C \ A'                       # Σ A'
    SAA = A * SA
    δQc = copy(δQ); δrc = copy(δr)
    all(isfinite, δQc) || fill!(δQc, zero(eltype(δQc)))
    all(isfinite, δrc) || fill!(δrc, zero(eltype(δrc)))
    K = I + SAA * δQc
    while (!isposdef(Symmetric(K)) || maximum(abs.(δrc)) > rmax) &&
          (maximum(abs.(δQc)) > 1e-14 || maximum(abs.(δrc)) > rmax)
        δQc ./= 2
        δrc ./= 2
        K = I + SAA * δQc
    end
    (isposdef(Symmetric(K)) && maximum(abs.(δrc)) <= rmax) || return nothing
    while true
        F = eigen(Symmetric(δQc))
        Cbak = copy(st.C)
        ok = true
        for (λ, v) in zip(F.values, eachcol(F.vectors))
            abs(λ) < 1e-300 && continue
            w = sqrt(abs(λ)) .* (A' * v)          # p-vector, Q += λ w w'... (sign via λ)
            if λ > 0
                LinearAlgebra.lowrankupdate!(st.C, w)
            else
                if !(1 - dot(w, Σa(st, w)) > 0)   # downdate would break PD
                    ok = false
                    break
                end
                LinearAlgebra.lowrankdowndate!(st.C, w)
            end
        end
        ok && break
        st.C = Cbak                                # restore and backtrack
        δQc ./= 2; δrc ./= 2
        (maximum(abs.(δQc)) > 1e-14 || maximum(abs.(δrc)) > rmax) || return nothing
    end
    st.r .+= A' * δrc
    LinearAlgebra.ldiv!(st.μ, st.C, st.r)
    st.logdetQ = 2 * sum(log, diag(st.C.factors))
    return (δQc, δrc)
end
