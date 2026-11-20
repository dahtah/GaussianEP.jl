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
EPState(Q0::AbstractMatrix, r0::AbstractVector) = DenseState(Q0, r0)
# TODO sparse backends (linear_algebra.md §3): keep working today via the dense
# fallback, swap in SparseState / sparse+low-rank state later.
EPState(Q0::SparseMatrixCSC, r0::AbstractVector) = DenseState(Q0, r0)
EPState(Q0::Diagonal, r0::AbstractVector) = DenseState(Q0, r0)

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

marginal(st::DenseState, A::AbstractMatrix) = (A * st.μ, Symmetric(A * st.Σ * A'))

# cavity moments of η = A x with a fraction `power` of the site (Qi, ri) removed:
#   Λc = S⁻¹ - t·Qi ,  λc = S⁻¹m - t·ri ,  shrink t until Λc is PD
function cavity(st::DenseState, A::AbstractMatrix, Qi::AbstractMatrix, ri::AbstractVector, power::Float64=1.0)
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
function cavity(st::DenseState, a::AbstractVector, qi::Real, ri::Real, power::Float64=1.0)
    Sa = st.Σ * a
    s = dot(a, Sa)
    m = dot(a, st.μ)
    Si = 1 / s
    t_min = qi > 0 ? 0.99 * Si / qi : 0.0
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
function apply_site!(st::DenseState, A::AbstractMatrix, δQ::AbstractMatrix, δr::AbstractVector; rmax::Float64=1e12)
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
    st.Σ .= Symmetric(st.Σ)
    st.μ .= st.Σ * st.r
    return (δQ, δr)                      # the applied change
end

# rank-1 fast path (GLM sites, A = a'): scalar cavity arithmetic + BLAS.ger!,
# same math as the generic path with d=1 but without forming d×d matrices
function apply_site!(st::DenseState, a::AbstractVector, δq::Real, δr::Real; rmax::Float64=1e12)
    isfinite(δq) || (δq = 0.0)
    isfinite(δr) || (δr = 0.0)
    Sa = st.Σ * a
    s = dot(a, Sa)
    K = 1 + s*δq
    while (K <= 0 || abs(δr) > rmax) && (abs(δq) > 1e-14 || abs(δr) > rmax)
        δq /= 2
        δr /= 2
        K = 1 + s*δq
    end
    (K > 0 && abs(δr) <= rmax) || return nothing
    BLAS.ger!(δq, a, a, st.Q)            # Q += δq a a'
    st.r .+= δr .* a
    st.logdetQ += log(K)
    BLAS.ger!(-δq/K, Sa, Sa, st.Σ)       # Σ -= (δq/K) (Σa)(Σa)'
    st.Σ .= Symmetric(st.Σ)
    st.μ .= st.Σ * st.r
    return (δq, δr)
end

# --- interface: resync from Q (Woodbury drift control) ---
function sync!(st::DenseState)
    C = cholesky(Symmetric(st.Q))
    st.Σ .= inv(Symmetric(st.Q))
    st.μ .= st.Σ * st.r
    st.logdetQ = 2 * sum(log, diag(C.L))
    return st
end

# diagnostic: how far Σ has drifted from Q⁻¹
function drift(st::DenseState)
    return maximum(abs.(st.Σ .- inv(Symmetric(st.Q))))
end
