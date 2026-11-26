# Tilted moments. For η ~ N(m, S) with η ∈ R^d, compute under the tilted
# density ∝ f(η) N(η; m, S):
#   z = E[f(η)],   μ = E[f(η) η]/z,   Σ = Cov of η under the tilted density.
# Dispatch on the likelihood: analytic rules where available, tensor
# Gauss-Hermite quadrature (in log-space) otherwise. For rank 1, moments
# can alternatively be obtained via Stein's lemma + autodiff (stein=true):
#   E[f η]   = m Z + s² E[f (log f)']
#   E[f η²]  = (m²+s²) Z + 2 m s² E[f (log f)']
# which reuses the same quadrature nodes as Z.

# generic fallback: tensor Gauss-Hermite quadrature, rank d ≤ 3.
# power=α implements fractional EP: moments of f^α instead of f.
# The rank is dispatched on Val(length(m)) so that each path infers
# concretely (a runtime `d == 1` branch would leak the d>1 path's abstract
# `log_dens(lik::Likelihood, η::Vector, y)` call into the rank-1 return type).
# TODO: Gauss-Hermite does poorly on non-differentiable likelihoods (hard
# spheres, |η| kinks): add a robust-but-slower rule (adaptive grid /
# trapezoid on a truncated range) as an option. For hard-sphere factors in
# d=1 the tilted integrals are exact in closed form via erf — add an
# analytic rule for that case instead of quadrature.
"""
    tilted_moments(lik, y, m, S, qr; stein=false, power=1.0)

Tilted moments of the factor `f(η) = exp(log_dens(lik, η, y))` against the
Gaussian cavity `η ~ N(m, S)`: returns `(z, lz, μ, Σ)` with
`z = E[f(η)]`, `lz = log z`, `μ = E[f(η)η]/z` and `Σ` the tilted covariance.
Dispatches on the likelihood: analytic rules where available (Gaussian,
Ising, spike-and-slab), log-space Gauss–Hermite quadrature otherwise
(rank 1, or tensor rule for rank ≤ 3). For rank 2–3 a sparse grid from
`smolyak_rule(level, d)` can be passed as `qr` — it is used as-is via
`qr.grid` (signed Smolyak weights are handled). `stein=true` uses Stein's
lemma + autodiff for rank 1; `power=α` gives moments of `f^α` (fractional EP).
"""
function tilted_moments(lik::Likelihood, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    _tilted_quad(Val(length(m)), lik, y, m, S, qr, stein, power)
end
_tilted_quad(::Val{1}, lik, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule, stein::Bool, power::Float64) =
    _tilted_moments_1d(lik, y, m[1], S[1,1], qr, stein, power)
function _tilted_quad(::Val{d}, lik, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule, stein::Bool, power::Float64) where {d}
    d <= 3 || throw(ArgumentError("tensor quadrature supports rank ≤ 3, got rank $d; provide an analytic moment rule for this likelihood"))
    if !all(isfinite, S) || !all(isfinite, m)
        # state drifted to NaN (aggressive updates on non-log-concave targets):
        # signal degenerate quadrature — site_update turns lz=-Inf into a no-op
        return (z=0.0, lz=-Inf, μ=copy(m), Σ=Symmetric(Matrix{Float64}(I, d, d)))
    end
    # clamp the cavity covariance to the PD cone if needed (a drifted state can
    # hand us a non-PD S; better an approximate cavity than a crash)
    L = try
        cholesky(Symmetric(S)).L
    catch
        F = eigen(Symmetric(S))
        cholesky(Symmetric(F.vectors * Diagonal(clamp.(F.values, 1e-12, Inf)) * F.vectors')).L
    end
    # qr.grid === nothing: full tensor product of the 1-D rule (x, w).
    # qr.grid !== nothing (smolyak_rule(level, d)): explicit d×npts sparse
    # grid with signed weights — handled with a signed log-sum-exp below.
    sgn = qr.grid === nothing ? nothing : sign.(qr.w)
    if qr.grid === nothing
        k = length(qr.x)
        npts = k^d
        lw = Vector{Float64}(undef, npts)
        ηs = Vector{Vector{Float64}}(undef, npts)
        for (n, I) in enumerate(CartesianIndices(ntuple(_ -> 1:k, d)))
            z = [qr.x[I[j]] for j in 1:d]
            η = m + L*z
            ηs[n] = η
            lw[n] = power*log_dens(lik, η, y) + sum(log(qr.w[I[j]]) for j in 1:d)
        end
        lz = logsumexp(lw)
    else
        Z = qr.grid
        size(Z, 1) == d || throw(ArgumentError("quadrature grid has $(size(Z,1)) rows but the site has rank $d"))
        npts = size(Z, 2)
        lw = Vector{Float64}(undef, npts)
        ηs = Vector{Vector{Float64}}(undef, npts)
        for n in 1:npts
            η = m + L*view(Z, :, n)
            ηs[n] = η
            lw[n] = power*log_dens(lik, η, y) + qr.logw[n]   # logw = log|w|
        end
        # signed log-sum-exp: lz = log(Σ sgn(wₙ) exp(lwₙ)); if the signed
        # quadrature sum is non-positive (possible with Smolyak's negative
        # weights on hard factors) treat as degenerate → cavity moments below
        lse = logsumexp(lw)
        s = 0.0
        @inbounds for n in 1:npts
            s += sign(qr.w[n]) * exp(lw[n] - lse)
        end
        lz = s > 0.0 ? lse + log(s) : -Inf
    end
    if !isfinite(lz)
        # all nodes have zero weight (e.g. cavity entirely inside a hard core),
        # or the signed sparse-grid sum cancelled: no information — fall back
        # to the cavity moments
        return (z=0.0, lz=-Inf, μ=copy(m), Σ=Symmetric(S))
    end
    a = sgn === nothing ? exp.(lw .- lz) : sgn .* exp.(lw .- lz)
    if maximum(a) >= 1 - 1e-12
        # degenerate quadrature: a single node carries all the weight — no
        # information; return the cavity moments (=> dQ = dr = 0) instead of a
        # near-zero tilted covariance (pois_stability.md §5a)
        return (z=exp(lz), lz=lz, μ=copy(m), Σ=Symmetric(S))
    end
    μ = zeros(d)
    for n in 1:length(ηs)
        μ .+= a[n] * ηs[n]
    end
    Σ = zeros(d, d)
    for n in 1:length(ηs)
        δ = ηs[n] .- μ
        Σ .+= a[n] * (δ * δ')
    end
    (z=exp(lz), lz=lz, μ=μ, Σ=Symmetric(Σ))
end

# rank-1 quadrature, direct moments or via Stein's lemma.
# The return type is annotated: the stein branch calls ForwardDiff.derivative,
# whose declared return type is abstract even though the scalar API returns
# Float64 at runtime — without the annotation the Any leaks into site_update.
function _tilted_moments_1d(lik::Likelihood, y, m1::Real, s2::Real, qr::QuadRule, stein::Bool, power::Float64)::NamedTuple{(:z, :lz, :μ, :Σ), Tuple{Float64, Float64, Vector{Float64}, Matrix{Float64}}}
    qr.grid === nothing || throw(ArgumentError("sparse-grid rule (smolyak_rule(level, d≥2)) used on a rank-1 site; use smolyak_rule(level, 1) for rank-1 sites"))
    lz, μ, σ2 = _tm1_scalar(lik, y, m1, s2, qr, stein, power)
    (z=exp(lz), lz=lz, μ=[μ], Σ=reshape([σ2], 1, 1))
end

# scalar core of the rank-1 quadrature: returns (lz, μ, σ2) with no small
# arrays — this is the EP hot path (perf_gap.md §7.1)
function _tm1_scalar(lik::Likelihood, y, m1::Real, s2::Real, qr::QuadRule, stein::Bool, power::Float64)
    # two passes over the quadrature nodes, no intermediate arrays
    sq = sqrt(s2)
    lz = -Inf
    @inbounds for j in eachindex(qr.x)
        lw = power*log_dens(lik, sq*qr.x[j] + m1, y) + qr.logw[j]
        lz = logaddexp(lz, lw)
    end
    if !isfinite(lz)
        # all nodes have zero weight (e.g. cavity entirely inside a hard core):
        # no information — fall back to the cavity moments
        return (-Inf, m1, s2)
    end
    if stein
        # Stein's lemma: E[f η] = m Z + s² E[f'],  E[f η²] = (m²+s²) Z + 2m s² E[f'] + s⁴ E[f''],
        # with f' = f (log f)' and f'' = f ((log f)'' + ((log f)')²), all on the same nodes
        Efg = 0.0; Efgg = 0.0
        @inbounds for j in eachindex(qr.x)
            x = sq*qr.x[j] + m1
            a = exp(power*log_dens(lik, x, y) + qr.logw[j] - lz)
            g1 = power*ForwardDiff.derivative(η -> log_dens(lik, η, y), x)
            g2 = power*ForwardDiff.derivative(η -> ForwardDiff.derivative(ξ -> log_dens(lik, ξ, y), η), x)
            Efg += a*g1
            Efgg += a*(g2 + g1^2)
        end
        μ = m1 + s2*Efg
        σ2raw = (m1^2 + s2) + 2*m1*s2*Efg + s2^2*Efgg - μ^2
    else
        μ = 0.0; m2 = 0.0
        @inbounds for j in eachindex(qr.x)
            x = sq*qr.x[j] + m1
            a = exp(power*log_dens(lik, x, y) + qr.logw[j] - lz)
            μ += a*x
            m2 += a*x*x
        end
        σ2raw = m2 - μ^2
    end
    if !(σ2raw > 1e-11)
        # degenerate quadrature: a single node carries (essentially) all the
        # weight, so the raw tilted variance underflows. The quadrature then
        # carries NO information about the factor — return the CAVITY moments
        # (=> site update dQ = dr = 0) instead of letting the variance floor
        # produce σ2 = 1e-12, i.e. site precision Qp = 1e12, which poisons Q
        # (pois_stability.md §5a; status.md open question 8).
        return (lz, m1, s2)
    end
    σ2 = max(σ2raw, 1e-12)   # last-resort floor for a genuine tiny variance
    (lz, μ, σ2)
end

# --- scalar analytic tilted moments for rank-1 sites ---
# Each returns (lz, μ, σ2) or `nothing` (→ quadrature fallback). Mirrors the
# vector rules above but without any small-matrix allocation (hot path).
function _tm1_analytic(::GLik, y::Real, m1::Real, s2::Real, power::Float64)
    α = power
    Λ = 1/s2 + α
    μ = (m1/s2 + α*y)/Λ
    σ2 = 1/Λ
    lz = -.5*α*log(2π) + .5*log(2π/α) - .5*(y-m1)^2/(s2+1/α) - .5*log(2π*(s2+1/α))
    (lz, μ, σ2)
end
_tm1_analytic(::GaussLik, y::AbstractVector, m1::Real, s2::Real, power::Float64) =
    length(y) == 1 ? _tm1_analytic(GLik(), y[1], m1, s2, power) : nothing

function _tm1_analytic(::IsingLik, y, m1::Real, s2::Real, power::Float64)
    t = tanh(m1/s2)
    σ2 = max(1 - t^2, 1e-12)
    lz = -.5*(1+m1^2)/s2 + log(2*cosh(m1/s2)) - .5*log(2π*s2)
    (lz, t, σ2)
end

function _tm1_analytic(l::ProbitLik, y, m1::Real, s2::Real, power::Float64)
    power == 1.0 || return nothing
    c = y ? 1.0 : -1.0
    γ = c * m1 / sqrt(1 + s2)
    Φγ = _normcdf(γ)
    !(Φγ > 0) && return (-Inf, m1, s2)   # cavity far in the tail: no information
    φγ = exp(_normlogpdf(γ))
    lz = _normlogcdf(γ)
    E1 = m1 * Φγ + c * s2 * φγ / sqrt(1 + s2)
    E2 = (m1^2 + s2) * Φγ + c * m1 * s2 * (2 + s2) * φγ / (1 + s2)^1.5
    μ = E1 / Φγ
    σ2 = max(E2 / Φγ - μ^2, 1e-14)
    (lz, μ, σ2)
end

function _tm1_analytic(h::HardSphere, y, m1::Real, s2::Real, power::Float64)
    s = sqrt(s2); R = h.R
    a = (-R - m1) / s
    b = (R - m1) / s
    Z = _normcdf(b) - _normcdf(a)
    !(Z > 0) && return (-Inf, m1, s2)     # cavity entirely outside the sphere
    φa = exp(_normlogpdf(a)); φb = exp(_normlogpdf(b))
    lz = log(Z)
    μ = m1 + s * (φa - φb) / Z
    σ2 = max(s2 * (1 + (a * φa - b * φb) / Z - ((φa - φb) / Z)^2), 1e-14)
    (lz, μ, σ2)
end

function _tm1_analytic(l::SpikeSlabLik, y, m1::Real, s2::Real, power::Float64)
    α = power
    τ2 = l.τ2/α
    lspike = log(l.w) - .5*(m1^2/s2 + log(2π*s2))
    lslab = log1p(-l.w) - .5*(m1^2/(s2+τ2) + log(2π*(s2+τ2)))
    lz = logaddexp(lspike, lslab)
    a = exp(lslab - lz)                     # slab weight in the tilted measure
    μs = m1*τ2/(τ2+s2)
    vs = τ2*s2/(τ2+s2)
    μ = a*μs
    σ2 = max(a*(μs^2 + vs) - μ^2, 1e-14)
    (lz, μ, σ2)
end

_tm1_analytic(lik, y, m1, s2, power) = nothing   # → quadrature fallback

# Fixed-node quadrature with exact Gaussian reweighting (augmented-EP scheme,
# see status.md / sparse_backends.md): the nodes are placed once at the
# reference marginal g = N(m_g, s_g) and each sweep reweights by
# r(η) = c(η)/g(η), c = N(m_c, s_c) the current cavity — exact identity
# E_c[f] = E_g[(c/g) f]. Returns (lz, μ, σ2, ess) with ess = 1/Σa² the effective
# sample size of the reweighted nodes (drift detector: ess drops as the cavity
# drifts from g, especially in variance).
function _tm1_scalar_reweight(lik::Likelihood, y, m_c::Real, s_c::Real, qr::QuadRule, m_g::Real, s_g::Real, power::Float64)
    sqg = sqrt(s_g)
    lz = -Inf
    @inbounds for j in eachindex(qr.x)
        η = sqg*qr.x[j] + m_g
        lr = -0.5*log(s_c/s_g) - (η-m_c)^2/(2*s_c) + (η-m_g)^2/(2*s_g)
        lz = logaddexp(lz, power*log_dens(lik, η, y) + qr.logw[j] + lr)
    end
    if !isfinite(lz)
        return (-Inf, m_c, s_c, 0.0)
    end
    μ = 0.0; m2 = 0.0; sa = 0.0
    @inbounds for j in eachindex(qr.x)
        η = sqg*qr.x[j] + m_g
        lr = -0.5*log(s_c/s_g) - (η-m_c)^2/(2*s_c) + (η-m_g)^2/(2*s_g)
        a = exp(power*log_dens(lik, η, y) + qr.logw[j] + lr - lz)
        μ += a*η; m2 += a*η*η; sa += a*a
    end
    ess = 1/sa
    σ2raw = m2 - μ^2
    if !(σ2raw > 1e-11)
        return (lz, m_c, s_c, ess)   # degenerate: no information, cavity moments
    end
    (lz, μ, max(σ2raw, 1e-12), ess)
end

# dispatcher: analytic rule when available (ess = NaN), else reweighting quadrature
function _tilted_moments_scalar_rw(lik::Likelihood, y, m_c::Real, s_c::Real, qr::QuadRule, m_g::Real, s_g::Real, power::Float64)
    qr.grid === nothing || throw(ArgumentError("sparse-grid rule (smolyak_rule(level, d≥2)) used on a rank-1 site; use smolyak_rule(level, 1) for rank-1 sites"))
    a = _tm1_analytic(lik, y, m_c, s_c, power)
    a !== nothing && return (a[1], a[2], a[3], NaN)
    _tm1_scalar_reweight(lik, y, m_c, s_c, qr, m_g, s_g, power)
end

# scalar dispatcher mirroring `tilted_moments` for rank-1 sites:
# analytic rule when available, quadrature core otherwise. Returns (lz, μ, σ2).
function _tilted_moments_scalar(lik::Likelihood, y, m1::Real, s2::Real, qr::QuadRule, stein::Bool, power::Float64)
    qr.grid === nothing || throw(ArgumentError("sparse-grid rule (smolyak_rule(level, d≥2)) used on a rank-1 site; use smolyak_rule(level, 1) for rank-1 sites"))
    a = _tm1_analytic(lik, y, m1, s2, power)
    a !== nothing && return a
    _tm1_scalar(lik, y, m1, s2, qr, stein, power)
end


# analytic: Gaussian likelihood N(y; η, 1), rank 1. power=α gives moments of f^α.
function tilted_moments(::GLik, y::Real, m::AbstractVector, S::AbstractMatrix, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    α = power
    m1 = m[1]; s2 = S[1,1]
    Λ = 1/s2 + α
    μ = (m1/s2 + α*y)/Λ
    σ2 = 1/Λ
    lz = -.5*α*log(2π) + .5*log(2π/α) - .5*(y-m1)^2/(s2+1/α) - .5*log(2π*(s2+1/α))
    (z=exp(lz), lz=lz, μ=[μ], Σ=reshape([σ2], 1, 1))
end

# analytic: Ising constraint |η| = 1, rank 1 (indicator^α = indicator for α > 0).
# The tilted measure is supported on {±1}, so E[η] = tanh(m/s²), E[η²] = 1.
function tilted_moments(::IsingLik, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    m1 = m[1]; s2 = S[1,1]
    t = tanh(m1/s2)
    σ2 = max(1 - t^2, 1e-12)
    lz = -.5*(1+m1^2)/s2 + log(2*cosh(m1/s2)) - .5*log(2π*s2)
    (z=exp(lz), lz=lz, μ=[t], Σ=reshape([σ2], 1, 1))
end

# analytic: probit Bernoulli likelihood f(η) = Φ(cη) with c = ±1, rank 1,
# power = 1 (for power ≠ 1, Φ(cη)^α has no closed form — fall back to
# quadrature). For η ~ N(m, s²), with γ = c·m/√(1+s²):
#   Z       = E[f]   = Φ(γ)
#   E[η f]  = m Φ(γ) + c s² φ(γ)/√(1+s²)
#   E[η² f] = (m²+s²) Φ(γ) + c m s² (2+s²) φ(γ)/(1+s²)^{3/2}
# (Stein's lemma twice, using E[φ(cη)] = φ(γ)/√(1+s²) and
#  E[η φ(cη)] = m φ(γ)/(1+s²)^{3/2}).
function tilted_moments(l::ProbitLik, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    length(m) == 1 || throw(ArgumentError("ProbitLik sites are rank 1, got rank $(length(m))"))
    power == 1.0 || return _tilted_moments_1d(l, y, m[1], S[1, 1], qr, stein, power)
    m1 = m[1]; s2 = S[1, 1]
    c = y ? 1.0 : -1.0
    γ = c * m1 / sqrt(1 + s2)
    Φγ = _normcdf(γ)
    if !(Φγ > 0)   # cavity far in the tail: no information — cavity moments
        return (z=0.0, lz=-Inf, μ=[m1], Σ=reshape([s2], 1, 1))
    end
    φγ = exp(_normlogpdf(γ))
    lz = _normlogcdf(γ)
    E1 = m1 * Φγ + c * s2 * φγ / sqrt(1 + s2)
    E2 = (m1^2 + s2) * Φγ + c * m1 * s2 * (2 + s2) * φγ / (1 + s2)^1.5
    μ = E1 / Φγ
    σ2 = max(E2 / Φγ - μ^2, 1e-14)
    (z=Φγ, lz=lz, μ=[μ], Σ=reshape([σ2], 1, 1))
end

# analytic: hard sphere |η| ≤ R, rank 1 (indicator^α = indicator for α > 0).
# Truncated normal on [-R, R]: with a = (-R-m)/s, b = (R-m)/s, Z = Φ(b)-Φ(a):
#   E[η]  = m + s (φ(a) - φ(b))/Z
#   Var   = s² (1 + (a φ(a) - b φ(b))/Z - ((φ(a)-φ(b))/Z)²)
function tilted_moments(h::HardSphere, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    length(m) == 1 || throw(ArgumentError("HardSphere sites are rank 1, got rank $(length(m))"))
    m1 = m[1]; s2 = S[1, 1]
    s = sqrt(s2); R = h.R
    a = (-R - m1) / s
    b = (R - m1) / s
    Z = _normcdf(b) - _normcdf(a)
    if !(Z > 0)   # cavity entirely outside the sphere: no information
        return (z=0.0, lz=-Inf, μ=[m1], Σ=reshape([s2], 1, 1))
    end
    φa = exp(_normlogpdf(a)); φb = exp(_normlogpdf(b))
    lz = log(Z)
    μ = m1 + s * (φa - φb) / Z
    σ2 = s2 * (1 + (a * φa - b * φb) / Z - ((φa - φb) / Z)^2)
    σ2 = max(σ2, 1e-14)
    (z=Z, lz=lz, μ=[μ], Σ=reshape([σ2], 1, 1))
end

# analytic: spike-and-slab prior π δ(η) + (1-π) N(η; 0, τ²), rank 1.
# Against a cavity N(m, s²) the tilted measure is a mixture: the spike
# contributes mass π φ(m; 0, s²) at η = 0, the slab contributes
# (1-π) φ(m; 0, s²+τ²) with Gaussian moments μ_s = m τ²/(τ²+s²),
# v_s = τ² s²/(τ²+s²). power=α sharpens the factor to
# π^α δ(η) + (1-π)^α N(η; 0, τ²/α) (up to constants).
function tilted_moments(l::SpikeSlabLik, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    length(m) == 1 || throw(ArgumentError("SpikeSlabLik sites are rank 1, got rank $(length(m))"))
    α = power
    m1 = m[1]; s2 = S[1,1]
    τ2 = l.τ2/α
    lspike = log(l.w) - .5*(m1^2/s2 + log(2π*s2))              # log w φ(m; 0, s²)
    lslab = log1p(-l.w) - .5*(m1^2/(s2+τ2) + log(2π*(s2+τ2)))  # log (1-w) φ(m; 0, s²+τ²)
    lz = logaddexp(lspike, lslab)
    a = exp(lslab - lz)                                         # slab weight in the tilted measure
    μs = m1*τ2/(τ2+s2)
    vs = τ2*s2/(τ2+s2)
    μ = a*μs
    σ2 = max(a*(μs^2 + vs) - μ^2, 1e-14)
    (z=exp(lz), lz=lz, μ=[μ], Σ=reshape([σ2], 1, 1))
end

# analytic: Gaussian likelihood N(y; η, I_d), any rank d. power=α gives moments of f^α.
function tilted_moments(::GaussLik, y::AbstractVector, m::AbstractVector, S::AbstractMatrix, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    α = power
    d = length(y)
    Λ = inv(Symmetric(S)) + α*I
    μ = Λ \ (Symmetric(S) \ m + α*y)
    Σ = inv(Symmetric(Λ))
    C = S + I/α
    lz = -.5*α*d*log(2π) + .5*d*log(2π/α) - .5*(y-m)'*(C\(y-m)) - .5*(d*log(2π) + logdet(Symmetric(C)))
    (z=exp(lz), lz=lz, μ=μ, Σ=Σ)
end

"""
    adaptive_trap_moments_1d(lik, y, m, s2; L=8.0, N0=101, Nmax=12801, tol=1e-10)

Tilted moments of a rank-1 factor by successive trapezoid refinement:
compute with `trap_rule(L, N)` for `N = N0, 2N0-1, 4N0-3, ...` until the
`(lz, μ, σ²)` triple changes by less than `tol` (or `Nmax` is reached).
Robust "integration mode" for non-smooth or bounded factors where
Gauss–Hermite under-resolves the tilted density (status.md: numerical
stability of moment computation; open question 7).
"""
function adaptive_trap_moments_1d(lik::Likelihood, y, m::Real, s2::Real;
                                  L::Real=8.0, N0::Int=101, Nmax::Int=12801, tol::Float64=1e-10)
    prev = nothing
    N = N0
    while true
        tm = _tilted_moments_1d(lik, y, m, s2, trap_rule(L, N), false, 1.0)
        if prev !== nothing
            d = max(abs(tm.lz - prev.lz), abs(tm.μ[1] - prev.μ[1]), abs(tm.Σ[1,1] - prev.Σ[1,1]))
            d < tol && return tm
        end
        prev = tm
        N >= Nmax && return tm
        N = min(2*N - 1, Nmax)
    end
end

"""
    moment_disagreement(lik, y, m, s2, qr::QuadRule; L=8.0, N=801)

Diagnostic: max abs difference between the Gauss–Hermite tilted moments
(rule `qr`) and trapezoid-rule moments of the same tilted density. A large
value flags a cavity where GH under-resolves the factor (strongly
informative, kinked or bounded sites) — the detector suggested in
status.md ("numerical stability of moment computation").
"""
function moment_disagreement(lik::Likelihood, y, m::Real, s2::Real, qr::QuadRule;
                             L::Real=8.0, N::Int=801)
    a = _tilted_moments_1d(lik, y, m, s2, qr, false, 1.0)
    b = _tilted_moments_1d(lik, y, m, s2, trap_rule(L, N), false, 1.0)
    max(abs(a.lz - b.lz), abs(a.μ[1] - b.μ[1]), abs(a.Σ[1,1] - b.Σ[1,1]))
end
