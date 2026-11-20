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
(rank 1, or tensor rule for rank ≤ 3). `stein=true` uses Stein's lemma +
autodiff for rank 1; `power=α` gives moments of `f^α` (fractional EP).
"""
function tilted_moments(lik::Likelihood, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    _tilted_quad(Val(length(m)), lik, y, m, S, qr, stein, power)
end
_tilted_quad(::Val{1}, lik, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule, stein::Bool, power::Float64) =
    _tilted_moments_1d(lik, y, m[1], S[1,1], qr, stein, power)
function _tilted_quad(::Val{d}, lik, y, m::AbstractVector, S::AbstractMatrix, qr::QuadRule, stein::Bool, power::Float64) where {d}
    d <= 3 || throw(ArgumentError("tensor quadrature supports rank ≤ 3, got rank $d; provide an analytic moment rule for this likelihood"))
    L = cholesky(Symmetric(S)).L
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
    if !isfinite(lz)
        # all nodes have zero weight (e.g. cavity entirely inside a hard core):
        # no information — fall back to the cavity moments
        return (z=0.0, lz=-Inf, μ=copy(m), Σ=Symmetric(S))
    end
    a = exp.(lw .- lz)
    if maximum(a) >= 1 - 1e-12
        # degenerate quadrature: a single node carries all the weight — no
        # information; return the cavity moments (=> dQ = dr = 0) instead of a
        # near-zero tilted covariance (pois_stability.md §5a)
        return (z=exp(lz), lz=lz, μ=copy(m), Σ=Symmetric(S))
    end
    μ = zeros(d)
    for n in 1:npts
        μ .+= a[n] * ηs[n]
    end
    Σ = zeros(d, d)
    for n in 1:npts
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
        return (z=0.0, lz=-Inf, μ=[m1], Σ=reshape([s2], 1, 1))
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
        return (z=exp(lz), lz=lz, μ=[m1], Σ=reshape([s2], 1, 1))
    end
    σ2 = max(σ2raw, 1e-12)   # last-resort floor for a genuine tiny variance
    (z=exp(lz), lz=lz, μ=[μ], Σ=reshape([σ2], 1, 1))
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
