#basic idea of Likelihood type stolen from GaussianProcesses.jl
abstract type Likelihood end

# standard-normal cdf/pdf in log form, built on SpecialFunctions (erfc/erfcx)
_normcdf(x) = 0.5 * erfc(-x / sqrt(2))
_normlogcdf(x) = x < 0 ? log(0.5) + log(erfcx(-x / sqrt(2))) - 0.5 * x^2 :
                          log(0.5 * erfc(-x / sqrt(2)))
_normlogpdf(x) = -0.5 * x^2 - 0.5 * log(2π)

#for debugging purposes, Gaussian likelihood with variance 1.
struct GLik <: Likelihood
end

function log_dens(glik::GLik, f, y)
    return -.5*(f-y)^2 - .5*log(2π)
end

"""
    Logit()

Bernoulli likelihood with logistic link: `log_dens(Logit(), η, y)` is
`log σ(η)` for `y == true` and `log(1 - σ(η))` otherwise.
"""
struct Logit <: Likelihood
end

#Link function - move to struct
logistic = (x)->1/(1+exp(-x))
loglogistic = (x)->-log1pexp(-x)
function log_dens(t::Logit, f, y)
    return y ? loglogistic(f) : loglogistic(-f)
end

"""
    PoisLik()

Poisson likelihood with exponential (log) link:
`log_dens(PoisLik(), η, y) = y*η - exp(η) - lgamma(1+y)`.
"""
struct PoisLik <: Likelihood
end

#Poisson observation with exponential link function
function log_dens(poisson::PoisLik, f, y)
    return y*f - exp(f) - lgamma(1.0 + y)
end

#Bernoulli outcome with probit link: log_dens = log Φ(η) for y true, log Φ(-η)
#otherwise. Tilted moments are analytic (see moments.jl), so quadrature is
#only used for fractional EP (power ≠ 1).
"""
    ProbitLik()

Bernoulli likelihood with probit link: `log_dens(ProbitLik(), η, y)` is
`log Φ(η)` for `y` true and `log Φ(-η)` otherwise. Tilted moments are
analytic for `power == 1` (closed forms in `src/moments.jl`).
"""
struct ProbitLik <: Likelihood
end

log_dens(::ProbitLik, f, y) = y ? _normlogcdf(f) : _normlogcdf(-f)

#Hard constraint |η| ≤ R in d = 1 (indicator of the hard sphere / interval).
#Tilted moments are analytic (truncated normal, see moments.jl).
"""
    HardSphere(R)

Hard constraint `|η| ≤ R` in d = 1 (indicator of the interval `[-R, R]`).
`log_dens(HardSphere(R), η, y)` is `0` inside and `-Inf` outside. Tilted
moments are analytic (truncated-normal closed forms via `erf`,
`src/moments.jl`), so quadrature is never called on this likelihood.
"""
struct HardSphere <: Likelihood
    R :: Float64
end

log_dens(h::HardSphere, η, y) = abs(η) <= h.R ? 0.0 : -Inf

#allow anonymous function as inverse link
struct BernLik{F} <: Likelihood
    invlink :: F
end


#Bernoulli outcome with user-specified link
function log_dens(bl::BernLik, f, y)
    p = bl.invlink(f)
    return (y>0 ? log(p) : log(1-p))
end

#Ising hard constraint: factor I(|η| = 1). Tilted moments are analytic
#(see moments.jl), so quadrature is never called on this likelihood.
"""
    IsingLik()

Hard constraint `|η| = 1` (Ising spin). Tilted moments are analytic
(`tanh`), so quadrature is never called on this likelihood.
"""
struct IsingLik <: Likelihood
end

function log_dens(::IsingLik, η, y)
    return abs(abs(η) - 1) < 1e-10 ? 0.0 : -Inf
end

#multivariate Gaussian likelihood N(y; η, I_d), any dimension d
"""
    GaussLik()

Multivariate Gaussian likelihood `N(y; η, I_d)` of any rank `d`:
`log_dens(GaussLik(), η, y) = -.5*sum(abs2, η .- y) - .5*d*log(2π)`.
"""
struct GaussLik <: Likelihood
end

function log_dens(::GaussLik, η, y)
    return -.5*sum(abs2, η .- y) - .5*length(y)*log(2π)
end

#pairwise repulsion factor exp(-β H(||η - center||)) for a gas model,
#η = x_i - x_j; center is used when one of the two particles is pinned
"""
    RepulseLik(β, H; center=nothing)

Pairwise repulsion factor `exp(-β H(‖η - center‖))` for a gas model, with
`η = x_i - x_j`; `center=nothing` means the origin (used when one of the two
particles is pinned).
"""
struct RepulseLik{F,C} <: Likelihood
    β :: Float64
    H :: F       # H(r) for r ≥ 0
    center :: C  # nothing means the origin
end

RepulseLik(β, H) = RepulseLik(β, H, nothing)

log_dens(r::RepulseLik, η, y) = -r.β * r.H(norm(r.center === nothing ? η : η .- r.center))

#unary confining factor exp(-V(η))
"""
    ConfineLik(V)

Unary confining factor `exp(-V(η))` (e.g. a harmonic trap `V(η) = ‖η‖²/2`).
"""
struct ConfineLik{F} <: Likelihood
    V :: F   # confining potential V(η)
end

log_dens(c::ConfineLik, η, y) = -c.V(η)

#Laplace (double exponential) prior factor, normalized: p(η) = (λ/2) exp(-λ|η|).
#Used in the sparse linear model (Seeger 2008), where the non-Gaussian
#sites are prior factors on the coefficients. Log-concave, which is what
#keeps EP numerically stable here.
"""
    LaplaceLik(λ)

Laplace (double-exponential) prior factor, normalized:
`log_dens(LaplaceLik(λ), η, y) = log(λ/2) - λ*abs(η)`. Used in the sparse
linear model (`sparse_lm`), where the non-Gaussian sites are prior factors
on the coefficients.
"""
struct LaplaceLik <: Likelihood
    λ :: Float64
end

log_dens(l::LaplaceLik, η, y) = log(l.λ/2) - l.λ*abs(η)

#Spike-and-slab prior factor f(η) = w δ(η) + (1-w) N(η; 0, τ2): a point mass
#at 0 (the spike, probability w) plus a Gaussian slab. The factor has no
#density w.r.t. Lebesgue measure, so log_dens below is only the continuous
#(slab) part — EP never calls it, because tilted_moments has an analytic
#rule for this likelihood (see moments.jl) that handles the spike mass
#exactly. (The field is named w, not π, so that `2π` elsewhere in a scope
#where w is bound still means 2*Base.π.)
"""
    SpikeSlabLik(w, τ2)

Spike-and-slab prior factor `f(η) = w δ(η) + (1-w) N(η; 0, τ2)`: a point mass
at 0 (the spike, probability `w`) plus a Gaussian slab of variance `τ2`.
The factor has no density w.r.t. Lebesgue measure, so `log_dens` is only the
continuous (slab) part — EP never calls it, because `tilted_moments` has an
analytic rule for this likelihood that handles the spike mass exactly.
"""
struct SpikeSlabLik <: Likelihood
    w :: Float64    # spike probability
    τ2 :: Float64   # slab variance
end

log_dens(l::SpikeSlabLik, η, y) = log1p(-l.w) - .5*(η^2/l.τ2 + log(2π*l.τ2))
