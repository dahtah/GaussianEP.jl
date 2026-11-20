#basic idea of Likelihood type stolen from GaussianProcesses.jl
abstract type Likelihood end

#for debugging purposes, Gaussian likelihood with variance 1.
struct GLik <: Likelihood
end

function log_dens(glik::GLik, f, y)
    return -.5*(f-y)^2 - .5*log(2π)
end

struct Logit <: Likelihood
end

#Link function - move to struct
logistic = (x)->1/(1+exp(-x))
loglogistic = (x)->-log1pexp(-x)
function log_dens(t::Logit, f, y)
    return y ? loglogistic(f) : loglogistic(-f)
end

struct PoisLik <: Likelihood
end

#Poisson observation with exponential link function
function log_dens(poisson::PoisLik, f, y)
    return y*f - exp(f) - lgamma(1.0 + y)
end

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
struct IsingLik <: Likelihood
end

function log_dens(::IsingLik, η, y)
    return abs(abs(η) - 1) < 1e-10 ? 0.0 : -Inf
end

#multivariate Gaussian likelihood N(y; η, I_d), any dimension d
struct GaussLik <: Likelihood
end

function log_dens(::GaussLik, η, y)
    return -.5*sum(abs2, η .- y) - .5*length(y)*log(2π)
end

#pairwise repulsion factor exp(-β H(||η - center||)) for a gas model,
#η = x_i - x_j; center is used when one of the two particles is pinned
struct RepulseLik{F,C} <: Likelihood
    β :: Float64
    H :: F       # H(r) for r ≥ 0
    center :: C  # nothing means the origin
end

RepulseLik(β, H) = RepulseLik(β, H, nothing)

log_dens(r::RepulseLik, η, y) = -r.β * r.H(norm(r.center === nothing ? η : η .- r.center))

#unary confining factor exp(-V(η))
struct ConfineLik{F} <: Likelihood
    V :: F   # confining potential V(η)
end

log_dens(c::ConfineLik, η, y) = -c.V(η)

#Laplace (double exponential) prior factor, normalized: p(η) = (λ/2) exp(-λ|η|).
#Used in the sparse linear model (Seeger 2008), where the non-Gaussian
#sites are prior factors on the coefficients. Log-concave, which is what
#keeps EP numerically stable here.
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
struct SpikeSlabLik <: Likelihood
    w :: Float64    # spike probability
    τ2 :: Float64   # slab variance
end

log_dens(l::SpikeSlabLik, η, y) = log1p(-l.w) - .5*(η^2/l.τ2 + log(2π*l.τ2))
