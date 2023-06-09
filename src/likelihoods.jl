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
    return y ? loglogistic(f) : loglogistic(-f)  :: Float64
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
