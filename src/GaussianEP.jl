module GaussianEP
using FastGaussQuadrature,LinearAlgebra,Statistics,Distributions
using LogExpFunctions,SpecialFunctions


include("quadrule.jl")
include("likelihoods.jl")
include("sites.jl")
include("gaussian_posterior.jl")
include("interface.jl")



export QuadRule, GLMSites, Logit, PoisLik, ep_glm,log_ml
end
