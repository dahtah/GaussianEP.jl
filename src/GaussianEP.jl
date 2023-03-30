module GaussianEP
using FastGaussQuadrature,LinearAlgebra,Statistics,Distributions
using GaussianProcesses,LogExpFunctions,SpecialFunctions


include("quadrule.jl")
include("sites.jl")
include("gaussian_posterior.jl")
include("interface.jl")

export QuadRule, GLMSites, BernLik, logreg, ep_glm
end
