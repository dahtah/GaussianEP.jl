module GaussianEP
using FastGaussQuadrature,LinearAlgebra,Statistics,Distributions
using GaussianProcesses

include("gaussian_posterior.jl")
include("quadrule.jl")
include("sites.jl")
export QuadRule, GLMSites, BernLik
end
