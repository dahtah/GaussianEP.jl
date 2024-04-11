module GaussianEP
using LinearAlgebra,Statistics,Distributions,TiltedGaussians,PDMats,StaticArrays
using LogExpFunctions,SpecialFunctions


include("quadrule.jl")
include("likelihoods.jl")
include("sites.jl")
include("gaussian_posterior.jl")
include("generic_sites.jl")

include("generic_posterior.jl")

include("interface.jl")
include("glm.jl")

export GLMSites, Logit, PoisLik, ep_glm,log_ml,se
end
