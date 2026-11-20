module GaussianEP
using FastGaussQuadrature,LinearAlgebra,Statistics,Distributions
using LogExpFunctions,SpecialFunctions
using SparseArrays,ForwardDiff


include("quadrule.jl")
include("likelihoods.jl")
include("sites.jl")
include("gaussian_posterior.jl")
include("model.jl")
include("moments.jl")
include("ep.jl")
include("interface.jl")

export QuadRule, GLMSites, Logit, PoisLik, ep_glm,log_ml,se,
       Site, EPModel, EPApprox, ep, tilted_moments,
       IsingLik, GaussLik, RepulseLik, ConfineLik, LaplaceLik, SpikeSlabLik,
       glm_model, ising_model, gas_model, sparse_lm, spike_slab_lm
end
