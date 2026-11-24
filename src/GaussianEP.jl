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
include("state.jl")
include("ep.jl")
include("regpath.jl")
include("spikeslab.jl")
include("interface.jl")

export QuadRule, GLMSites, Logit, PoisLik, ep_glm,log_ml,se,logdensity,
       Site, EPModel, EPApprox, ep, fit!, tilted_moments,
       IsingLik, GaussLik, RepulseLik, ConfineLik, LaplaceLik, SpikeSlabLik,
       ProbitLik, HardSphere,
       glm_model, ising_model, gas_model, sparse_lm, spike_slab_lm,
       gmrf_model, gmrf_1d,
       glm_regpath, glm_autotune,
       trap_rule, adaptive_trap_moments_1d, moment_disagreement,
       spike_slab_glm, inclusion_probs, support, slab_weight, spike_slab_autotune,
       spike_slab_glm_autotune
end
