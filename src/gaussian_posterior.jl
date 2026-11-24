# Legacy GLM posterior representation — REWIRED onto the new EP core
# (status.md open question 5 / TODO "Rewire ep_glm onto the new core").
#
# There is now ONE EP implementation: `EPApprox` over an `EPModel`
# (src/model.jl, src/ep.jl, src/state.jl). The legacy rank-1 `ger!` loop that
# used to live in this file (struct `GLMApprox` with an explicit `Σ` maintained
# by Woodbury rank-1 updates, plus `update_approx!` / `site_update` / `cavity` /
# `set_tau!` / `_approx_fit!` / `fit!`) has been removed. `GLMApprox` survives
# as a compat alias of `EPApprox` (defined in src/interface.jl, which is
# included after the core), and `ep_glm` is a thin wrapper that builds a
# `glm_model` and calls `ep` on it.
#
# What remains here are only the small generic helpers that are not part of
# the EP loop: the `AbstractGaussApprox` supertype, 1-D moment ↔ natural
# parameter conversions, and Gaussian log-partition functions.

abstract type AbstractGaussApprox end

# moment (μ, σ²) ↔ natural (q, r) parameters of a 1-D Gaussian
function m2exp(μ,σ2)
    q=1/σ2
    r=q*μ
    (q=q,r=r)
end

function exp2m(q,r)
    σ2=1/q
    μ=σ2*r
    (μ=μ,σ2=σ2)
end

#log ∫exp(-.5*x'Qx+r'x)dx  (Q symmetric PD: Cholesky-based, no explicit inv)
function log_partition(Q :: AbstractMatrix,r :: AbstractVector)
    @assert size(Q,1) == length(r)
    n = length(r)
    C=cholesky(Symmetric(Q))
    .5*(n*log(2π)-logdet(C)+r'*(C\r))
end

function log_partition(q :: Real,r :: Real)
    .5*(log(2π)-log(q)+(1/q)*(r)^2)
end

function partition(Q :: AbstractMatrix,r :: AbstractVector)
    n = length(r)
    sqrt( ((2π)^n)/det(Q))*exp(.5*r'*(Q\r))
end
