Distributions.MultivariateNormal(G :: AbstractGaussApprox) = MultivariateNormal(mean(G),Symmetric(cov(G)))
Distributions.MultivariateNormal(G :: EPApprox) = MultivariateNormal(mean(G),Symmetric(cov(G)))

# ---------------------------------------------------------------------------
# Legacy GLM API, rewired onto the new EP core (status.md open question 5 /
# TODO "Rewire ep_glm onto the new core"): there is a single EP implementation
# (`EPApprox` over an `EPModel`, src/ep.jl). `GLMApprox` is kept as a compat
# alias of `EPApprox`, and `ep_glm` is a thin wrapper that builds a
# `glm_model` and calls `ep`. The legacy convergence criterion (mean|Δμ| per
# cycle) maps to the new core's `crit=:mean`.
# ---------------------------------------------------------------------------
const GLMApprox = EPApprox

# legacy constructor GLMApprox(S, τ): an EPApprox over glm_model(S.X, S.y, S.lik, τ)
GLMApprox(S::GLMSites, τ::Real=.01) = EPApprox(glm_model(S.X, S.y, S.lik, Float64(τ)); qr=S.qr)

# legacy fit!(G, S, maxcycles, tol): sequential EP with the legacy criterion
function fit!(G::EPApprox, S::GLMSites, maxcycles=24, tol=.001)
    fit!(G; qr=S.qr, maxcycles=maxcycles, tol=tol, crit=:mean)
end

#for debugging
function check_consistency(G::EPApprox)
    C = cholesky(Symmetric(G.Q))
    @assert G.Σ ≈ C \ Matrix{Float64}(I,size(G.Q))
    @assert G.μ ≈ G.Σ*G.r
end

function logreg(X,y)
    ep_glm(X,y,Logit())
end
"""
    ep_glm(X,y,l :: Likelihood;max_cycles=20,tol=1e-5,τ=.01,nquad=20)

Estimate a GLΜ using Expectation Propagation.

Thin wrapper over the new EP core: builds `glm_model(X, y, l, τ)` and runs
`ep(...; qr=QuadRule(nquad), maxcycles=max_cycles, tol=tol, crit=:mean)`
(`crit=:mean` is the legacy convergence criterion, mean|Δμ| per cycle), so
there is a single EP implementation. Returns the `EPApprox` (aliased
`GLMApprox`).

# Arguments

- X : design matrix, dimension p × n, where n is the number of observations ("examples"). If you want to include an intercept you should add it yourself.
- y : observations, a vector of dimension n. 
- l : likelihood for the observations. Logit() for Bernoulli data with logistic link. PoisLik() for Poisson likelihood with exponential link. See src/likelihoods.jl for more
- max_cycles: max. number of times the EP algorithm can go through the observations.
- tol: tolerance. Iterations stop when the estimated mean changes less than the tolerance.
- τ: prior precision. The prior for the coefficients is βᵢ ∼ N(0,1/τ) for i in 1 … p. Set to τ=:autotune to set τ to its approximate maximum likelihood value.
- nquad: number of quadrature nodes to use when computing approximate moments. Default, 20, increase if you encounter stability issues. More nodes: slower.
"""
function ep_glm(X,y,l :: Likelihood;max_cycles=20,tol=1e-5,τ=.01,nquad=20)
    qr = QuadRule(nquad)
    if τ == :autotune
        S = GLMSites(X,y,l,qr)
        τ_opt,G=_autotune_tau(S,10,.01)
        @info "Optimal value of prior precision τ : $(τ_opt)"
    else
        G = ep(glm_model(X, y, l, τ); qr=qr, maxcycles=max_cycles, tol=tol, crit=:mean)
    end
    G
end

function loglik(S::GLMSites,f)
    s = 0.0
    m = S.X'*f
    for i in 1:nsites(S)
        s+=loglik(S,i,m[i])
    end
    s
end


function logpost(S::GLMSites,Q0 :: Matrix,f)
    s = 0.0
    m = S.X'*f
    for i in 1:nsites(S)
        s+=loglik(S,i,m[i])
    end
    s-((1/2)*f'*Q0*f)
end

function is_correction(G::GLMApprox,S :: GLMSites,ns,γ=1.1)
    xx,lw=importance_sampling(G,S,ns,γ)
    ee=ess(lw)
    @info "ESS: $(ee), ESS/n: $(ee/ns)"
    w = softmax(lw)
    m=xx*w
    S=(xx .- m)*Diagonal(w)*(xx .- m)'
    logz = logsumexp(lw) - log(ns)
    (logz,m,S)
end

function importance_sampling(G::GLMApprox,S :: GLMSites,ns,γ=1.1)
    lw = zeros(ns)
    N = MultivariateNormal(mean(G),γ*Symmetric(cov(G)))
    #prior covariance (τ·Q0)⁻¹ via Cholesky (effective prior precision is st.τ·Q0)
    Q0 = G.state.τ .* Matrix{Float64}(G.model.Q0)
    C0 = cholesky(Symmetric(Q0))
    P = MultivariateNormal(zeros(dim(G)),Symmetric(C0 \ Matrix{Float64}(I,dim(G),dim(G))))
    z = rand(N,ns)
    for i in 1:ns
        lw[i] = loglik(S,z[:,i]) + logpdf(P,z[:,i]) - logpdf(N,z[:,i])
    end
    (z,lw)
end

#Effective Sample Size for Importance Sampling
function ess(lw)
    exp(log_ess(lw))
end

function log_ess(lw)
    2*logsumexp(lw)-logsumexp(2*lw)
end

#For debugging - log ML for Gaussian data (GLM sites have A_i = X[:,i]', so the
#p×n design matrix is recovered from the sites)
function log_ml_gaussian(G::GLMApprox,y)
    M = G.model
    X = reduce(hcat, [s.A' for s in M.sites])
    n = length(y)
    Q0 = G.state.τ .* Matrix{Float64}(M.Q0)
    N = MultivariateNormal(zeros(n),Symmetric(X'*(Q0\X)+I))
    logpdf(N,y)
end



# Regularisation path over τ on the new core, with the legacy signature:
# returns (τs ascending, μs, logmls). One model (unscaled base Q0 = I) and one
# EPApprox are built for the whole path; the grid point τ is moved in place
# with set_tau! (one refactorisation, no per-site loop) and the site
# parameters are warm-started across grid points.
"""
    _regpath_tau(S::GLMSites, τ0, τ1; nsteps=10, maxcycles=24, tol=1e-3, qr=S.qr,
                 batch=0, refresh=1, ess_min=0.0, crit=:rel, verbose=true)

Legacy-signature regularisation path over τ on the new EP core: returns
`(τs, μs, logmls)` with `τs` **ascending**. The `fit!` keywords `batch`,
`refresh`, `ess_min`, `crit` and `verbose` are accepted with the same
defaults as `fit!` and forwarded to every `fit!` call along the path.
"""
function _regpath_tau(S,τ0,τ1;nsteps=10,maxcycles=24,tol=1e-3,qr=S.qr,
                      batch::Int=0,refresh::Int=1,ess_min::Float64=0.0,
                      crit::Symbol=:rel,verbose::Bool=true)
    m = size(S.X,1)
    μs = zeros(m,nsteps)
    τs = exp.(range(log(τ0),log(τ1),nsteps))
    lz = zeros(nsteps)
    M = glm_model(S.X, S.y, S.lik, 1.0)
    G = EPApprox(M; qr=qr)
    for (i,τ) in enumerate(τs)
        set_tau!(G,τ)
        fit!(G; qr=qr, maxcycles=maxcycles, tol=tol, crit=crit, batch=batch,
             refresh=refresh, ess_min=ess_min, verbose=verbose)
        μs[:,i] = mean(G)
        lz[i] = log_ml(G)
    end
    τs,μs,lz
end

# Empirical-Bayes selection of τ on the new core, legacy signature: scans the
# grid (warm-started, τ moved in place with set_tau!) and refits at the τ
# maximising the EP log marginal likelihood. Returns (τ_opt=..., G=...).
"""
    _autotune_tau(S::GLMSites, τ0, τ1; nsteps=10, maxcycles=24, tol=1e-3, qr=S.qr,
                  batch=0, refresh=1, ess_min=0.0, crit=:rel, verbose=true)

Legacy-signature autotune over τ on the new EP core: returns
`(τ_opt=τ_opt, G=G)`. The `fit!` keywords `batch`, `refresh`, `ess_min`,
`crit` and `verbose` are accepted with the same defaults as `fit!` and
forwarded to every `fit!` call.
"""
function _autotune_tau(S,τ0,τ1;nsteps=10,maxcycles=24,tol=1e-3,qr=S.qr,
                       batch::Int=0,refresh::Int=1,ess_min::Float64=0.0,
                       crit::Symbol=:rel,verbose::Bool=true)
    τs = exp.(range(log(τ0),log(τ1),nsteps))
    M = glm_model(S.X, S.y, S.lik, 1.0)
    G = EPApprox(M; qr=qr)
    set_tau!(G,τ0)
    fit!(G; qr=qr, maxcycles=maxcycles, tol=tol, crit=crit, batch=batch,
         refresh=refresh, ess_min=ess_min, verbose=verbose)
    lz = log_ml(G)
    τ_opt=τ0
    for τ in τs
        set_tau!(G,τ)
        fit!(G; qr=qr, maxcycles=maxcycles, tol=tol, crit=crit, batch=batch,
             refresh=refresh, ess_min=ess_min, verbose=verbose)
        nz = log_ml(G)
        if nz > lz
            lz = nz
            τ_opt = τ
        end
    end
    set_tau!(G,τ_opt)
    fit!(G; qr=qr, maxcycles=maxcycles, tol=tol, crit=crit, batch=batch,
         refresh=refresh, ess_min=ess_min, verbose=verbose)
    (τ_opt=τ_opt,G=G)
end

# ---------------------------------------------------------------------------
# Log-density of the MODEL (the target EP approximates), not of the EP
# approximation. The model is
#   p(x) ∝ exp(-.5 x'Q0 x + r0'x) ∏ᵢ fᵢ(Aᵢx),   fᵢ(η) = exp(log_dens(likᵢ, η, yᵢ)),
# so, up to an additive constant,
#   log p(x) = -.5 x'Q0 x + r0'x + Σᵢ log_dens(likᵢ, Aᵢx, yᵢ).
# Q0 keeps its own type (Matrix / Diagonal / SparseMatrixCSC), so a sparse
# prior precision is not densified. ForwardDiff-compatible: x may carry
# duals, so ForwardDiff.gradient(x -> logdensity(M, x), x0) works.
# ---------------------------------------------------------------------------
"""
    logdensity(M::EPModel, x::AbstractVector)

Log-density of the model at `x`, up to an additive constant:
`p(x) ∝ exp(-.5 x'Q0 x + r0'x) ∏ᵢ fᵢ(Aᵢx)` with `fᵢ(η) = exp(log_dens(likᵢ, η, yᵢ))`,
i.e. `log p(x) = -.5 x'Q0 x + r0'x + Σᵢ log_dens(likᵢ, Aᵢx, yᵢ)`.
`Q0` keeps its own type, so sparse/diagonal prior precisions stay sparse.
`x` may carry ForwardDiff duals, so `ForwardDiff.gradient(x -> logdensity(M, x), x0)`
works.
"""
function logdensity(M::EPModel, x::AbstractVector)
    lq = -0.5 * dot(x, M.Q0 * x) + dot(M.r0, x)
    for s in M.sites
        η = size(s.A, 1) == 1 ? (s.A * x)[1] : s.A * x   # scalar η for rank-1 sites
        lq += log_dens(s.lik, η, s.y)
    end
    return lq
end
