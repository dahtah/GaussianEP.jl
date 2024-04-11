Distributions.MultivariateNormal(G :: AbstractGaussApprox) = MultivariateNormal(mean(G),Symmetric(cov(G)))

function logreg(X,y)
    ep_glm(X,y,Logit())
end
"""
    ep_glm(X,y,l :: Likelihood;max_cycles=20,tol=1e-5,τ=.01,nquad=20)

Estimate a GLΜ using Expectation Propagation.

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
    qr = QuadRule(1,nquad,:gh)
    S=GLMSites(X,y,l,qr)
    if τ == :autotune
        τ_opt,G=_autotune_tau(S,10,.01)
        @info "Optimal value of prior precision τ : $(τ_opt)"
    else

        G=GLMApprox(S,τ)
        fit!(G,S,max_cycles,tol)
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

function is_correction(G :: GLMApprox,S :: GLMSites,ns,γ=1.1)
    xx,lw=importance_sampling(G,S,ns,γ)
    ee=ess(lw)
    @info "ESS: $(ee), ESS/n: $(ee/ns)"
    w = softmax(lw)
    m=xx*w
    S=(xx .- m)*Diagonal(w)*(xx .- m)'
    logz = logsumexp(lw) - log(ns)
    (logz,m,S)
end

function importance_sampling(G :: GLMApprox,S :: GLMSites,ns,γ=1.1)
    lw = zeros(ns)
    N = MultivariateNormal(mean(G),γ*Symmetric(cov(G)))
    P = MultivariateNormal(zeros(dim(G)),Symmetric(inv(G.Q0)))
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

#For debugging - log ML for Gaussian data
function log_ml_gaussian(G,y)
    n = length(y)
    N = MultivariateNormal(zeros(n),Symmetric(G.X'*(G.Q0\G.X)+I))
    logpdf(N,y)
end




function _regpath_tau(S,τ0,τ1;nsteps=10)
    m,n = size(S.X)
    G0=GLMApprox(S,τ0)
    μs = zeros(m,nsteps)
    lt = range(log(τ0),log(τ1),nsteps)
    τs = exp.(lt)
    lz = zeros(nsteps)
    for (i,τ) = enumerate(τs)
        set_tau!(G0,τ)
        GaussianEP.fit!(G0,S)
        μs[:,i] = mean(G0)
        lz[i] = log_ml(G0)
    end
    τs,μs,lz
end

function _autotune_tau(S,τ0,τ1;nsteps=10)
    m,n = size(S.X)
    G0=GLMApprox(S,τ0)
    GaussianEP.fit!(G0,S)
    μ = mean(G0)
    C = cov(G0)
    lt = range(log(τ0),log(τ1),nsteps)
    τs = exp.(lt)
    lz = log_ml(G0)
    τ_opt=τ0
    for (i,τ) = enumerate(τs)
        set_tau!(G0,τ)
        GaussianEP.fit!(G0,S)
        nz = log_ml(G0)
        if nz > lz
            lz = nz
            μ = mean(G0)
            C = cov(G0)
            τ_opt = τ
        end
    end
    set_tau!(G0,τ_opt)
    GaussianEP.fit!(G0,S)
    (τ_opt=τ_opt,G=G0)
end


