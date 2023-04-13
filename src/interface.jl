Distributions.MultivariateNormal(G :: AbstractGaussApprox) = MultivariateNormal(mean(G),Symmetric(cov(G)))

function logreg(X,y)
    ep_glm(X,y,Logit())
end

function ep_glm(X,y,l :: Likelihood;max_cycles=20,tol=1e-5,τ=.01,nquad=20)
    qr = QuadRule(nquad)
    S=GLMSites(X,y,l,qr)
    G=GLMApprox(S,τ)
    fit!(G,S,max_cycles,tol)
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
