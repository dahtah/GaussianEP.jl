Distributions.MultivariateNormal(G :: AbstractGaussApprox) = MultivariateNormal(mean(G),Symmetric(cov(G)))

function logreg(X,y)
    ep_glm(X,y,BernLik())
end

function ep_glm(X,y,l :: Likelihood,max_cycles=20,tol=1e-5)
    S=GLMSites(X,y,l)
    G=GLMApprox(S)
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
