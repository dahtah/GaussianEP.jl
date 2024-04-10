abstract type AbstractSites end

struct GLMSites{TX,TY,L <: Likelihood} <: AbstractSites
    X :: Matrix{TX}
    y :: Vector{TY}
    lik :: L
    qr :: QuadRule{1}
end

nsites(S :: GLMSites) = length(S.y)
npred(S :: GLMSites) = size(S.X,1)
function Base.show(io::IO, S::GLMSites{TX,TY,L}) where {TX, TY,L}
    println("GLM with $(npred(S)) predictors and $(nsites(S)) observations of type $(L)  ")
end

function GLMSites(X,y :: BitVector)
    @assert length(y) == size(X,2)
    qr = QuadRule(20)
    GLMSites{eltype(X),Bool,Logit}(X,y,Logit(),qr)
end

function GLMSites(X,y,lik::Likelihood)
    @assert length(y) == size(X,2)
    qr = QuadRule(20)
    GLMSites{eltype(X),eltype(y),typeof(lik)}(X,y,lik,qr)
end

function GLMSites(X,y,lik::Likelihood,qr::QuadRule)
    @assert length(y) == size(X,2)
    GLMSites{eltype(X),eltype(y),typeof(lik)}(X,y,lik,qr)
end



function loglik(S :: GLMSites{TX,TY,L},i,x) where {TX, TY,L}
    log_dens(S.lik,x,S.y[i]) :: Float64
end


function hybrid_moments(S :: GLMSites,i,m,s2)
    z=0.0
    m1=0.0
    m2=0.0
    @inbounds for j in 1:length(S.qr.xq)
        xq = sqrt(s2)*S.qr.xq[j] + m
        wq = S.qr.wq[j]
        f = exp(loglik(S,i,xq))*wq
        z+=f
        tmp=f*xq
        m1+=tmp
        m2+=tmp*xq
    end
    μ = m1/z
    σ2 = m2/z - μ^2 
    (z=z,μ=μ,σ2=σ2)
end


function hybrid_moments_check(S :: GLMSites,i,m,s2)
    xq = sqrt(s2)*S.qr.xq .+ m
    wq = S.qr.wq
    f = [exp(loglik(S,i,x)) for x in xq]
    z = dot(wq,f)
    wq /= z
    μ = dot(wq,f .* xq)
    σ2 = dot(wq,f .* (xq .- μ).^2)
    (z=z,μ=μ,σ2=σ2)
end
