abstract type AbstractSites end

struct GLMSites{TX,TY,L <: Likelihood} <: AbstractSites
    X :: Matrix{TX}
    y :: Vector{TY}
    lik :: L
    qr :: QuadRule
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



function loglik(S :: GLMSites{TX,TY,L},i,x) where {TX, TY,L}
    log_dens(S.lik,x,S.y[i])
end

function hybrid_moments(S :: GLMSites,i,m,s2)
    xq = sqrt(s2)*S.qr.x .+ m
    wq = S.qr.w
    f = [exp(loglik(S,i,x)) for x in xq]
    z = dot(wq,f)
    wq /= z
    μ = dot(wq,f .* xq)
    σ2 = dot(wq,f .* (xq .- μ).^2)
    (z=z,μ=μ,σ2=σ2)
end
