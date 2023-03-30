abstract type AbstractSites end

#for debugging purposes
struct GLik <: Likelihood
end



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
    GLMSites{eltype(X),Bool,BernLik}(X,y,BernLik(),qr)
end

function GLMSites(X,y,lik::Likelihood)
    @assert length(y) == size(X,2)
    qr = QuadRule(20)
    GLMSites{eltype(X),Bool,typeof(lik)}(X,y,lik,qr)
end

#Link function - move to struct
Φ = (x)->1/(1+exp(-x))

function log_dens(bernoulli::BernLik, f, y)
    return y ? log(Φ(f)) : log(1.0 - Φ(f)) 
end

#Poisson observation with exponential link function
function log_dens(poisson::PoisLik, f, y)
    return y*f - exp(f) - lgamma(1.0 + y)
end

function log_dens(glik::GLik, f, y)
    return -.5*(f-y)^2 - .5*log(2π)
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
