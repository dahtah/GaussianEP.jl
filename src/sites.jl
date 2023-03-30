abstract type AbstractSites end

struct GLMSites{TX,TY} <: AbstractSites
    X :: Matrix{TX}
    y :: Vector{TY}
    lik :: Vector{Likelihood}
    qr :: QuadRule
end

function GLMSites(X,y :: BitVector)
    qr = QuadRule(20)
    l = Vector{Likelihood}();

    for _ in 1:size(X,2)
        push!(l,BernLik())
    end
    GLMSites{eltype(X),Bool}(X,y,l,qr)
end

function logdens(s :: BernLik(),)

function loglik(S :: GLMSites,i,x)
    GaussianProcesses.log_dens(S.lik[i],[x],[S.y[i]])[1] :: Float64
end


nsites(S :: GLMSites) = length(S.y)

function hybrid_moments(S :: GLMSites,i,m,s2)
    xq = sqrt(s2)*S.qr.x .+ m
    wq = S.qr.w
    #@assert dot(wq,(xq .- m).^2) ≈ s2
    f = [exp(loglik(S,i,x)) for x in xq]
    z = dot(wq,f)
    wq /= z
    μ = dot(wq,f .* xq)
    σ2 = dot(wq,f .* (xq .- μ).^2)
    (z=z,μ=μ,σ2=σ2)
end
