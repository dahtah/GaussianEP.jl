#Various representations for Gaussian posteriors
#that support the updates necessary for EP

#We need:
#A way to compute means and covariances of the form
#x'*μ, x'*Σ*x
#A way to compute site updates/downdates

abstract type AbstractGaussApprox end

#Appropriate for a GLM with p parameters and n observations
#p(z) ∝ exp(-.5*z'*Q*z + r'*z)
#with
#Q = Q0+X'*H*X with H = diag(h_1 \ldots h_n)
#(Q0 is the prior covariance)
#r = \sum r_i x_i
#NB: stores an explicit inverse for Q, updated using Woodbury lemma
struct GLMApprox{T} <: AbstractGaussApprox
    Q0 :: Matrix{T}
    Q :: Matrix{T}
    X :: Matrix{T}
    Σ :: Matrix{T}
    logz :: Vector{T}
    r :: Vector{T}
    h :: Vector{T}
end

function dim(G :: GLMApprox{T}) where T
    size(G.Q,1)
end

function nsites(G :: GLMApprox{T}) where T
    size(G.X,2)
end

function cavity(G :: GLMApprox{T}, i) where T
    xi = @view A.X[:,i]
    s = xi'*A.Σ*xi
    hi = A.h[i]
    α = (1-hi*s)
    sc = s+(hi*s^2 )/α
    m = dot(xi,mean(A))
    γ = m-A.r[i]*s
    μ = γ*(α+A.h[i]*s)/α
    (μ=μ,σ2=sc)
end


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

function update_approx!(G :: GLMApprox{T},i,dq,dr) where T <: Union{Float32,Float64}
    n = nsites(G)
    xi = G.X[:,i]
    γ = -G.h[i] + dq
    BLAS.ger!(γ,xi,xi,G.Q) #rank one update to Q
    G.r[i] = dr
    G.h[i] = dq
    α = -γ/(1+γ*xi'*G.Σ*xi)
    z = G.Σ*xi
    #rank one update to Σ
    BLAS.ger!(α,z,z,G.Σ)
    #    @assert A.Σ ≈ inv(A.Q)
    return
end

