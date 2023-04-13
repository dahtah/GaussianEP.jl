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
    μ :: Vector{T}
    logz :: Vector{T}
    r :: Vector{T}
    h :: Vector{T}
end

function Base.show(io::IO, G::GLMApprox)
    println("Approx. posterior of GLM form, with $(size(G.X,2)) sites, in dimension $(size(G.X,1))")
end
function Statistics.mean(G :: GLMApprox)
    copy(G.μ)
end
Statistics.cov(G :: GLMApprox) = G.Σ
function GLMApprox(S :: GLMSites,τ=.01)
    n = nsites(S)
    m = npred(S)
    Q0=τ*Matrix(I,m,m)
    #Initialise via one run of parallel EP
    r=zeros(n)
    h=zeros(n)
    for i in 1:nsites(S)
        ms=GaussianEP.hybrid_moments(S,i,0,1)
        qi,ri=m2exp(ms.μ,ms.σ2)
        r[i] = ri
        h[i] = max(0.0,qi-1)
    end
    Q=Q0 + S.X*Diagonal(h)*S.X'
    Σ=inv(Q)
    μ=Σ*S.X*r

    logz=zeros(n)
    GLMApprox{Float64}(Q0,Q,S.X,Σ,μ,logz,r,h)
end

function dim(G :: GLMApprox{T}) where T
    size(G.Q,1)
end

function nsites(G :: GLMApprox{T}) where T
    size(G.X,2)
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

#Linear response for site i
function fitted(G :: GLMApprox,i)
    G.X[:,i]'*G.μ
end

function fitted(G :: GLMApprox)
    C.X'*G.μ
end

function cavity(G :: GLMApprox, i)
    xi = @view G.X[:,i]
    s = xi'*G.Σ*xi
    hi = G.h[i]
    α = (1-hi*s)
    sc = s+(hi*s^2 )/α
    #fitted: G.X[:,i]'*G.μ
    #m = fitted(G,i)
    m = xi'*G.μ
    γ = m-G.r[i]*s
    μ = γ*(α+G.h[i]*s)/α
    (μ=μ,σ2=sc)
end



#for debugging
function check_consistency(G :: GLMApprox)
    @assert G.Σ ≈ inv(G.Q)
    @assert G.μ ≈ G.Σ*G.X*G.r
end

function update_approx!(G :: GLMApprox{T},i,dq,dr,lz) where T <: Union{Float32,Float64}
    # @info "Top"
    # @assert G.μ ≈ G.Σ*G.X*G.r
    #check_consistency(G)
    G.logz[i] = lz;
    n = nsites(G)
    xi = @view G.X[:,i]
    Sx = G.Σ*xi
    xSx = dot(xi,Sx)
    #γ = -G.h[i] + dq
    δr = dr-G.r[i]
    δq = dq-G.h[i]
    α = -δq/(1+δq*xSx)
    G.μ .+=  (δr + α*dot(G.μ,xi)+δr*α*xSx)*Sx
    G.r[i] = dr
    G.h[i] = dq
    BLAS.ger!(δq,xi,xi,G.Q) #rank one update to Q
    #rank one update to Σ
    BLAS.ger!(α,Sx,Sx,G.Σ)
    return
end

function site_update(G :: GLMApprox,S :: GLMSites,i)
    cv = cavity(G,i)
    ms = hybrid_moments(S,i,cv.μ,cv.σ2)
    qc,rc=m2exp(cv.μ,cv.σ2)
    qp,rp=m2exp(ms.μ,ms.σ2) 
    dq = qp - qc
    dr = rp - rc
    δ=log_partition(qc,rc)-log_partition(qp,rp)
    ν=log(ms.z)+δ
    (dq=dq,dr=dr,lz=ν)
end

function natural_parameters(G:: GLMApprox)
    (G.Q0+G.X*Diagonal(G.h)*G.X',G.X*G.r)
end

function set_tau!(G :: GLMApprox,τ)
    m = dim(G)
    G.Q0 .= τ*Matrix(I,m,m)
    G.Q .= G.Q0 + G.X*Diagonal(G.h)*G.X'
    G.Σ .= inv(G.Q)
    G.μ .= G.Σ*G.X*G.r
end

function cavity_naive(G :: GLMApprox,i)
    h = copy(G.h)
    r = copy(G.r)
    h[i] = 0.0
    r[i] = 0.0
    (G.Q0+G.X*Diagonal(h)*G.X',G.X*r)
end



function fit!(G :: GLMApprox,S :: GLMSites,maxcycles=24,tol=.001)
    m = mean(G)
    for c in 1:maxcycles
        for i in 1:nsites(S)
            dq,dr,lz=site_update(G,S,i)
            #ν = lz-log_partition(natural_parameters(G)...)+log_partition(cavity_naive(G,i)...)
            update_approx!(G,i,dq,dr,lz)
        end
        delta= mean(abs.(m-mean(G)))
  #      @info delta
        (delta < tol ) && return
        m = mean(G)
    end
    @info "Could not reach tolerance level. 
Increase prior precision, or increase number of quadrature nodes"
    return 
end

function log_ml(G :: GLMApprox)
    log_partition(Symmetric(G.Q),G.X*G.r)-log_partition(Symmetric(G.Q0),zeros(dim(G))) + sum(G.logz)
end

#log ∫exp(-.5*x'Qx+r'x)dx
function log_partition(Q :: AbstractMatrix,r :: AbstractVector)
    @assert size(Q,1) == length(r)
    n = length(r)
    C=cholesky(Symmetric(Q))
    .5*(n*log(2π)-logdet(C)+r'*(C\r))
end

# function log_partition(Q :: AbstractMatrix,r :: AbstractVector)
#     log(partition(Q,r))
# end

function partition(Q :: AbstractMatrix,r :: AbstractVector)
    n = length(r)
    sqrt( ((2π)^n)/det(Q))*exp(.5*r'*(Q\r))
end

function parallel_update(G :: GLMApprox,S :: GLMSites)
    # μ = G.X'*mean(G)
    # v = diag(G.X'*G.Σ*G.X)
    n = nsites(G)
    dq = zeros(n)
    dr = zeros(n)
    lz = zeros(n)
    for i in 1:nsites(S)
        dq[i],dr[i],lz[i]=site_update(G,S,i)
        # ms=GaussianEP.hybrid_moments(S,i,μ[i],v[i])
        # qc,rc=m2exp(μ[i],v[i])
        # qp,rp=m2exp(ms.μ,ms.σ2) 
        # dq[i] = qp - qc
        # dr[i] = rp - rc
    end
    dq,dr,lz
end


function log_partition(q :: Real,r :: Real)
    .5*(log(2π)-log(q)+(1/q)*(r)^2)
end
