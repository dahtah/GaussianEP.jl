# #dumb type for now
# struct MyPDMat{T} <: AbstractPDMat{T}
#     S :: Matrix{T}
# end
# function Base.:*(M::MyPDMat, x::AbstractVecOrMat)
#     M.S*x
# end
# function Base.:/(M::MyPDMat, x::AbstractVecOrMat)
#     M.S/x
# end
# function Base.size(M :: MyPDMat)
#     size(M.S)
# end
# function Base.getindex(M :: MyPDMat,i,j)
#     M.S[i,j]
# end
# function Base.setindex!(M :: MyPDMat,v,i,j)
#     M.S[i,j] = v
# end
# function Base.inv(M :: MyPDMat)
#     inv(M.S)
# end

# #Add a A*D*A' term to matrix
# function addlowrank!(M :: MyPDMat,A,D)
#     BLAS.gemm!('N','N',1.,A,D*A',1.,M.S)
# end


function addlowrank!(M :: Matrix{Float64},A,D)
    BLAS.gemm!('N','N',1.,A,D*A',1.,M)
end


#A generalisation of the GLM posterior of the form
#q(x) ∝ exp(-.5 x' Q x + r'*x)
#where Q = Q₀ +  ∑ A_i' H_i A_i
#Each site is assumed to be of the form
#f(x) = g(A_i x)
#where A_i is a matrix of dimension D x n and D is small
struct GenericApprox{Tsig <: AbstractMatrix,Tm,Th,Ts} <: AbstractGaussApprox
    Σ :: Tsig
    μ :: Vector{Tm}
    r :: Vector{Tm}
    logz :: Vector{Tm}
    R :: Matrix{Tm}
    H :: Th
    S :: Ts
end

sites(G :: GenericApprox) = G.S
nsites(G :: GenericApprox) = nsites(sites(G))

function linearshift(G :: GenericApprox)
    G.Σ\G.μ
end

function GenericApprox(S:: GenericSites,τ =0.01)
    m = npred(S)
    GenericApprox(S,τ*Matrix(I,m,m))
end

function GenericApprox(S :: GenericSites,Q0 :: AbstractMatrix)
    n = nsites(S)
    m = npred(S)
    d = dim(S)
    #Q0=τ*Matrix(I,m,m)
    Σ=inv(Q0)
    μ=zeros(m)
    r=zeros(m)
    logz=zeros(n)
    R=zeros(d,n)
    H=LinearMaps([zeros(d,d) for _ in 1:n])
    GenericApprox(Σ,μ,r,logz,R,H,S)
end

function Statistics.mean(G :: GenericApprox)
    copy(G.μ)
end
Statistics.cov(G :: GenericApprox) = G.Σ

#naive implementation
function cavity(G :: GenericApprox,i)
    Qc = inv(Symmetric(G.Σ)) - G.S.A[i]'*G.H[i]*G.S.A[i]
    mc = Qc\(linearshift(G) - G.S.A[i]'*G.R[:,i])
    MvNormal(mc,inv(Symmetric(Qc)))
    #mv,Qc
end


function cavity_marginal(G :: GenericApprox,i)
    Sm = G.S.A[i]*G.Σ*G.S.A[i]'
    mm = G.S.A[i]*G.μ
    Qm = inv(Sm)
    rm = Qm*mm
    S = inv(Qm - G.H[i])
    r = rm - G.R[:,i]
    MvNormal(S*r,S)
    #mv,Qc
end

function compute_site_contribution(G::GenericApprox,i)
    mvc = cavity_marginal(G,i)
    Qc = inv(mvc.Σ)
    rc = Qc*mvc.μ
    mm=TiltedGaussians.moments(mvc,G.S.qr,Base.Fix2(G.S.f,i))
    Qh = inv(mm.C)
    δQ = Qh-Qc
    rh = Qh*mm.m
    δr = rh - rc
    δz = log_partition(Qh,rh)-log_partition(Qc,rc)
    (δz=δz,δr=δr,δQ=δQ)
end


function compute_site_contribution_old(G::GenericApprox,i)
    mc = cavity(G,i)
    TiltedGaussians.contributions_ep(mc,G.S.qr,G.S.A[i],Base.Fix2(G.S.f,i))
end

function update!(G::GenericApprox,i,α=0.0)
    ct = compute_site_contribution(G,i)
    D=(1-α)*(ct.δQ - G.H[i])
    Z = Matrix(G.Σ*G.S.A[i]')

    G.H[i] = α*G.H[i] + (1-α)*ct.δQ
    dr = G.S.A[i]'*(1-α)*(ct.δr - G.R[:,i])
    G.R[:,i] = α*G.R[:,i]+ (1-α)*ct.δr
    addlowrank!(G.Σ,Z,-inv(inv(D)+G.S.A[i]*Z))
    G.r .+= dr
#    mp  =G.Σ*sum((G.S.A[i]'*(G.R[:,i]) for i in 1:nsites(G)))
    G.μ .= G.Σ*G.r
end

function run_ep!(G;α=0.0,npasses=4,schedule=1:nsites(G))
    for ip in 1:npasses
        for i in schedule
            update!(G,i,α)
        end
    end
end

function compute_update!(H :: HybridDistr, G :: GenericApprox,i;α=0.0)
    B = Matrix(G.Σ*G.S.A[i]')
    try
        compute_cavity_marginal!(H,G,i,B)
    catch
        @warn "Site $(i) has non-positive variance, skipping"
        return 
    end
    compute_moments!(H,G,i)
    H.Nh.Q .-= H.Nm.Q
    H.Nh.r .-= H.Nm.r
    D=(1-α)*(H.Nh.Q - G.H[i])
    G.H[i] = α*G.H[i] + (1-α)*H.Nh.Q
    dr = G.S.A[i]'*(1-α)*(H.Nh.r - G.R[:,i])
    G.R[:,i] = α*G.R[:,i]+ (1-α)*H.Nh.r
    addlowrank!(G.Σ,B,-inv(inv(D)+G.S.A[i]*B))
    G.r .+= dr
    #G.μ .= G.Σ*G.r
    BLAS.gemv!('N',1.0,G.Σ,G.r,0.0,G.μ)
    return
end

function compute_cavity_marginal!(H :: HybridDistr, G :: GenericApprox,i,buf)
    H.Nm.Q .= inv(Matrix(G.S.A[i]*buf)) - G.H[i]
    H.Nm.r .= inv(Matrix(G.S.A[i]*buf))*(G.S.A[i]*G.μ) - G.R[:,i]
    moments_from_exp!(H.Nm)
end


function chol_lower(a::Cholesky{F, T}) where F where T
    return a.uplo === 'L' ? LowerTriangular{F,T}(a.factors) : LowerTriangular{F,T}(a.factors')
end

function unwhiten_quad!(H :: HybridDistr{D},G) where D
    G.S.qr.xbuf .= chol_lower(H.Nm.L)*G.S.qr.xq
end

function compute_moments!(H :: HybridDistr{D},G :: GenericApprox,ind) where D
    n = TiltedGaussians.nnodes(G.S.qr)
    unwhiten_quad!(H,G)
    z = 0.0
    f = Base.Fix2(G.S.f,ind)
    x = @MVector zeros(D)
    H.Nh.μ .= 0.0
    H.Nh.Σ .= 0.0
    @inbounds for i in eachindex(G.S.qr.wq)
        x .=  G.S.qr.xbuf[:,i] + H.Nm.μ
        s = f(x)*G.S.qr.wq[i]
        z += s
        H.Nh.μ .+= s*x
        for j in 1:D
            for k in 1:D
                H.Nh.Σ[j,k] += s*x[j]*x[k]
            end
        end
    end
    H.Nh.μ ./= z
    for i in 1:D
        for j in 1:D
            H.Nh.Σ[i,j] = H.Nh.Σ[i,j]/z - H.Nh.μ[i]*H.Nh.μ[j]
        end
    end
    exp_from_moments!(H.Nh)
    return
end

#Compute contribution of site number i and store in H.Nh.Q and H.Nh.r 
function compute_site_contribution!(H :: HybridDistr, G :: GenericApprox,i)
    buf = G.Σ*G.S.A[i]'
    compute_cavity_marginal!(H,G,i,buf)
    compute_moments!(H,G,i)
    H.Nh.Q .-= H.Nm.Q
    H.Nh.r .-= H.Nm.r
    return
end
