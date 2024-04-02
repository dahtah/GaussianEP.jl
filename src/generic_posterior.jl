#dumb type for now
struct MyPDMat{T} <: AbstractPDMat{T}
    S :: Matrix{T}
end
function Base.size(M :: MyPDMat)
    size(M.S)
end
function Base.getindex(M :: MyPDMat,i,j)
    M.S[i,j]
end
function Base.setindex!(M :: MyPDMat,v,i,j)
    M.S[i,j] = v
end
function Base.inv(M :: MyPDMat)
    inv(M.S)
end

#Add a A*D*A' term to matrix
function addlowrank!(M :: MyPDMat,A,D)
    BLAS.gemm!('N','N',1.,A,D*A',1.,M.S)
end


struct LinearMaps{T}
    H :: Vector{T}
end

function dim(LM :: LinearMaps)
    size(LM.H[1],1)
end
function Base.getindex(LM :: LinearMaps,i)
    LM.H[i]
end
function Base.setindex!(LM :: LinearMaps,v,i)
    LM.H[i] = v
end

function Base.length(LM :: LinearMaps)
    length(LM.H)
end


struct GenericSites{Tf,Th,D}
    f :: Tf
    A :: LinearMaps{Th}
    qr :: QuadRule{D}
end

function GenericSites(f,A :: LinearMaps,n::Integer,method=:gh)
    qr = QuadRule(dim(A),n,method)
    GenericSites(f,A,qr)
end

function dim(GS :: GenericSites)
    dim(GS.A)
end

function nsites(GS :: GenericSites)
    length(GS.A)
end

function npred(GS :: GenericSites)
    size(GS.A[1],2)
end

#A generalisation of the GLM posterior of the form
#q(x) ∝ exp(-.5 x' Q x + r'*x)
#where Q = Q₀ +  ∑ A_i' H_i A_i
#Each site is assumed to be of the form
#f(x) = g(A_i x)
#where A_i is a matrix of dimension D x n and D is small
struct GenericApprox{Tsig <: AbstractPDMat,Tm,Th,Ts} <: AbstractGaussApprox
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
    Σ=MyPDMat(inv(Q0))
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
    #rn = linearshift(G) + G.S.A[i]'*(1-α)*(ct.δr - G.R[:,i])
    dr = G.S.A[i]'*(1-α)*(ct.δr - G.R[:,i])
    G.R[:,i] = α*G.R[:,i]+ (1-α)*ct.δr
    addlowrank!(G.Σ,Z,-inv(inv(D)+G.S.A[i]*Z))
    G.r .+= dr
#    @assert G.r ≈ sum((G.S.A[i]'*(G.R[:,i]) for i in 1:nsites(G)))
    mp  =G.Σ*sum((G.S.A[i]'*(G.R[:,i]) for i in 1:nsites(G)))
    G.μ .= G.Σ*G.r
#    @show norm(G.μ - mp)
end

function run_ep!(G;α=0.0,npasses=4,schedule=1:nsites(G))
    for ip in 1:npasses
        for i in schedule
            update!(G,i,α)
        end
    end
end

