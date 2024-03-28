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

function GenericApprox(S :: GenericSites,τ=.01)
    n = nsites(S)
    m = npred(S)
    d = dim(S)
    Q0=τ*Matrix(I,m,m)
    Σ=MyPDMat(inv(Q0))
    μ=zeros(m)
    logz=zeros(n)
    R=zeros(d,n)
    H=LinearMaps([zeros(d,d) for _ in 1:n])
    GenericApprox(Σ,μ,logz,R,H,S)
end

function Statistics.mean(G :: GenericApprox)
    copy(G.μ)
end
Statistics.cov(G :: GenericApprox) = G.Σ

#naive implementation
function cavity(G :: GenericApprox,i)
    Qc = inv(G.Σ) - G.S.A[i]'*G.H[i]*G.S.A[i]
    mc = Qc\(linearshift(G) - G.S.A[i]'*G.R[:,i])
    MvNormal(mc,inv(Qc))
end

function compute_site_contribution(G::GenericApprox,i)
    mc = cavity(G,i)
    TiltedGaussians.contributions_ep(mc,G.S.qr,G.S.A[i],Base.Fix2(G.S.f,i))
end

function update!(G::GenericApprox,i)
    ct = compute_site_contribution(G,i)
    Qn = inv(G.Σ) + G.S.A[i]'*(ct.δQ - G.H[i])*G.S.A[i]
    G.H[i] = ct.δQ
    rn = linearshift(G) + G.S.A[i]'*(ct.δr - G.R[:,i])
    G.R[:,i] = ct.δr
    S = inv(Qn)
    G.Σ .= S
    G.μ .= S*sum((G.S.A[i]'*(G.R[:,i]) for i in 1:nsites(G)))
end
