
#D-dimensional Gaussian using StaticArrays
mutable struct StaticGaussian{D,M,F}
    Σ :: MArray{Tuple{D,D},F,2,M}
    L :: Cholesky{F,MMatrix{D,D,F,M}}
    Q :: MArray{Tuple{D,D},F,2,M}
    μ :: MArray{Tuple{D},F,1,D}
    r :: MArray{Tuple{D},F,1,D}
end

function StaticGaussian{D,F}() where D where F
    Σ = @MArray zeros(F,D,D)
    for i in 1:D
        Σ[i,i] = 1.0
    end
    L = cholesky(Σ)
    Q = @MArray zeros(F,D,D)
    μ = @MArray zeros(F,D)
    r = @MArray zeros(F,D)
    S=StaticGaussian{D,D*D,F}(Σ,L,Q,μ,r)
    exp_from_moments!(S)
    S
end

#Compute log-partition function from exponential parameters
function log_partition(N :: StaticGaussian{D,M,F}) where D where M where F
    C=cholesky(Symmetric(N.Q))
    .5*(D*log(2π)-logdet(C)+N.r'*(C\N.r))
end


function Base.show(io::IO, N::StaticGaussian{D}) where D
    println("Gaussian distribution in dimension $(D). μ=$(N.μ), Σ=$(N.Σ), Q = $(N.Q), r = $(N.r)")
end

function moments_from_exp!(N  :: StaticGaussian)
    N.Σ .= inv(N.Q)
    N.L = cholesky(Symmetric(N.Σ))
    N.μ .= N.Σ*N.r
end

function exp_from_moments!(N :: StaticGaussian)
    N.Q .= inv(N.Σ)
    N.r .= N.Q*N.μ
end


#some support for computations on marginals
#to ensure minimal allocations during updates
mutable struct HybridDistr{D,M,F}
    Nm :: StaticGaussian{D,M,F}
    Nh :: StaticGaussian{D,M,F}
    logzm :: F
    logzh :: F
end

function HybridDistr{D,F}() where D where F
    Nm = StaticGaussian{D,F}()
    Nh = StaticGaussian{D,F}()
    HybridDistr{D,D*D,F}(Nm,Nh,F(0.0),F(0.0))
end


abstract type AbstractLinearMaps{Tf,Tm} end

struct UnivariateLinearMaps{Tf,Tm <: AbstractMatrix{Tf}} <: AbstractLinearMaps{Tf,Tm}
    H :: Tm;
end

function Base.show(io::IO, M::UnivariateLinearMaps)
    println("A collection of $(length(M)) linear maps from ℜ^$(indim(M)) to ℜ, represented as a matrix")
end

indim(M :: UnivariateLinearMaps) = size(M.H,1)
outdim(M :: UnivariateLinearMaps) =    1

function Base.getindex(LM :: UnivariateLinearMaps,i)
    reshape(LM.H[:,i],1,indim(LM))
end

function Base.setindex!(LM :: UnivariateLinearMaps,v,i)
    LM.H[:,i] = v
end

function Base.length(LM :: UnivariateLinearMaps)
    size(LM.H,2)
end


struct LinearMaps{Tf,Tm <: AbstractMatrix{Tf}}  <: AbstractLinearMaps{Tf,Tm}
    H :: Vector{Tm}
end

indim(LM :: LinearMaps) = size(LM.H[1],2)
outdim(LM :: LinearMaps) = size(LM.H[1],1)

function Base.getindex(LM :: LinearMaps,i)
    LM.H[i]
end
function Base.setindex!(LM :: LinearMaps,v,i)
    LM.H[i] = v
end

function Base.length(LM :: LinearMaps)
    length(LM.H)
end

#Signals that the moments are computed analytically
struct AnalyticMoments{Tf}
    f :: Tf
end


#Signals that the moments are computed via quadrature
struct QuadratureMoments{Tf,D}
    f :: Tf
    qr :: QuadRule{D}
end

function nodes(qm :: QuadratureMoments)
    qm.qr.xq
end



function nodes_buffer(qm :: QuadratureMoments)
    qm.qr.xbuf
end

function weights(qm :: QuadratureMoments)
    qm.qr.wq
end

function nnodes(qm :: QuadratureMoments)
    length(weights(qm))
end



function QuadratureMoments(f,d,n::Integer,method=:gh)
    qr = QuadRule(d,n,method)
    QuadratureMoments{typeof(f),d}(f,qr)
end

function unwhiten_quad!(H :: HybridDistr{D},qm) where D
    nodes_buffer(qm) .= chol_lower(H.Nm.L)*nodes(qm)
end

function compute_moments!(H :: HybridDistr{D}, am ::AnalyticMoments{Tf}, ind) where D where Tf
    z,m,C=am.f(H.Nm.μ,H.Nm.Σ)
    H.Nh.μ .= m
    H.Nh.Σ .= C
    H.logzh = log(z)
end


function compute_moments!(H :: HybridDistr{D,M,Ht}, qm ::QuadratureMoments{Tf,D}, ind) where D where Tf where M where Ht
    n = nnodes(qm)
    #unwhiten_quad!(H,qm)
    z = 0.0
    #f = Base.Fix2(qm.f,ind)
    x = @MVector zeros(Ht,D)
    H.Nh.μ .= 0.0
    H.Nh.Σ .= 0.0
    @inbounds for i in 1:n
        #x .=  nodes_buffer(qm)[:,i] + H.Nm.μ
        x .= chol_lower(H.Nm.L)*nodes(qm)[:,i] + H.Nm.μ
        s = qm.f(x,ind)*weights(qm)[i]
        z += s
        H.Nh.μ .+= s*x
        for j in 1:D
            for k in 1:D
                H.Nh.Σ[j,k] += s*x[j]*x[k]
            end
        end
    end
    H.logzh = log(z)
    H.Nh.μ ./= z
    for i in 1:D
        for j in 1:D
            H.Nh.Σ[i,j] = H.Nh.Σ[i,j]/z - H.Nh.μ[i]*H.Nh.μ[j]
        end
    end
    return
end

function Base.show(io::IO, qm::QuadratureMoments{F,D}) where F where D
    println("Quadrature rule for moment computation in dimension $(D). Number of nodes $(nnodes(qm))")
end


#Holds info on moment computation and per-site linear maps
struct GenericSites{Tf,Ta <: AbstractLinearMaps}
    mc :: Tf
    A :: Ta
end


function compute_moments!(H:: HybridDistr,S :: GenericSites{Tf,Th},i) where Tf where Th
    compute_moments!(H,S.mc,i)
end


# function GenericSites(f,A :: LinearMaps{Th}) where Th
#     GenericSites{typeof(f),Th}(f,A)
# end

outdim(GS :: GenericSites) = outdim(GS.A)


function nsites(GS :: GenericSites)
    length(GS.A) :: Int
end

function npred(GS :: GenericSites)
    indim(GS.A)
end

function Base.show(io::IO, qm::GenericSites{F,D}) where F <: QuadratureMoments where D
    println("Sites representation for EP with moment computation using quadrature. Number of sites $(nsites(qm)) ")
end

function Base.show(io::IO, qm::GenericSites{F,D}) where F <: AnalyticMoments where D
    println("Sites representation for EP with analytic moment computation. Number of sites $(nsites(qm)) ")
end



