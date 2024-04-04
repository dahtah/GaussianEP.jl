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

#Signals that the moments are computed analytically, i.e.
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


function compute_moments!(H :: HybridDistr{D}, qm ::QuadratureMoments{Tf,D}, i) where D where Tf
    n = nnodes(qm)
    unwhiten_quad!(H,qm)
    z = 0.0
    #f = Base.Fix2(qm.f,ind)
    x = @MVector zeros(D)
    H.Nh.μ .= 0.0
    H.Nh.Σ .= 0.0
    @inbounds for i in 1:n
        x .=  nodes_buffer(qm)[:,i] + H.Nm.μ
        s = qm.f(x,i)*weights(qm)[i]
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

function Base.show(io::IO, qm::QuadratureMoments{F,D}) where F where D
    println("Quadrature rule for moment computation in dimension $(D). Number of nodes $(nnodes(qm))")
end



struct GenericSites{Tf,Th}
    f :: Tf
    A :: LinearMaps{Th}
end

function compute_moments!(H:: HybridDistr,S :: GenericSites{Tf,Th},i) where Tf where Th
    compute_moments!(H,S.f,i)
end


function GenericSites(f,A :: LinearMaps{Th}) where Th
    GenericSites{typeof(f),Th}(f,A)
end

function dim(GS :: GenericSites)
    dim(GS.A)
end

function nsites(GS :: GenericSites)
    length(GS.A) :: Int
end

function npred(GS :: GenericSites)
    size(GS.A[1],2)
end

function Base.show(io::IO, qm::GenericSites{F,D}) where F <: QuadratureMoments where D
    println("Sites representation for EP with moment computation using quadrature. Number of sites $(nsites(qm)) ")
end

function Base.show(io::IO, qm::GenericSites{F,D}) where F <: AnalyticMoments where D
    println("Sites representation for EP with analytic moment computation. Number of sites $(nsites(qm)) ")
end



#D-dimensional Gaussian using StaticArrays
mutable struct StaticGaussian{D,M}
    Σ :: MArray{Tuple{D,D},Float64,2,M}
    L :: Cholesky{Float64,MMatrix{D,D,Float64,M}}
    Q :: MArray{Tuple{D,D},Float64,2,M}
    μ :: MArray{Tuple{D},Float64,1,D}
    r :: MArray{Tuple{D},Float64,1,D}
end

function StaticGaussian{D}() where D
    Σ = @MArray zeros(D,D)
    for i in 1:D
        Σ[i,i] = 1.0
    end
    L = cholesky(Σ)
    Q = @MArray zeros(D,D)
    μ = @MArray zeros(D)
    r = @MArray zeros(D)
    StaticGaussian{D,D*D}(Σ,L,Q,μ,r)
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
mutable struct HybridDistr{D}
    Nm :: StaticGaussian{D}
    Nh :: StaticGaussian{D}
    zm :: Float64
    zh :: Float64
end

function HybridDistr{D}() where D
    Nm = StaticGaussian{D}()
    Nh = StaticGaussian{D}()
    HybridDistr{D}(Nm,Nh,0.0,0.0)
end


