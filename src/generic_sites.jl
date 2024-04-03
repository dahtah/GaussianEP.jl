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
    length(GS.A) :: Int
end

function npred(GS :: GenericSites)
    size(GS.A[1],2)
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


