
struct LinearMaps{T}
    H :: Vector{T}
end

#FIXME: check that all elements have the same dimensions
function LinearMaps(H :: AbstractVector)
    LinearMaps{eltype(H)}(H)
end

struct GenericSites{Tf,Th}
    f :: Tf
    H :: LinearMaps{T,Th}
end

function nsites(GS :: GenericSites)
    length(GS.h)
end

function npred(GS :: GenericSites)
    size(GS.h[1],2)
end

#A generalisation of the GLM posterior of the form
#q(x) ∝ exp(-.5 x' Q x + r'*x)
#where Q = Q₀ +  ∑ A_i' H_i A_i
#Each site is assumed to be of the form
#f(x) = g(A_i x)
#where A_i is a matrix of dimension D x n and D is small
struct GenericApprox{T,Ta,Th} <: AbstractGaussApprox
    Q0 :: Matrix{T}
    Q :: Matrix{T}
    A :: Ta
    Σ :: Matrix{T}
    μ :: Vector{T}
    logz :: Vector{T}
    r :: Vector{T}
    H :: Th
end

function GenericApprox(GS :: GenericSites,τ=.01)
    n = nsites(S)
    m = npred(S)
    Q0=τ*Matrix(I,m,m)
    

end

function Statistics.mean(G :: GenericApprox)
    copy(G.μ)
end
Statistics.cov(G :: GenericApprox) = G.Σ


