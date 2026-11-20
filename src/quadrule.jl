"""
    QuadRule(deg)

Gauss–Hermite quadrature rule of degree `deg` for standard-normal integrals:
nodes `x` and weights `w` with `sum(w .* f.(x)) ≈ E[f(Z)]` for `Z ~ N(0,1)`.
"""
struct QuadRule
    x :: Vector{Float64}
    w :: Vector{Float64}
    logw :: Vector{Float64}   # precomputed log(w), for allocation-free moment loops
end

degree(qr :: QuadRule) = length(qr.w)
#Gauss-Hermite rule of deg. d for N(0,1) 
function QuadRule(deg)
    x,w=gausshermite(deg);
    x = sqrt(2)*x; w = w/sqrt(π)
    QuadRule(x, w, log.(w))
end
