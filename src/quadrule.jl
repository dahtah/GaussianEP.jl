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

"""
    trap_rule(L=8.0, N=401)

Trapezoid rule on the truncated range `[-L, L]` for standard-normal
integrals: uniform nodes `xᵢ`, weights `wᵢ = φ(xᵢ)·h` (trapezoid endpoints
halved), normalised so `sum(w) = 1`. Returns a `QuadRule`, so it plugs into
`tilted_moments` / `ep(...; qr=...)` for rank-1 and tensor (rank ≤ 3) sites
alike. Unlike Gauss–Hermite (exact only for polynomial × Gaussian
integrands) this handles bounded / non-smooth factors (kinks, near-hard
constraints) gracefully, at the cost of more nodes for smooth integrands
(Trefethen, arXiv:2101.09501 §5; status.md open question 7).
"""
function trap_rule(L::Real=8.0, N::Int=401)
    N >= 3 || throw(ArgumentError("N >= 3 required"))
    r = range(-L, L, N)
    x = collect(r)
    h = step(r)
    w = exp.(-0.5 .* x .^ 2) ./ sqrt(2π) .* h
    w[1] *= 0.5
    w[end] *= 0.5
    w ./= sum(w)          # normalise: exact for constants, compensates truncation
    QuadRule(x, w, log.(w))
end
