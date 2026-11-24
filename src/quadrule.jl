"""
    QuadRule(deg)

Gauss–Hermite quadrature rule of degree `deg` for standard-normal integrals:
nodes `x` and weights `w` with `sum(w .* f.(x)) ≈ E[f(Z)]` for `Z ~ N(0,1)`.

For rank-`d > 1` sites a rule with `grid === nothing` is interpreted as the
full tensor product of `x`/`w` (see `_tilted_quad` in `moments.jl`). If
`grid !== nothing` (as produced by `smolyak_rule(level, d)` for `d ≥ 2`),
`grid` is an explicit `d × npts` matrix of nodes in standard-normal
coordinates with `npts = length(w)` (possibly signed) weights — a sparse
grid. `logw` is `log.(w)` for positive-weight rules and `log.(abs.(w))` for
signed sparse-grid weights.
"""
struct QuadRule
    x :: Vector{Float64}
    w :: Vector{Float64}
    logw :: Vector{Float64}   # precomputed log(w) (log|w| for signed sparse-grid weights), for allocation-free moment loops
    grid :: Union{Nothing, Matrix{Float64}}   # nothing = tensor product of x; else explicit d×npts sparse-grid nodes
end
QuadRule(x::Vector{Float64}, w::Vector{Float64}, logw::Vector{Float64}) = QuadRule(x, w, logw, nothing)

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

# 1-D rule at Smolyak level l: m(l) = 2^(l-1)+1 nodes, weights summing to 1.
#   base=:gh   -> Gauss–Hermite for N(0,1) with m(l) nodes (exact for
#                 polynomials of degree ≤ 2^l+1)
#   base=:trap -> nested dyadic trapezoid rule on [-L, L] (trap_rule(L, m(l)),
#                 m(1) = 3 since trap_rule needs N ≥ 3); nested, so nodes are
#                 shared between levels and the sparse grid is much smaller
function _smolyak_1d(l::Int, base::Symbol, L::Real)
    n = base === :gh ? 2^(l-1) + 1 : max(3, 2^(l-1) + 1)
    if base === :gh
        x, w = gausshermite(n)
        (sqrt(2)*x, w/sqrt(π))
    else
        qr = trap_rule(L, n)
        (qr.x, qr.w)
    end
end

"""
    smolyak_rule(level, d; base=:gh, L=8.0)

Smolyak sparse-grid quadrature rule for standard-normal integrals in `d`
dimensions, returned as a `QuadRule` so it plugs into `tilted_moments` /
`ep(...; qr=...)` for rank-`d` sites (`d = 2, 3`) with **no changes at call
sites** — the rank-`d` quadrature path in `moments.jl` detects the sparse grid
via `qr.grid !== nothing`.

Construction: 1-D rules `U_l` at levels `l = 1..level` with `m(l) = 2^(l-1)+1`
nodes, combined as the classical Smolyak operator
`A(q,d) = Σ_{q-d+1 ≤ |l| ≤ q} (-1)^(q-|l|) binom(d-1, q-|l|) U_{l₁} ⊗ … ⊗ U_{l_d}`
with `q = level`. With the default `base=:gh` the 1-D rules are Gauss–Hermite
for `N(0,1)` with `m(l)` nodes, so the sparse grid is **exact for all
polynomials of total degree ≤ 2·level−1 whenever level ≥ 2d−1** (checked
numerically for d = 2, 3; the exactness degree is often higher, e.g. 9 at
level 4 in d = 2) while using far fewer nodes than the full tensor grid
`m(level)^d` — e.g. 89 vs 289 nodes at level 5 in d = 2, and 146 vs 4913 at
level 5 in d = 3. `base=:trap` uses nested dyadic trapezoid rules on `[-L, L]`
instead (like `trap_rule`): node sharing makes the grid even sparser (37 nodes
at level 5 in d = 2) and it copes with non-smooth factors, but it is not
polynomial-exact (only spectrally accurate for smooth integrands).

The Smolyak combination has **signed weights** (`w` may be negative; `logw`
stores `log|w|`), which the rank-`d` moment path handles via a signed
log-sum-exp. If the signed quadrature sum `z = Σ wₙ f(ηₙ)` comes out
non-positive for a hard non-polynomial factor, the moment code falls back to
the cavity moments (no information), exactly as for a degenerate rule.

For `d = 1` the rule is just the level-`level` 1-D rule (`grid === nothing`),
so it also works on rank-1 sites. A rule with `d ≥ 2` is rejected on rank-1
sites with an `ArgumentError`. Requires `level ≥ d` for `d ≥ 2`.
"""
function smolyak_rule(level::Int, d::Int; base::Symbol=:gh, L::Real=8.0)
    (base === :gh || base === :trap) || throw(ArgumentError("base must be :gh or :trap, got :$base"))
    level >= 1 || throw(ArgumentError("level >= 1 required, got $level"))
    d >= 1 || throw(ArgumentError("d >= 1 required, got $d"))
    (d == 1 || level >= d) || throw(ArgumentError("level >= d required for d >= 2 (got level=$level, d=$d)"))
    rules = [_smolyak_1d(l, base, L) for l in 1:level]
    d == 1 && return QuadRule(rules[level][1], rules[level][2], log.(rules[level][2]))
    q = level
    nodes = Dict{NTuple{d,Float64}, Float64}()
    for l in Iterators.product(ntuple(_ -> 1:q, d)...)
        s = sum(l)
        (q - d + 1 <= s <= q) || continue
        c = (-1)^(q - s) * binomial(d - 1, q - s)
        for idx in Iterators.product(ntuple(i -> 1:length(rules[l[i]][1]), d)...)
            z = ntuple(i -> rules[l[i]][1][idx[i]], d)
            w = c * prod(i -> rules[l[i]][2][idx[i]], 1:d)
            nodes[z] = get(nodes, z, 0.0) + w
        end
    end
    npts = length(nodes)
    Z = Matrix{Float64}(undef, d, npts)
    w = Vector{Float64}(undef, npts)
    for (j, kv) in enumerate(nodes)
        z = kv.first
        for i in 1:d
            Z[i, j] = z[i]
        end
        w[j] = kv.second
    end
    # x = 1-D nodes of the top-level rule (informational; rank-1 use of a d≥2
    # rule is rejected in moments.jl)
    QuadRule(rules[q][1], w, log.(abs.(w)), Z)
end
