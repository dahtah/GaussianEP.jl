# General product-structure models for EP:
#   p(x) ∝ exp(-.5 x'Q0 x + r0'x) ∏_i f_i(A_i x),   x ∈ R^m
# Each factor is a Site carrying a d×m matrix A_i with d small.

abstract type AbstractSite end

"""
    Site(A, y, lik)

Factor f_i(x) = exp(log_dens(lik, A*x, y)) with A a d×m matrix.
The rank d is a type parameter so that moment computation and site
updates can dispatch on it.
"""
struct Site{d,A,L,Y} <: AbstractSite
    A :: A      # d × m matrix
    y :: Y      # observation attached to the factor
    lik :: L    # likelihood, log_dens(lik, η, y) defined on R^d
end

Site(A::AbstractMatrix, y, lik) = Site{size(A,1),typeof(A),typeof(lik),typeof(y)}(A, y, lik)

Base.show(io::IO, s::Site{d}) where d = print(io, "Site of rank $d, lik=$(typeof(s.lik))")

"""
    EPModel(Q0, sites; r0=zeros(m))

Product-structure model p(x) ∝ exp(-.5 x'Q0 x + r0'x) ∏_i f_i(A_i x).
Q0 keeps its own type (Matrix, Diagonal, SparseMatrixCSC, ...), so sparse
or diagonal prior precisions are stored without densification.
"""
struct EPModel{Q,S<:AbstractSite}
    Q0 :: Q
    r0 :: Vector{Float64}
    sites :: Vector{S}
end

function EPModel(Q0::AbstractMatrix, sites::Vector{S}; r0=zeros(size(Q0,1))) where S<:AbstractSite
    EPModel{typeof(Q0),S}(Q0, Vector{Float64}(r0), sites)
end

dim(M::EPModel) = size(M.Q0, 1)
nsites(M::EPModel) = length(M.sites)

function Base.show(io::IO, M::EPModel)
    print(io, "EPModel in dim $(dim(M)) with $(nsites(M)) sites, Q0::$(typeof(M.Q0))")
end

# --- model builders ---

"""
    glm_model(X, y, lik, τ=0.01)

GLM as an `EPModel`: prior `N(0, τ⁻¹I)` on the p coefficients and one rank-1
site per observation, `A_i = X[:,i]'`, `f_i(η) = exp(log_dens(lik, η, y_i))`.
`X` is p×n.
"""
function glm_model(X::AbstractMatrix, y, lik, τ=0.01)
    p, n = size(X)
    sites = [Site(reshape(X[:,i], 1, p), y[i], lik) for i in 1:n]
    EPModel(τ*Matrix{Float64}(I, p, p), sites)
end

"""
    ising_model(H; field=zeros(size(H,1)))

Ising model `p(x) ∝ exp(-.5 x'Hx + field'x) ∏ᵢ I(|xᵢ| = 1)` as an `EPModel`:
the Gaussian part is the base measure (`Q0 = H`, `r0 = field`) and each
hypercube constraint is a unary `IsingLik` site with analytic tilted moments.
"""
function ising_model(H::AbstractMatrix; field=zeros(size(H,1)))
    m = size(H, 1)
    sites = [Site(_unit_row(i, m), nothing, IsingLik()) for i in 1:m]
    EPModel(H, sites; r0=field)
end

function _unit_row(i, m)
    e = zeros(m); e[i] = 1.0
    reshape(e, 1, m)
end


"""
    gmrf_model(Q0, y, lik)

GMRF with unary observations: latent field x ∈ R^p with Gaussian prior
N(0, Q0⁻¹) (`Q0` sparse, e.g. a graph Laplacian plus a ridge) and one rank-1
site per grid point, `A_i = e_i'` (sparse), `f_i(η) = exp(log_dens(lik, η, y_i))`.
Uses the `SparseState` backend by default (`backend=:auto` on `ep`).
"""
function gmrf_model(Q0::AbstractMatrix, y::AbstractVector, lik)
    p = size(Q0, 1)
    length(y) == p || throw(DimensionMismatch("length(y) must equal size(Q0,1)"))
    sites = [Site(_sparse_unit_row(i, p), y[i], lik) for i in 1:p]
    EPModel(Q0, sites)
end

_sparse_unit_row(i, p) = sparse([1], [i], [1.0], 1, p)

"""
    gmrf_1d(y, lik=Logit(); ρ=1.0)

1D GMRF: prior precision `Q0 = ρ·L + I` with `L` the 1D finite-difference
Laplacian (tridiagonal: diag 2, off-diag −1) and unary sites `e_i'` — the
sparse analog of GP classification with an exponential (Matérn-1/2) kernel,
with observations `y_i ~ lik(x_i)` (e.g. `Logit()` for binary labels).
"""
function gmrf_1d(y::AbstractVector, lik=Logit(); ρ::Real=1.0)
    p = length(y)
    L = spdiagm(-1 => -ones(p-1), 0 => 2.0 .* ones(p), 1 => -ones(p-1))
    gmrf_model(sparse(ρ .* L + I), y, lik)
end

"""
    lgcp_model(counts; ρ=1.0, τ=1.0)
    lgcp_model(y::AbstractVector, L::Int; ρ=1.0, τ=1.0)

Log-Gaussian Cox process on the unit square, discretized on an L×L grid
(`counts` an L×L matrix, or a length-L² vector with `L` given): the latent
field `x = log λ` at the p = L² cell centers has GMRF prior precision
`Q0 = τ*(ρ*L2d + I)` with `L2d` the 2D finite-difference Laplacian, and each
cell count is a unary site `y_j ~ Poisson(a*exp(x_j))` with cell area
`a = 1/L²` (`PoisLikA(a)`). This is the grid approximation of a Poisson
process with intensity λ(s) = exp(x(s)) — the sparse analog of GP
regression with Poisson likelihood. Everything is sparse → `SparseState`
backend by default (`backend=:auto`).
"""
function lgcp_model(counts::AbstractMatrix; ρ::Real=1.0, τ::Real=1.0)
    size(counts, 1) == size(counts, 2) || throw(ArgumentError("counts must be square (L×L)"))
    lgcp_model(vec(counts), size(counts, 1); ρ=ρ, τ=τ)
end
function lgcp_model(y::AbstractVector, L::Int; ρ::Real=1.0, τ::Real=1.0)
    p = L * L
    length(y) == p || throw(DimensionMismatch("length(y) must equal L²"))
    L1 = spdiagm(-1 => -ones(L-1), 0 => 2.0 .* ones(L), 1 => -ones(L-1))
    L2d = kron(L1, sparse(1.0I, L, L)) + kron(sparse(1.0I, L, L), L1)
    Q0 = sparse(τ .* (ρ .* L2d + I))
    a = 1.0 / p
    lik = PoisLikA(a)
    sites = [Site(_sparse_unit_row(i, p), Float64(y[i]), lik) for i in 1:p]
    return EPModel(Q0, sites)
end

# --- SPDE / Matérn precision on the unit square -----------------------------
# A Matérn field with smoothness ν in d=2 satisfies (κ² - Δ)^c x = W with
# c = ν + 1 (integer c: 1 = exponential, 2, 3, ...) and W white noise of
# intensity σ². Discretized with spacing h = 1/L and the 5-point FD Laplacian
# L_h (scaled by 1/h²), the discrete equation A^c x = w with A = κ²I - L_h and
# w ~ N(0, σ²/h²) per node (white noise integrated over a cell of area h²)
# gives the grid-consistent precision
#     Q = (h²/σ²) · A^(2c),
# whose marginal variance (Q⁻¹)_jj approximates the continuous Matérn
# variance independently of L — unlike the unscaled Laplacian ρL + I, whose
# level (constant-mode) precision grows like τ·p and over-regularises fine
# grids. Boundary conditions: the domain is extended by m cells on each side
# and Dirichlet (x = 0) is imposed on the outer ring of the extended grid;
# the extension nodes stay latent (no sites) and only mediate the BC.
"""
    matern_precision_2d(L, m; c::Int=2, κ=1.0, σ²=1.0) -> (Q0, interior)

SPDE Matérn precision on the unit square: extended (L+2m)² grid, spacing
h = 1/L, 5-point FD Laplacian `L_h` (scaled by 1/h²) with Dirichlet on the
outer ring, `A = κ²I - L_h`, and `Q0 ∝ A^(2c)` on the (L+2m−2)² free nodes.
The scale is normalized so that the marginal prior variance at the domain
center equals `σ²` (one sparse solve at setup) — this absorbs the h-power of
the discrete Green's function and makes the prior **grid-consistent**: the
marginal variance is constant in L (verified: c=1, κ=1 gives var ≈ 0.02·σ²
for L = 20…160), and the level (constant-mode) precision ~ h²κ^{4c} shrinks
with L instead of growing like τ·p as with the unscaled Laplacian ρL + I.
Returns `Q0` (sparse, SPD) and `interior`, the vector of free-node indices of
the original L² cells (column-major, matching `vec(counts)`).
"""
function matern_precision_2d(L::Int, m::Int; c::Int=2, κ::Real=1.0, σ²::Real=1.0)
    c >= 1 || throw(ArgumentError("c must be a positive integer"))
    N = L + 2m                 # extended grid size
    h = 1.0 / L
    nf = N - 2                 # free nodes per side (Dirichlet ring removed)
    n = nf * nf
    # free node k = (i-2) + (j-2)*nf  ↔  extended grid point (i, j), i,j ∈ 2..N-1
    # FD Laplacian with Dirichlet: diag 4/h², off-diag -1/h² to free neighbors
    invh2 = 1.0 / h^2
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for j in 2:N-1, i in 2:N-1
        k = (i - 2) + (j - 2) * nf + 1
        push!(I_, k); push!(J_, k); push!(V_, 4 * invh2)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii, jj = i + di, j + dj
            if 2 <= ii <= N - 1 && 2 <= jj <= N - 1
                kk = (ii - 2) + (jj - 2) * nf + 1
                push!(I_, k); push!(J_, kk); push!(V_, -invh2)
            end
            # neighbor on the Dirichlet ring contributes 0 (x = 0 there)
        end
    end
    Lh = sparse(I_, J_, V_, n, n)
    A = κ^2 * sparse(1.0I, n, n) - Lh          # SPD, eigenvalues ≥ κ²
    Q = A
    for _ in 2:(2c)
        Q = Q * A                              # sparse power A^(2c), fill grows mildly
    end
    Q = sparse((Q + Q') / 2)                   # sparse matmul is not exactly symmetric
    # normalize: marginal prior variance at the domain center = σ².
    # (sQ)⁻¹ = Q⁻¹/s, so we need s = v/σ² with v = (A^{-2c})_ctr,ctr ~ h²·const
    ctr = ((L÷2 + m - 2) + (L÷2 + m - 2) * nf + 1)
    e = zeros(n); e[ctr] = 1.0
    v = sum(e .* (Q \ e))                      # (A^{-2c})_ctr,ctr
    Q = sparse((v / σ²) .* Q)
    # interior map: original cell (i,j) ∈ 1..L → extended (i+m, j+m) → free node
    interior = [((i + m - 2) + (j + m - 2) * nf + 1) for j in 1:L for i in 1:L]
    return Q, interior
end

"""
    lgcp_model_spde(counts; c=2, κ=1.0, σ²=1.0, m=nothing)
    lgcp_model_spde(y::AbstractVector, L::Int; c=2, κ=1.0, σ²=1.0, m=nothing)

Log-Gaussian Cox process with an SPDE Matérn prior (grid-consistent in L):
latent `x = log λ` on the extended grid of `matern_precision_2d` (Dirichlet
extension of `m = max(4, L÷8)` cells per side by default, precision
`∝ (κ²I - L_h)^(2c)` normalized to marginal variance `σ²`), Poisson cell-count
sites `y_j ~ Poisson(a·exp(x_j))`, `a = 1/L²`, only on the original L² cells.
Matérn smoothness ν = c − 1 in 2D (c = 1 exponential, c = 2 ν = 1, ...);
κ is the inverse range. Fully sparse → `SparseState` backend by default.
"""
function lgcp_model_spde(counts::AbstractMatrix; kwargs...)
    size(counts, 1) == size(counts, 2) || throw(ArgumentError("counts must be square (L×L)"))
    lgcp_model_spde(vec(counts), size(counts, 1); kwargs...)
end
function lgcp_model_spde(y::AbstractVector, L::Int; c::Int=2, κ::Real=1.0, σ²::Real=1.0, m=nothing)
    p = L * L
    length(y) == p || throw(DimensionMismatch("length(y) must equal L²"))
    m = m === nothing ? max(4, L ÷ 8) : m
    Q0, interior = matern_precision_2d(L, m; c=c, κ=κ, σ²=σ²)
    n = size(Q0, 1)
    a = 1.0 / p
    lik = PoisLikA(a)
    sites = [Site(_sparse_unit_row(interior[i], n), Float64(y[i]), lik) for i in 1:p]
    return EPModel(Q0, sites)
end

# Gas model: p(x) ∝ exp(-β Σ_{i<j} H(||x_i - x_j||) - Σ_i V(x_i)), x ∈ R^{np*d}.
# Pairwise sites have A_ij = [.. I_d .. -I_d ..] (rank d); unary confining
# sites have A_i = [.. I_d ..].
#
# Symmetry breaking: the full posterior is multimodal and EP can only center
# on one local minimum, so particles are pinned. Particle 1 is pinned at the
# origin; pinning a second particle at `pin2` (a d-vector hyperparameter) is
# necessary to break the reflection symmetry — with only particle 1 pinned
# and V even, E[x_i] = 0 for every free particle and EP stays at the
# symmetric saddle. Pass pin2=nothing to pin only particle 1.
"""
    gas_model(np, d; β=1.0, H=r->r^2, V=η->0.5*sum(abs2,η), τ0=1e-6, pin2=ones(d))

Gas of `np` particles in `R^d` as an `EPModel`:
`p(x) ∝ exp(-β Σ_{i<j} H(‖x_i - x_j‖)) exp(Σ_i V(x_i))` with a weak Gaussian
base of precision `τ0`. Pairwise factors are rank-`d` sites
`A_ij = [.. I_d .. -I_d ..]`, unary confining factors are rank-`d` sites
`A_i = [.. I_d ..]`. The full posterior is multimodal, so EP can only center
on one local minimum: particle 1 is pinned at the origin and a second
particle at `pin2` (a d-vector hyperparameter) breaks the reflection
symmetry — `pin2=nothing` pins only particle 1, in which case EP stays at
the symmetric saddle with `μ = 0`.
"""
function gas_model(np::Int, d::Int; β=1.0, H=(r)->r^2, V=(η)->0.5*sum(abs2, η), τ0=1e-6, pin2=ones(d))
    npinned = pin2 === nothing ? 1 : 2
    mfull = np*d
    kept = collect((npinned*d+1):mfull)
    m = length(kept)
    cmap = zeros(Int, mfull)
    cmap[kept] .= 1:m
    # d×m matrix with `sign` on the block of particle i (0 if block dropped)
    function blockA(i, sign)
        A = zeros(d, m)
        for k in 1:d
            c = (i-1)*d + k
            cmap[c] > 0 && (A[k, cmap[c]] = sign)
        end
        A
    end
    sites = AbstractSite[]
    # pairs with pinned particle 1 (at the origin): unary factors on x_j
    for j in (npinned+1):np
        push!(sites, Site(blockA(j, 1.0), nothing, RepulseLik(β, H)))
    end
    # pairs with pinned particle 2 (at pin2): unary factors on x_j
    if npinned == 2
        for j in 3:np
            push!(sites, Site(blockA(j, 1.0), nothing, RepulseLik(β, H, pin2)))
        end
    end
    # pairs between free particles
    for i in (npinned+1):np, j in (i+1):np
        push!(sites, Site(blockA(i, 1.0) - blockA(j, 1.0), nothing, RepulseLik(β, H)))
    end
    # confining potential on free particles
    for i in (npinned+1):np
        push!(sites, Site(blockA(i, 1.0), nothing, ConfineLik(V)))
    end
    EPModel(τ0*Matrix{Float64}(I, m, m), sites)
end

# Sparse linear model (Seeger 2008): y = X a + ε with ε ~ N(0, σ² I_n) and
# independent Laplace priors p(a_i) = (λ/2) exp(-λ|a_i|) on the coefficients.
# X is p×n (p coefficients, n observations). The Gaussian likelihood is the
# base measure — Q0 = X'X/σ² (+ τ0 ridge to keep it PD), r0 = X'y/σ² — and
# each Laplace prior is a unary site with A_i = e_i', so the non-Gaussian
# sites live in the prior, not in the likelihood.
"""
    sparse_lm(X, y, λ, σ2=1.0; τ0=1e-8)

Sparse linear model (Seeger 2008): `y = X'a + ε`, `ε ~ N(0, σ²I)`, with
independent Laplace priors `p(a_i) = (λ/2) exp(-λ|a_i|)` on the coefficients.
`X` is p×n (p coefficients, n observations). The Gaussian likelihood is the
base measure — `Q0 = XX'/σ²` (+ `τ0` ridge to keep it PD), `r0 = Xy/σ²` — and
each Laplace prior is a unary site with `A_i = e_i'`, so the non-Gaussian
sites live in the prior, not in the likelihood.
"""
function sparse_lm(X::AbstractMatrix, y::AbstractVector, λ::Real, σ2::Real=1.0; τ0=1e-8)
    p, n = size(X)
    length(y) == n || throw(DimensionMismatch("length(y) must equal size(X,2)"))
    Q0 = (X*X')/σ2 + τ0*Matrix{Float64}(I, p, p)
    r0 = (X*y)/σ2
    sites = [Site(_unit_row(i, p), nothing, LaplaceLik(Float64(λ))) for i in 1:p]
    EPModel(Q0, sites; r0=r0)
end

# Spike-and-slab linear model: same Gaussian likelihood as sparse_lm, but
# each coefficient has the mixture prior p(a_i) = w δ(a_i) + (1-w) N(a_i; 0, τ2)
# (w = spike probability). One mixture site per coefficient; tilted moments
# are analytic (moments.jl). The exact posterior is a mixture of 2^p Gaussians
# (one per active subset), which makes m ≤ 3 cases checkable in closed form.
"""
    spike_slab_lm(X, y, w, τ2, σ2=1.0; τ0=1e-8)

Spike-and-slab linear model: same Gaussian likelihood as `sparse_lm`, but
each coefficient has the mixture prior `p(a_i) = w δ(a_i) + (1-w) N(a_i; 0, τ2)`
(`w` = spike probability). One `SpikeSlabLik` site per coefficient; tilted
moments are analytic. The exact posterior is a mixture of `2^p` Gaussians
(one per active subset), which makes `p ≤ 3` cases checkable in closed form.
"""
function spike_slab_lm(X::AbstractMatrix, y::AbstractVector, w::Real, τ2::Real, σ2::Real=1.0; τ0=1e-8)
    p, n = size(X)
    length(y) == n || throw(DimensionMismatch("length(y) must equal size(X,2)"))
    0 ≤ w ≤ 1 || throw(ArgumentError("w must be in [0,1]"))
    τ2 > 0 || throw(ArgumentError("τ2 must be positive"))
    Q0 = (X*X')/σ2 + τ0*Matrix{Float64}(I, p, p)
    r0 = (X*y)/σ2
    sites = [Site(_unit_row(i, p), nothing, SpikeSlabLik(Float64(w), Float64(τ2))) for i in 1:p]
    EPModel(Q0, sites; r0=r0)
end
