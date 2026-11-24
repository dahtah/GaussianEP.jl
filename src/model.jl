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
