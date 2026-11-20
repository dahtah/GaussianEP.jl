# General EP approximation for an EPModel.
# Natural parameters: Q = Q0 + Σ_i A_i' Qi A_i,  r = r0 + Σ_i A_i' ri,
# with Qi a d×d site precision and ri a d-vector. Σ = Q⁻¹ and μ = Σ r
# are maintained explicitly via rank-d Woodbury updates.

struct EPApprox{T,M<:EPModel}
    model :: M
    Q :: Matrix{T}
    Σ :: Matrix{T}
    μ :: Vector{T}
    r :: Vector{T}
    Qi :: Vector{Matrix{T}}
    ri :: Vector{Vector{T}}
    logz :: Vector{T}
end

Statistics.mean(G::EPApprox) = copy(G.μ)
Statistics.cov(G::EPApprox) = G.Σ
se(G::EPApprox) = sqrt.(diag(G.Σ))
dim(G::EPApprox) = size(G.Q, 1)
nsites(G::EPApprox) = nsites(G.model)

function Base.show(io::IO, G::EPApprox)
    print(io, "EPApprox in dim $(dim(G)) with $(nsites(G)) sites")
end

# project a symmetric matrix onto the PSD cone
function psd_project(S::Symmetric)
    F = eigen(S)
    Symmetric(F.vectors * Diagonal(clamp.(F.values, 0, Inf)) * F.vectors')
end

# inverse of a symmetric matrix, clamped to stay PD
function pd_inv(S::Symmetric)
    F = eigen(S)
    Symmetric(F.vectors * Diagonal(1 ./ clamp.(F.values, 1e-12, Inf)) * F.vectors')
end

# log ∫ exp(-.5 η'Λη + λ'η) dη, robust to indefinite Λ (bookkeeping only)
function safe_log_partition(Λ::AbstractMatrix, λ::AbstractVector)
    d = length(λ)
    F = eigen(Symmetric(Λ))
    v = clamp.(F.values, 1e-300, Inf)
    .5*(d*log(2π) - sum(log, v) + sum((F.vectors'λ).^2 ./ v))
end

# Initialise with one parallel EP step from the prior, projecting each
# site's natural-parameter change onto the PSD cone so Q stays PD.
function EPApprox(M::EPModel; qr=QuadRule(20), T=Float64, stein::Bool=false, power::Float64=1.0)
    m = dim(M)
    Q0 = Matrix{T}(M.Q0)
    Σ0 = inv(Symmetric(Q0))
    μ0 = Σ0 * M.r0
    n = nsites(M)
    Qi = Vector{Matrix{T}}(undef, n)
    ri = Vector{Vector{T}}(undef, n)
    Q = copy(Q0)
    r = Vector{T}(M.r0)
    for i in 1:n
        s = M.sites[i]
        A = s.A
        S = Symmetric(A * Σ0 * A')
        mv = A * μ0
        z, lzt, μt, Σt = tilted_moments(s.lik, s.y, mv, S, qr, stein=stein, power=power)
        Λc = pd_inv(S)
        λc = Λc * mv
        Qp = pd_inv(Symmetric(Σt))
        rp = Qp * μt
        # site natural params are tilted minus cavity (absolute, not incremental)
        Qi[i] = psd_project(Symmetric(Qp - Λc))
        ri[i] = rp - λc
        Q .+= A' * Qi[i] * A
        r .+= A' * ri[i]
    end
    Σ = inv(Symmetric(Q))
    μ = Σ * r
    EPApprox{T,typeof(M)}(M, Q, Σ, μ, r, Qi, ri, zeros(T, n))
end

# cavity parameters of η_i = A_i x, in moment form. If the cavity precision
# Fractional EP (Seeger 2008, Sec. 3.5): the cavity removes only a fraction
# `power` of the site approximation, Q\i ∝ Q / t̃_i^power. If the resulting
# cavity precision inv(S) - t*Qi is not PD (site too strong), shrink the
# removed fraction t further until it is.
function cavity(G::EPApprox, i, power::Float64=1.0)
    A = G.model.sites[i].A
    S = Symmetric(A * G.Σ * A')
    m = A * G.μ
    Si = pd_inv(S)
    t = power
    Λc = Symmetric(Si - t*G.Qi[i])
    while !isposdef(Λc) && t > 1e-8
        t /= 2
        Λc = Symmetric(Si - t*G.Qi[i])
    end
    λc = Si * m - t*G.ri[i]
    Sc = pd_inv(Λc)
    (m=Sc * λc, S=Sc)
end

function site_update(G::EPApprox, i, qr::QuadRule; stein::Bool=false, power::Float64=1.0)
    s = G.model.sites[i]
    cv = cavity(G, i, power)
    z, lzt, μt, Σt = tilted_moments(s.lik, s.y, cv.m, cv.S, qr, stein=stein, power=power)
    Qp = pd_inv(Symmetric(Σt))
    rp = Qp * μt
    Λc = pd_inv(Symmetric(cv.S))
    λc = Λc * cv.m
    dQ = Qp - Λc
    dr = rp - λc
    # evidence term: always computed with the true factor against the full
    # cavity (power=1), so log_ml estimates the evidence of the unchanged
    # target even under fractional EP
    if power == 1.0
        lz = lzt + safe_log_partition(Λc, λc) - safe_log_partition(Qp, rp)
    else
        cv1 = cavity(G, i, 1.0)
        _, lzt1, μt1, Σt1 = tilted_moments(s.lik, s.y, cv1.m, cv1.S, qr, stein=stein, power=1.0)
        Qp1 = pd_inv(Symmetric(Σt1))
        rp1 = Qp1 * μt1
        Λc1 = pd_inv(Symmetric(cv1.S))
        λc1 = Λc1 * cv1.m
        lz = lzt1 + safe_log_partition(Λc1, λc1) - safe_log_partition(Qp1, rp1)
    end
    (dQ=dQ, dr=dr, lz=lz)
end

function update_site!(G::EPApprox, i, dQ, dr, lz, α=1.0, power::Float64=1.0)
    A = G.model.sites[i].A
    # dQ, dr are the proposed new absolute site natural params (tilted minus
    # fractional cavity). Fractional EP (Seeger 2008, Sec. 3.5) updates the
    # site by the convex combination  Q_i ← (1-power) Q_i + dQ,  i.e. the
    # applied change is δ = α (dQ - power·Q_i); the target is unchanged —
    # for Gaussian factors the fixed point is the exact factor for any power.
    δQ = α .* (dQ .- power .* G.Qi[i])
    δr = α .* (dr .- power .* G.ri[i])
    SA = G.Σ * A'
    SAA = A * SA
    # Q + A'δQ A stays PD iff K = I + AΣA' δQ is PD (determinant lemma);
    # backtrack δ if not, so the approximation never leaves the PD cone
    K = I + SAA * δQ
    while !isposdef(Symmetric(K)) && maximum(abs.(δQ)) > 1e-14
        δQ ./= 2
        δr ./= 2
        K = I + SAA * δQ
    end
    G.Qi[i] .+= δQ
    G.ri[i] .+= δr
    G.logz[i] = lz
    G.Q .+= A' * δQ * A
    G.r .+= A' * δr
    # rank-d Woodbury update of Σ:  Q += A'δQ A  ⟹
    # Σ -= ΣA' δQ (I + AΣA' δQ)⁻¹ AΣ
    G.Σ .-= SA * (δQ / K) * SA'
    G.Σ .= Symmetric(G.Σ)
    G.μ .= G.Σ * G.r
    return
end

function fit!(G::EPApprox; qr=QuadRule(20), maxcycles=50, tol=1e-8, damp=1.0, stein::Bool=false, parallel::Bool=false, power::Float64=1.0)
    m = copy(G.μ)
    δ = Inf
    n = nsites(G)
    for c in 1:maxcycles
        # resync Σ = Q⁻¹, μ = Σr: Woodbury updates drift, and the PD
        # backtracking in update_site! is only valid when Σ ≈ Q⁻¹
        G.Σ .= inv(Symmetric(G.Q))
        G.μ .= G.Σ * G.r
        δsite = 0.0
        if parallel
            # Jacobi-style: all site updates computed against the same state
            # (embarrassingly parallel), then applied with damping
            ups = Vector{NamedTuple{(:dQ, :dr, :lz)}}(undef, n)
            Threads.@threads for i in 1:n
                ups[i] = site_update(G, i, qr, stein=stein, power=power)
            end
            for i in 1:n
                δsite = max(δsite, maximum(abs.(ups[i].dQ .- power .* G.Qi[i])), maximum(abs.(ups[i].dr .- power .* G.ri[i])))
                update_site!(G, i, ups[i].dQ, ups[i].dr, ups[i].lz, damp, power)
            end
        else
            for i in 1:n
                dQ, dr, lz = site_update(G, i, qr, stein=stein, power=power)
                δsite = max(δsite, maximum(abs.(dQ .- power .* G.Qi[i])), maximum(abs.(dr .- power .* G.ri[i])))
                update_site!(G, i, dQ, dr, lz, damp, power)
            end
        end
        # convergence on both the mean and the site natural parameters
        # (the mean alone can be stationary in symmetric models)
        δ = max(maximum(abs.(G.μ .- m)), δsite)
        δ < tol && return G
        m .= G.μ
    end
    @info "EP did not reach tolerance (δ=$δ); consider damping or more quadrature nodes"
    return G
end

function ep(M::EPModel; qr=QuadRule(20), maxcycles=50, tol=1e-8, damp=1.0, stein::Bool=false, parallel::Bool=false, power::Float64=1.0)
    G = EPApprox(M, qr=qr, stein=stein, power=power)
    fit!(G, qr=qr, maxcycles=maxcycles, tol=tol, damp=damp, stein=stein, parallel=parallel, power=power)
end

function log_ml(G::EPApprox)
    M = G.model
    safe_log_partition(G.Q, G.r) - safe_log_partition(Matrix(M.Q0), M.r0) + sum(G.logz)
end
