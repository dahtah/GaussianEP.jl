# Regularisation path over the prior precision τ for GLMs on the new EP core.
# The model is built ONCE with the unscaled base measure (glm_model(X, y, lik, 1.0),
# Q0 = I); the path scale τ lives in the state (st.τ, effective base = τ·Q0) and
# is changed in place with set_tau! — Q ← Q + (τ_new − τ_old)·Q0 plus one
# refactorisation, no per-site loop, no model/EPApprox reconstruction per grid
# point. Site parameters Qi, ri are carried over between grid points (warm
# start). The path runs from the high-τ end (τ1) down to τ0: at high τ the
# sites are weak and the posterior is close to the prior, which makes the
# first fit stable (status.md TODO "start with high τ").

"""
    rebuild!(G::EPApprox)

Recompute the state (`Q, r, Σ, μ, logdetQ`) from the model's base measure
`(G.state.τ * Q0, r0)` and the current site parameters `Qi, ri`. Used to
warm-start a fit across hyperparameter values (e.g. a change of prior
precision τ).
"""
function rebuild!(G::EPApprox)
    M = G.model
    st = G.state
    Q = st.τ .* M.Q0
    r = Vector{Float64}(M.r0)
    for i in eachindex(M.sites)
        A = M.sites[i].A
        Q = Q + A' * G.Qi[i] * A
        r .+= A' * G.ri[i]
    end
    reset!(st, Q, r)          # backend-agnostic (DenseState / CholeskyState)
    return G
end

"""
    set_tau!(G::EPApprox, τ)

Change the base-measure scale in place: the effective prior precision matrix
goes from `G.state.τ · Q0` to `τ · Q0`. The state's `Q` (which already contains
the site contributions `Σᵢ Aᵢ'QᵢAᵢ`) is updated as `Q ← Q + (τ − τ_old)·Q0` and
refactorised once — `O(p²)` plus one `O(p³)` factorisation, no per-site loop —
and the cached base-measure partition `G.logZ0` is recomputed, so `log_ml`
stays correct. Site parameters `Qi, ri` are kept (warm start). The model's
`Q0` is interpreted as the *unscaled* base measure; `G.state.τ` defaults to 1.
"""
function set_tau!(G::EPApprox, τ::Real)
    st = G.state
    τ == st.τ && return G
    M = G.model
    Q0 = M.Q0                        # keep its type (sparse stays sparse)
    Q = st.Q .+ (τ - st.τ) .* Q0     # st.Q = st.τ·Q0 + Σ A'QiA  (st.Q materialises for CholeskyState)
    reset!(st, Q, st.r)              # r = r0 + Σ A'ri is τ-independent
    st.τ = τ
    G.logZ0 = safe_log_partition(τ .* Q0, Vector{Float64}(M.r0))
    return G
end

"""
    glm_regpath(X, y, lik, τ0, τ1; nsteps=10, qr=QuadRule(20), maxcycles=50,
                tol=1e-8, damp=1.0, stein=false, parallel=false, power=1.0,
                backend=:dense)

Regularisation path over the prior precision τ ∈ [τ0, τ1] (log-spaced,
`nsteps` points) for a GLM `glm_model(X, y, lik, τ)` on the new EP core.
The path is traversed **from the high-τ end (τ1) down to τ0**, so the
returned `τs` are in decreasing order; each point is fitted with `fit!`,
warm-started from the previous point's site parameters. Starting at high τ
(strong prior, weakly informative sites, posterior ≈ prior) gives the most
stable first fit, and the continuation then tracks the posterior as the
prior is relaxed.

Implementation: one model (unscaled base `Q0 = I`) and one mutable `EPApprox`
are built for the whole path; the grid point τ is moved in place with
`set_tau!` (`Q ← Q + (τ_new − τ_old)·Q0`, one refactorisation — no per-site
loop, no reconstruction per grid point).

Returns a named tuple:

- `τs` — the grid, in **decreasing** order (τ1 first, τ0 last),
- `μs` — p×nsteps matrix of posterior means, `sds` — marginal posterior sds,
- `logml` — EP log marginal likelihood per point (for hyperparameter
  selection),
- `μavg`, `sdavg` — model-averaged mean and marginal sds, with weights
  ∝ exp(logml) over the path.
"""
function glm_regpath(X, y, lik, τ0, τ1; nsteps=10, qr=QuadRule(20), maxcycles=50,
                     tol=1e-8, damp=1.0, stein::Bool=false, parallel::Bool=false,
                     power::Float64=1.0, backend::Symbol=:dense)
    τs = exp.(range(log(τ1), log(τ0), nsteps))   # high-τ end first (see docstring)
    p = size(X, 1)
    μs = zeros(p, nsteps)
    sds = zeros(p, nsteps)
    lml = zeros(nsteps)
    # one model (unscaled base Q0 = I) and one EPApprox for the whole path; τ is
    # carried by the state and moved with set_tau!, site params warm-start
    M = glm_model(X, y, lik, 1.0)
    G = EPApprox(M; qr=qr, stein=stein, power=power, backend=backend)
    for (k, τ) in enumerate(τs)
        set_tau!(G, τ)
        fit!(G; qr=qr, maxcycles=maxcycles, tol=tol, damp=damp, stein=stein,
             parallel=parallel, power=power)
        μs[:, k] .= mean(G)
        sds[:, k] .= se(G)
        lml[k] = log_ml(G)
    end
    w = softmax(lml)                    # model weights ∝ exp(logml)
    μavg = μs * w
    varavg = vec(sum(w' .* (sds .^ 2 .+ μs .^ 2), dims=2)) .- μavg .^ 2
    sdavg = sqrt.(max.(varavg, 0.0))
    (τs=τs, μs=μs, sds=sds, logml=lml, μavg=μavg, sdavg=sdavg)
end

"""
    glm_autotune(X, y, lik, τ0, τ1; nsteps=10, kwargs...)

Empirical-Bayes selection of the prior precision τ for a GLM on the new EP
core: runs `glm_regpath` over [τ0, τ1] (traversed from τ1 down to τ0) and
refits at the τ maximising the EP log marginal likelihood. Returns
`(τ_opt, G, path)` with `G` the fitted `EPApprox` at `τ_opt` and `path` the
full `glm_regpath` result (its `τs` in decreasing order).
"""
function glm_autotune(X, y, lik, τ0, τ1; nsteps=10, kwargs...)
    path = glm_regpath(X, y, lik, τ0, τ1; nsteps=nsteps, kwargs...)
    k = argmax(path.logml)
    τ_opt = path.τs[k]
    G = ep(glm_model(X, y, lik, τ_opt); kwargs...)
    (τ_opt=τ_opt, G=G, path=path)
end

# ---------------------------------------------------------------------------
# Pretty-printing: compact summaries instead of dumping the raw arrays.
# The results of glm_regpath / glm_autotune are plain named tuples, so we
# specialise show on their exact key sets.
# ---------------------------------------------------------------------------

# compact rendering of a vector of numbers (truncate long grids)
function _compact_nums(v::AbstractVector; maxn::Int=6)
    r = round.(v, sigdigits=4)
    s = join(string.(r), ", ")
    length(v) > maxn && (s = join(string.(r[1:maxn]), ", ") * ", ...")
    return s
end

"""
    show(io, path)  # result of `glm_regpath`

Compact summary of a regularisation path: number of τ grid points, the τ
grid, the log marginal likelihood per point, and the best τ (max logml).
"""
function Base.show(io::IO, path::NamedTuple{(:τs, :μs, :sds, :logml, :μavg, :sdavg)})
    k = argmax(path.logml)
    print(io, "GLM regpath over tau: $(length(path.τs)) points in [$(minimum(path.τs)), $(maximum(path.τs))]")
    print(io, "\n  tau grid: ", _compact_nums(path.τs))
    print(io, "\n  logml:    ", _compact_nums(path.logml))
    print(io, "\n  best tau = $(path.τs[k]) with logml = $(path.logml[k])")
    return nothing
end
Base.show(io::IO, ::MIME"text/plain", path::NamedTuple{(:τs, :μs, :sds, :logml, :μavg, :sdavg)}) = show(io, path)

"""
    show(io, at)  # result of `glm_autotune`

Compact summary of an autotune result: the selected `τ_opt`, the log marginal
likelihood at the optimum, and the size of the underlying τ path.
"""
function Base.show(io::IO, at::NamedTuple{(:τ_opt, :G, :path)})
    print(io, "GLM autotune: tau_opt = $(at.τ_opt), logml at tau_opt = $(maximum(at.path.logml)) ",
          "(max over a $(length(at.path.τs))-point tau path)")
    return nothing
end
Base.show(io::IO, ::MIME"text/plain", at::NamedTuple{(:τ_opt, :G, :path)}) = show(io, at)
