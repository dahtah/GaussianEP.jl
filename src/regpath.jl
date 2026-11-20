# Regularisation path over the prior precision τ for GLMs on the new EP core.
# glm_model(X, y, lik, τ) has Q0 = τ·I, so τ rescales the base measure; each
# point of the path is fitted with `ep`, warm-started from the previous
# point's site parameters (rebuild! recomputes the state from Q0 + sites).

"""
    rebuild!(G::EPApprox)

Recompute the state (`Q, r, Σ, μ, logdetQ`) from the model's base measure
`(Q0, r0)` and the current site parameters `Qi, ri`. Used to warm-start a
fit across hyperparameter values (e.g. a change of prior precision τ).
"""
function rebuild!(G::EPApprox)
    st = G.state
    st.Q .= Matrix{Float64}(G.model.Q0)
    st.r .= Vector{Float64}(G.model.r0)
    for i in eachindex(G.model.sites)
        A = G.model.sites[i].A
        st.Q .+= A' * G.Qi[i] * A
        st.r .+= A' * G.ri[i]
    end
    sync!(st)
    return G
end

"""
    glm_regpath(X, y, lik, τ0, τ1; nsteps=10, qr=QuadRule(20), maxcycles=50,
                tol=1e-8, damp=1.0, stein=false, parallel=false, power=1.0)

Regularisation path over the prior precision τ ∈ [τ0, τ1] (log-spaced,
`nsteps` points) for a GLM `glm_model(X, y, lik, τ)` on the new EP core.
Each point is fitted with `ep`, warm-started from the previous point's site
parameters. Returns a named tuple:

- `τs` — the grid,
- `μs` — p×nsteps matrix of posterior means, `sds` — marginal posterior sds,
- `logml` — EP log marginal likelihood per point (for hyperparameter
  selection),
- `μavg`, `sdavg` — model-averaged mean and marginal sds, with weights
  ∝ exp(logml) over the path.
"""
function glm_regpath(X, y, lik, τ0, τ1; nsteps=10, qr=QuadRule(20), maxcycles=50,
                     tol=1e-8, damp=1.0, stein::Bool=false, parallel::Bool=false,
                     power::Float64=1.0)
    τs = exp.(range(log(τ0), log(τ1), nsteps))
    p = size(X, 1)
    μs = zeros(p, nsteps)
    sds = zeros(p, nsteps)
    lml = zeros(nsteps)
    Gprev = nothing
    for (k, τ) in enumerate(τs)
        M = glm_model(X, y, lik, τ)
        G = EPApprox(M; qr=qr, stein=stein, power=power)
        if Gprev !== nothing            # warm start from the previous point
            G.Qi .= Gprev.Qi
            G.ri .= Gprev.ri
            rebuild!(G)
        end
        fit!(G; qr=qr, maxcycles=maxcycles, tol=tol, damp=damp, stein=stein,
             parallel=parallel, power=power)
        μs[:, k] .= mean(G)
        sds[:, k] .= se(G)
        lml[k] = log_ml(G)
        Gprev = G
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
core: runs `glm_regpath` over [τ0, τ1] and refits at the τ maximising the EP
log marginal likelihood. Returns `(τ_opt, G, path)` with `G` the fitted
`EPApprox` at `τ_opt` and `path` the full `glm_regpath` result.
"""
function glm_autotune(X, y, lik, τ0, τ1; nsteps=10, kwargs...)
    path = glm_regpath(X, y, lik, τ0, τ1; nsteps=nsteps, kwargs...)
    k = argmax(path.logml)
    τ_opt = path.τs[k]
    G = ep(glm_model(X, y, lik, τ_opt); kwargs...)
    (τ_opt=τ_opt, G=G, path=path)
end
