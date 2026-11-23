# Spike-and-slab models: convenient interface + hyperparameter handling.
#   spike_slab_lm  — linear model (Gaussian likelihood in the base measure),
#                    one SpikeSlabLik site per coefficient (src/model.jl)
#   spike_slab_glm — GLM (Logit/PoisLik/... sites on the design rows) plus one
#                    SpikeSlabLik prior site per coefficient
#   inclusion_probs(G) — posterior inclusion probabilities (PIPs) from the
#                    slab weight of each SpikeSlabLik site at its final cavity
#   spike_slab_autotune — empirical-Bayes grid over (w, τ2) maximising EP log_ml

"""
    spike_slab_glm(X, y, lik, w, τ2; τ0=1e-8)

Spike-and-slab GLM: likelihood sites `f_i(x_i'β)` (`lik` = `Logit()`,
`PoisLik()`, ...) on the design rows, plus one `SpikeSlabLik(w, τ2)` prior
site per coefficient (`A_i = e_i'`). Base measure `N(0, τ0⁻¹ I)` (tiny ridge).
Sites 1..n are the likelihood sites, sites n+1..n+p the prior sites.
"""
function spike_slab_glm(X::AbstractMatrix, y, lik, w::Real, τ2::Real; τ0::Real=1e-8)
    p, n = size(X, 1), size(X, 2)
    0 ≤ w ≤ 1 || throw(ArgumentError("w must be in [0,1]"))
    τ2 > 0 || throw(ArgumentError("τ2 must be positive"))
    sites = [Site(reshape(X[:, i], 1, p), y[i], lik) for i in 1:n]
    append!(sites, [Site(_unit_row(i, p), nothing, SpikeSlabLik(Float64(w), Float64(τ2))) for i in 1:p])
    EPModel(τ0*Matrix{Float64}(I, p, p), sites)
end

"""
    slab_weight(l::SpikeSlabLik, m, s2)

Posterior slab weight (1 − spike mass) of a spike-and-slab prior site against
a cavity `N(m, s²)` — the tilted-measure slab probability `a` of the analytic
rule in `moments.jl`. Used for posterior inclusion probabilities.
"""
function slab_weight(l::SpikeSlabLik, m::Real, s2::Real)
    lspike = log(l.w) - .5*(m^2/s2 + log(2π*s2))
    lslab = log1p(-l.w) - .5*(m^2/(s2+l.τ2) + log(2π*(s2+l.τ2)))
    exp(lslab - logaddexp(lspike, lslab))
end

"""
    inclusion_probs(G::EPApprox)

Posterior inclusion probabilities (PIPs) of the `SpikeSlabLik` sites of the
model, in site order: for each prior site, the slab weight at its final EP
cavity (`slab_weight`). `PIP_i > 0.5` is the natural EP support selector.
"""
function inclusion_probs(G::EPApprox)
    pips = Float64[]
    for (i, s) in enumerate(G.model.sites)
        if s.lik isa SpikeSlabLik
            cv = cavity(G, i, 1.0)
            push!(pips, slab_weight(s.lik, cv.m[1], cv.S[1, 1]))
        end
    end
    pips
end

"""
    support(G::EPApprox, thresh=0.5)

Indices of the `SpikeSlabLik` sites with PIP > `thresh` (EP support estimate).
"""
support(G::EPApprox, thresh::Real=0.5) = findall(>(thresh), inclusion_probs(G))

"""
    spike_slab_autotune(X, y; ws, τ2s, σ2=1.0, τ0=1e-8, kwargs...)

Empirical-Bayes hyperparameter selection for the spike-and-slab linear model
`spike_slab_lm`: fits `ep` on the grid `ws × τ2s` and returns the setting
maximising the EP log marginal likelihood. Returns
`(w, τ2, logml, G, grid)` with `G` the fitted `EPApprox` at the optimum and
`grid` the full `(w, τ2, logml)` list. `kwargs...` are passed to `ep`.
"""
function spike_slab_autotune(X::AbstractMatrix, y::AbstractVector;
                             ws=(0.5, 0.8, 0.9, 0.95), τ2s=(1.0, 10.0, 100.0),
                             σ2::Real=1.0, τ0::Real=1e-8, kwargs...)
    best = nothing
    grid = NamedTuple{(:w, :τ2, :logml, :converged)}[]
    for w in ws, τ2 in τ2s
        M = spike_slab_lm(X, y, w, τ2, σ2; τ0=τ0)
        G = ep(M; kwargs...)
        lml = log_ml(G)
        # reject diverged fits (huge residual / non-finite evidence): they
        # would otherwise win the argmax with a garbage logml
        ok = isfinite(lml) && isfinite(G.lastδ) && G.lastδ < 1e3
        push!(grid, (w=w, τ2=τ2, logml=lml, converged=ok))
        if ok && (best === nothing || lml > best.logml)
            best = (w=w, τ2=τ2, logml=lml, G=G)
        end
    end
    best === nothing && error("spike_slab_autotune: no grid point converged — try damping or a larger τ0")
    (w=best.w, τ2=best.τ2, logml=best.logml, G=best.G, grid=grid)
end
