# 2D square-lattice Ising: EP vs classical loopy BP vs Onsager analytic formula.
# Loose tolerances (EP is approximate on loopy graphs; Onsager is the L=∞ limit).
# Helpers (grid builder, BP, Onsager, EP runner, exact enumeration) come from
# eval/ising2d.jl / eval/ising2d_bp.jl (main loops there are PROGRAM_FILE-guarded).
include(joinpath(@__DIR__, "..", "eval", "ising2d.jl"))

@testset "ising2d_logml_convention_single_site" begin
    # Single-site model: EP moment matching is exact, so log_ml + logZ0 must
    # equal the enumerated log Σ_{x=±1} exp(-.5 x'Hx + b'x) to machine
    # precision. Validates the convention used in eval/ising2d.jl:
    #   log Z_Ising = log_ml + logZ0 + 0.5·c·N   (H = c·I − K·A on the hypercube).
    for (c, h) in [(1.6, 0.05), (2.0, 0.7), (5.0, 0.0)]
        H = reshape([c], 1, 1)
        b = [h]
        M = ising_model(H; field=b)
        G = ep(M; backend=:dense, maxcycles=200, tol=1e-12, verbose=false)
        logZ0 = GaussianEP.safe_log_partition(H, b)
        enum = log(exp(-0.5 * c + h) + exp(-0.5 * c - h))
        @test abs(log_ml(G) + logZ0 - enum) < 1e-10
    end
end

@testset "ising2d_ep_vs_onsager_L16" begin
    # L=16, K=0.3, h=0: EP logml/site within 2% of the Onsager free energy;
    # EP vs BP marginals agree to 1e-2 (both exactly 0 at h=0 by symmetry);
    # EP vs Bethe logml within 2%.
    L, K = 16, 0.3
    r = ep_ising2d(L, K, 0.0)
    lz_ons = onsager_logZ_site(K)
    @test abs(r.logZ_per_site - lz_ons) / abs(lz_ons) < 0.02
    lz_bp, m_bp, v_bp = ising2d_bp(L, K, 0.0)
    @test maximum(abs.(r.mean .- m_bp)) < 1e-2
    @test abs(r.logZ_per_site - lz_bp / (L * L)) < 0.02
end

@testset "ising2d_ep_vs_exact_L4_field" begin
    # L=4, K=0.2, h=0.05 vs exact enumeration (2^16 states): EP logml/site
    # within 0.5% and magnetization within 0.02 of exact. (At this K, h the
    # exact mag is 0.1386; EP gives 0.1328, BP only 0.0823 — EP beats BP.)
    L, K, h = 4, 0.2, 0.05
    r = ep_ising2d(L, K, h)
    lz_ex, mu_ex = ising2d_exact(L, K, h)
    @test abs(r.logZ_per_site - lz_ex / (L * L)) < 0.005
    @test abs(mean(r.mean) - mean(mu_ex)) < 0.02
end

@testset "ising2d_ep_vs_bp_marginals_field" begin
    # L=16, K=0.2, h=0.05: EP and BP marginals agree to 0.1 (loose — the two
    # approximations differ by ~5e-2 here; at h=0 they agree exactly).
    L, K, h = 16, 0.2, 0.05
    r = ep_ising2d(L, K, h)
    lz_bp, m_bp, v_bp = ising2d_bp(L, K, h)
    @test maximum(abs.(r.mean .- m_bp)) < 0.1
end
