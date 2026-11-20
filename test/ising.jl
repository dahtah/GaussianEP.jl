# Ising model: p(x) ∝ exp(-.5 x'Hx + b'x) ∏_i I(|x_i| = 1).
# Site moments are analytic (E[η] = tanh(m/s²), E[η²] = 1).
@testset "ising_1d" begin
    h, b = 2.0, 0.7
    M = EPModel(reshape([h],1,1), [Site(reshape([1.0],1,1), nothing, IsingLik())]; r0=[b])
    G = ep(M)
    # with a single site EP moment matching is exact; the true moments of
    # the two-point posterior are tanh(b) and 1 - tanh(b)²
    @test mean(G)[1] ≈ tanh(b) atol=1e-10
    @test cov(G)[1,1] ≈ 1 - tanh(b)^2 atol=1e-10
end

@testset "ising_2d" begin
    H = [2.0 0.5; 0.5 2.0]
    M = ising_model(H)
    G = ep(M)
    # exact posterior by enumeration over the 4 vertices of the hypercube
    vs = [collect(x) for x in Iterators.product([-1.0, 1.0], [-1.0, 1.0])]
    lw = [-.5*dot(x, H*x) for x in vs]
    w = exp.(lw .- GaussianEP.logsumexp(lw))
    μ = sum(w[i] .* vs[i] for i in eachindex(vs))
    C = sum(w[i] .* (vs[i]*vs[i]') for i in eachindex(vs)) .- μ*μ'
    # EP is only approximate here (coupled sites), hence loose tolerances
    @test isapprox(mean(G), μ, atol=0.1)
    @test isapprox(cov(G), C, atol=0.15)
end
