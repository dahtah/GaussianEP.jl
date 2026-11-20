# EP on a model whose factors are all Gaussian must recover the exact
# Gaussian posterior. Tests rank-1 and rank-2 sites, and dense / diagonal /
# sparse prior precision (Q0's type is a parameter of EPModel).
@testset "general_gaussian" begin
    m, n = 4, 6
    As = [randn(2, m) for _ in 1:n]
    ys = [randn(2) for _ in 1:n]
    τ = 0.5
    Qstar = τ*Matrix(I,m,m) + sum(A'*A for A in As)
    rstar = sum(A'*y for (A,y) in zip(As, ys))
    for Q0 in (τ*Matrix(I,m,m), Diagonal(τ*ones(m)), sparse(τ*I, m, m))
        M = EPModel(Q0, [Site(As[i], ys[i], GaussLik()) for i in 1:n])
        @test typeof(M.Q0) == typeof(Q0)
        G = ep(M)
        @test mean(G) ≈ Qstar\rstar atol=1e-8
        @test cov(G) ≈ inv(Qstar) atol=1e-8
    end

    # rank-1 Gaussian sites
    p, k = 3, 5
    X = randn(p, k)
    y = randn(k)
    M = EPModel(τ*Matrix(I,p,p), [Site(reshape(X[:,i],1,p), y[i], GaussianEP.GLik()) for i in 1:k])
    G = ep(M)
    Qs = τ*Matrix(I,p,p) + X*X'
    @test mean(G) ≈ Qs\(X*y) atol=1e-8
    @test cov(G) ≈ inv(Qs) atol=1e-8
end
