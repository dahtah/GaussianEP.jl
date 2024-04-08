
@testset "ep_gaussian_qd" begin
    n = 5
    Ai = [randn(1,2) for _ in 1:n]
    Hi = [rand(1,1) for _ in  1:n]
    ri = [rand(1,1) for _ in 1:n]
    Q0 = τ*Matrix(I,2,2)
    Q= Q0+sum([Ai[i]'*Hi[i]*Ai[i] for i in 1:n])
    r = sum((Ai[i]'*ri[i] for i in 1:n ))[:]
    μ=Q\r
    f = (z,i)->exp(-.5*(z'*Hi[i]*z) + dot(ri[i],z))
    qm = EP.QuadratureMoments(f,1,40)
    S=EP.GenericSites(qm,EP.LinearMaps(Ai))
    G=EP.GenericApprox(S,Q0);
    EP.run_ep!(G,npasses=2,α=0)
    @assert norm(μ-mean(G)) < 1e-5
    @assert norm(Q-inv(cov(G))) < 1e-5
end
