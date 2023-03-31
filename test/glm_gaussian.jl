#Fit EP approx to a Gaussian posterior, should be exact

@testset "ep_gaussian" begin
    X=float.([ones(1,3); (1:3)'])
    y = float.([0,1,2]);
    τ=.1
    G=ep_glm(X,y,GaussianEP.GLik(),τ=τ)
    C_th = inv(X*X'+τ*I)
    @assert norm(C_th - cov(G)) < 1e-3
    #NB: we can't expect very high accuracy due to quadrature
    m_th = C_th*X*y
    @assert norm(m_th - mean(G)) < 1e-3
    @assert abs(log_ml(G)-GaussianEP.log_ml_gaussian(G,y)) < 1e-3
end
