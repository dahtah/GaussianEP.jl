#Test accuracy in simple cases
@testset "onesite" begin
    n = 1
    X = randn(1,n)
    y = vec(X) .> 0
    τ = .1 #prior precision
    G = ep_glm(X,y,Logit(),τ=τ)
    S = GLMSites(X,y,Logit())
    ll = (x)->GaussianEP.loglik(S,x)

    #use quadrature to test accuracy of approx posterior
    xq = sqrt(1/τ)*S.qr.x #Quadrature nodes for N(0,1/τ) prior
    wq = S.qr.w #Quadrature weights
    f = [exp(ll(x)) for x in xq]
    z = dot(wq,f)
    @assert log_ml(G) ≈ log(z)
    wq /= z
    μ = dot(wq,f .* xq)
    σ2 = dot(wq,f .* (xq .- μ).^2)
    @assert mean(G)[1] ≈ μ
    @assert cov(G)[1,1] ≈ σ2
end
