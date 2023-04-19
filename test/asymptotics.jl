#Test that the large-n asymptotics are what we'd expect
@testset "asymptotics" begin
    n=10000
    p=2
    X= randn(p,n)
    wt = randn(p)
    pr = GaussianEP.logistic.(X'*wt)
    y = rand(n) .<= pr
    for t in [0.01,:autotune]
        G = ep_glm(X,y,Logit(),τ=t)
        se = GaussianEP.se(G)
        δ = mean(G)-wt
        @assert all(abs.(δ) .< 6*se)
    end
end
