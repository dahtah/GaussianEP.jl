#Check that EP returns the correct quantities for a Gaussian posterior (EP is exact in this case)
EP = GaussianEP

#For debugging purposes: compute exactly the moments of a
#f(x,j) = exp(-.5*j*x'*x - j*r)
#under a N(μ,σ2) Gaussian
function moments_gaussian(μ,σ2,j)
    N = Normal(μ,σ2)
    q = 1/σ2
    r = q*μ
    qt = q+j
    rt = r-j
    z = exp(EP.log_partition(qt,rt) - EP.log_partition(q,r))
    sh = 1/qt
    rh = sh*rt
    (z,rh,sh)
end


function check_EP_gaussian(n,d;τ=0.01,α=0,npasses=10,univar=true)
    Ai = [randn(1,d) for _ in 1:n]
    Q0= τ*Matrix(I(d))
    r0 = randn(d)
    Qt = Q0+sum([i*Ai[i]'*Ai[i] for i in 1:n]) #True posterior precision
    rt = vec(r0-sum([Ai[i]'*i for i in 1:n])) #True posterior shift
    #Set up moment computation
    mm = EP.AnalyticMoments((μ,σ,i)->moments_gaussian(μ[1],σ[1],i))
    #Two possible ways of setting up the linear maps
    M = (univar ?  EP.UnivariateLinearMaps(Matrix(reduce(vcat,Ai)')) :  EP.LinearMaps(Ai))
    S=EP.GenericSites(mm,M);
    G=EP.GenericApprox(S,Q0,r0);
    EP.run_ep!(G,npasses=npasses,α=α)
    mt = Qt\rt
    logz = EP.log_partition(Qt,rt)-EP.log_partition(Q0,r0)
    abs(logz-EP.logz(G)),norm(mt-mean(G)),norm(Qt-inv(cov(G)))
end

@testset "ep_gaussian" begin
    for d = [2,3,4]
        for α = [0.,0.2,0.5]
            for n = 1:5
                for τ = [0.1,2.]
                    for uni = [true,false]
                        a,b,c = check_EP_gaussian(n,d,α=α,τ=τ,npasses=50,univar=uni)
                        @assert a < 1e-8 #log partition function is correct
                        @assert b < 1e-8 #mean is correct
                        @assert c < 1e-8 #cov is correct
                    end
                end
            end
        end
    end
end
