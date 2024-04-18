#new interface for setting up for glm models

function epglm(X,y,d :: Distribution;invlink=:default,nquad=40,τ0=0.01)
    Ai = UnivariateLinearMaps(X')

    D = typeof(d)
    @show D
    if (invlink == :default)
        if D <: Bernoulli
            invlink = (x)-> 1/(1+exp(-x))
        elseif D <: Poisson
            invlink = (x)->exp(x)
        else
            error("No default inverse link function for distribution $(D)")
        end
    end
    f = (z,i)-> begin
        μ=invlink(z[1])
        pdf(D(μ),y[i])
    end
    qm = QuadratureMoments(f,1,nquad)
    S=GenericSites(qm,Ai)
    G=GenericApprox(S,float(τ0))
end

