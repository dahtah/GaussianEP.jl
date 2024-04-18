function glmlik(d :: D, z :: T, y, invlink) where T where D 
    pdf(D(invlink(z)),y) :: T
end


#new interface for setting up for glm models
function epglm(X,y,d :: Distribution;invlink=:default,nquad=40,τ0=0.01)
    τ0 = float(τ0)
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
    f = (z,i)-> glmlik(d,z[1],y[i],invlink)
    qm = QuadratureMoments(f,1,nquad)
    S=GenericSites(qm,Ai)
    G=GenericApprox(S,float(τ0))
end

