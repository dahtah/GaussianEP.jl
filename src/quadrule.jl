struct QuadRule
    x :: Vector{Float64}
    w :: Vector{Float64}
end

degree(qr :: QuadRule) = length(qr.w)
#Gauss-Hermite rule of deg. d for N(0,1) 
function QuadRule(deg)
    x,w=gausshermite(deg);
    QuadRule(sqrt(2)*x ,w/sqrt(π))
end
