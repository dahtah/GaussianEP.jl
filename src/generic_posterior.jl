
function addlowrank!(M :: Matrix{Float64},A,D)
    BLAS.gemm!('N','N',1.,A,D*A',1.,M)
end


#A generalisation of the GLM posterior of the form
#q(x) ∝ exp(-.5 x' Q x + r'*x)
#where Q = Q₀ +  ∑ A_i' H_i A_i
#Each site is assumed to be of the form
#f(x) = g(A_i x)
#where A_i is a matrix of dimension D x n and D is small
struct GenericApprox{Tsig <: AbstractMatrix,Tm,Th,Ts} <: AbstractGaussApprox
    Σ :: Tsig
    μ :: Vector{Tm}
    r :: Vector{Tm}
    logz :: Vector{Tm}
    R :: Matrix{Tm}
    H :: Th
    S :: Ts
end

sites(G :: GenericApprox) = G.S
nsites(G :: GenericApprox) = nsites(sites(G))

function linearshift(G :: GenericApprox)
    G.Σ\G.μ
end

function GenericApprox(S:: GenericSites,τ =0.01)
    m = npred(S)
    GenericApprox(S,τ*Matrix(I,m,m))
end

function GenericApprox(S :: GenericSites,Q0 :: AbstractMatrix)
    n = nsites(S)
    m = npred(S)
    d = dim(S)
    Σ=inv(Q0)
    μ=zeros(m)
    r=zeros(m)
    logz=zeros(n)
    R=zeros(d,n)
    H=LinearMaps([zeros(d,d) for _ in 1:n])
    GenericApprox(Σ,μ,r,logz,R,H,S)
end

function Statistics.mean(G :: GenericApprox)
    copy(G.μ)
end
Statistics.cov(G :: GenericApprox) = G.Σ

#naive implementation
function cavity(G :: GenericApprox,i)
    Qc = inv(Symmetric(G.Σ)) - G.S.A[i]'*G.H[i]*G.S.A[i]
    mc = Qc\(linearshift(G) - G.S.A[i]'*G.R[:,i])
    MvNormal(mc,inv(Symmetric(Qc)))
    #mv,Qc
end


function cavity_marginal(G :: GenericApprox,i)
    Sm = G.S.A[i]*G.Σ*G.S.A[i]'
    mm = G.S.A[i]*G.μ
    Qm = inv(Sm)
    rm = Qm*mm
    S = inv(Qm - G.H[i])
    r = rm - G.R[:,i]
    MvNormal(S*r,S)
    #mv,Qc
end

function compute_site_contribution(G::GenericApprox,i)
    mvc = cavity_marginal(G,i)
    Qc = inv(mvc.Σ)
    rc = Qc*mvc.μ
    mm=TiltedGaussians.moments(mvc,G.S.qr,Base.Fix2(G.S.f,i))
    Qh = inv(mm.C)
    δQ = Qh-Qc
    rh = Qh*mm.m
    δr = rh - rc
    δz = log_partition(Qh,rh)-log_partition(Qc,rc)
    (δz=δz,δr=δr,δQ=δQ)
end


function compute_site_contribution_old(G::GenericApprox,i)
    mc = cavity(G,i)
    TiltedGaussians.contributions_ep(mc,G.S.qr,G.S.A[i],Base.Fix2(G.S.f,i))
end

function update!(G::GenericApprox,i,α=0.0)
    ct = compute_site_contribution(G,i)
    D=(1-α)*(ct.δQ - G.H[i])
    Z = Matrix(G.Σ*G.S.A[i]')

    G.H[i] = α*G.H[i] + (1-α)*ct.δQ
    dr = G.S.A[i]'*(1-α)*(ct.δr - G.R[:,i])
    G.R[:,i] = α*G.R[:,i]+ (1-α)*ct.δr
    addlowrank!(G.Σ,Z,-inv(inv(D)+G.S.A[i]*Z))
    G.r .+= dr
#    mp  =G.Σ*sum((G.S.A[i]'*(G.R[:,i]) for i in 1:nsites(G)))
    G.μ .= G.Σ*G.r
end

function run_ep!(G;α=0.0,npasses=4,schedule=1:nsites(G))
    H = HybridDistr{dim(G.S)}()
    for ip in 1:npasses
        for i in schedule
            compute_update!(H,G,i,α=α)
        end
    end
end

function compute_update!(H :: HybridDistr, G :: GenericApprox,i;α=0.0)
    B = Matrix(G.Σ*G.S.A[i]')
    try
        compute_cavity_marginal!(H,G,i,B)
    catch
        @warn "Site $(i) has non-positive variance, skipping"
        return 
    end
    compute_moments!(H,G,i)
    H.Nh.Q .-= H.Nm.Q
    H.Nh.r .-= H.Nm.r
    D=(1-α)*(H.Nh.Q - G.H[i])
    G.H[i] = α*G.H[i] + (1-α)*H.Nh.Q
    dr = G.S.A[i]'*(1-α)*(H.Nh.r - G.R[:,i])
    G.R[:,i] = α*G.R[:,i]+ (1-α)*H.Nh.r
    addlowrank!(G.Σ,B,-inv(inv(D)+G.S.A[i]*B))
    G.r .+= dr
    #G.μ .= G.Σ*G.r
    BLAS.gemv!('N',1.0,G.Σ,G.r,0.0,G.μ)
    return
end

function compute_cavity_marginal!(H :: HybridDistr, G :: GenericApprox,i,buf)
    H.Nm.Q .= inv(Matrix(G.S.A[i]*buf)) - G.H[i]
    H.Nm.r .= inv(Matrix(G.S.A[i]*buf))*(G.S.A[i]*G.μ) - G.R[:,i]
    moments_from_exp!(H.Nm)
end


function chol_lower(a::Cholesky{F, T}) where F where T
    return a.uplo === 'L' ? LowerTriangular{F,T}(a.factors) : LowerTriangular{F,T}(a.factors')
end


function compute_moments!(H :: HybridDistr{D},G :: GenericApprox,ind) where D
    compute_moments!(H,G.S,ind)
    return
end

#Compute contribution of site number i and store in H.Nh.Q and H.Nh.r 
function compute_site_contribution!(H :: HybridDistr, G :: GenericApprox,i)
    buf = G.Σ*G.S.A[i]'
    compute_cavity_marginal!(H,G,i,buf)
    compute_moments!(H,G,i)
    H.Nh.Q .-= H.Nm.Q
    H.Nh.r .-= H.Nm.r
    return
end
