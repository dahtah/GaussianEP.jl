#should use BLAS (syrk! ?)
function addlowrank!(M :: AbstractMatrix,A,sign=+1)
    if sign == 1
        M .= M+ A*A'
    elseif sign == -1
        M .= M - A*A'
    end
end

function addlowrank!(M :: AbstractMatrix,A :: AbstractVecOrMat,B :: AbstractVecOrMat,sign=+1)
    if sign == 1
        M .= M + A*B
    elseif sign == -1
        M .= M - A*B
    end
end


function addlowrank!(M :: Matrix{Float64},A :: AbstractVecOrMat,B :: AbstractVecOrMat,sign=+1)
    BLAS.gemm!('N','N',float(sign),Matrix(A),Matrix(B),1.0,M)
end



#A generalisation of the GLM posterior of the form
#q(x) ∝ exp(-.5 x' Q x + r'*x)
#where Q = Q₀ +  ∑ A_i' H_i A_i
#Each site is assumed to be of the form
#f(x) = g(A_i x)
#where A_i is a matrix of dimension D x n and D is small
struct GenericApprox{Tsig <: AbstractMatrix,Tm,Th,Ts} <: AbstractGaussApprox
    Σ :: Tsig
    Q0 :: Matrix{Tm}
    r0 :: Vector{Tm}
    μ :: Vector{Tm}
    r :: Vector{Tm}
    logz :: Vector{Tm}
    R :: Matrix{Tm}
    H :: Th
    S :: Ts
end


dim(G :: GenericApprox) = length(G.μ)
sites(G :: GenericApprox) = G.S
nsites(G :: GenericApprox) = nsites(sites(G))

#Recompute Σ and μ from site parameters
function recompute_from_site_params!(G :: GenericApprox)
    Q = G.Q0+sum((G.S.A[i]'*(G.H[i])*G.S.A[i] for i in 1:nsites(G)))
    r = G.r0 + sum((G.S.A[i]'*(G.R[:,i]) for i in 1:nsites(G)))
    G.Σ .= inv(Q)
    G.r .= r
    G.μ .= G.Σ*G.r
end

function linearshift(G :: GenericApprox)
    G.Σ\G.μ
end

function GenericApprox(S:: GenericSites,τ =0.01)
    m = npred(S)
    GenericApprox(S,τ*Matrix(I,m,m))
end

function GenericApprox(S :: GenericSites,Q0 :: AbstractMatrix{T},r0 = zeros(T,size(Q0,1))) where T
    n = nsites(S)
    m = npred(S)
    d = outdim(S)
    Σ=inv(Q0)
    μ=Σ*r0
    r=copy(r0)
    logz=zeros(T,n)
    R=zeros(T,d,n)
    H=LinearMaps([zeros(T,d,d) for _ in 1:n])
    GenericApprox(Σ,Q0,r0,μ,r,logz,R,H,S)
end

function Statistics.mean(G :: GenericApprox)
    copy(G.μ)
end
Statistics.cov(G :: GenericApprox) = G.Σ

function Base.show(io::IO, G::GenericApprox)
    println("Generic Gaussian approximation in dimension $(dim(G)).")
end


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

function run_ep!(G;α=0.0,npasses=4,schedule=1:nsites(G))
    H = HybridDistr{outdim(G.S),eltype(G.μ)}()
    for ip in 1:npasses
        for i in schedule
            compute_update!(H,G,i,α=α)
        end
    end
end

function compute_update!(H :: HybridDistr{D,M,F}, G :: GenericApprox,i;α=0.0) where D where M where F
    Ai = G.S.A[i]
    B = Matrix(G.Σ*Ai')
    Σtot = Symmetric(Ai*B)
    compute_cavity_marginal!(H,G,i,B)
    compute_moments!(H,G,i)
    exp_from_moments!(H.Nh) #compute exponential parameters

    G.logz[i] = H.logzh + log_partition(H.Nm) - log_partition(H.Nh)
    H.Nh.Q .-= H.Nm.Q #contribution to precision from site
    H.Nh.r .-= H.Nm.r #contribution to shift
    #@show H.Nh.r


    δH =(1-α)*(H.Nh.Q - G.H[i])
    G.H[i] .+= δH
    dri = (1-α)*(H.Nh.r - G.R[:,i])
    G.R[:,i] .+= dri
    S = I+Σtot*δH
    L = G.Σ*Ai'
    addlowrank!(G.Σ,L*δH,S\L',-1)
    G.r .+= Ai'*dri
    G.μ .= G.Σ*G.r
    return
end

function compute_cavity_marginal!(H :: HybridDistr, G :: GenericApprox,i,buf)
    #@show Matrix(G.S.A[i]*buf)
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
    exp_from_moments!(H.Nh)
    H.Nh.Q .-= H.Nm.Q
    H.Nh.r .-= H.Nm.r
    return
end

function logz(G :: GenericApprox)
    n = size(cov(G),1)
    log_partition(Symmetric(inv(G.Σ)),G.r)-log_partition(Symmetric(G.Q0),G.r0) + sum(G.logz)
end
