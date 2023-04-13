# GaussianEP.jl

An experimental package for inference in latent Gaussian models via Expectation Propagation (Minka, 2001).

Expectation Propagation (EP) can be used to form a Gaussian approximation to a distribution. Here, the target distribution is the distribution over the coefficients (weights) of a Generalised Linear Model, for instance a logistic regression.

In GLMs, EP is typically much more accurate than other variational approximations, and much faster than MCMC.

## Usage 

Running a logistic regression is easy. Let us first generate some data. 
```{julia}
n = 100
X = randn(2,n) #regression matrix. Note that X is p × n contrary to common convention!
φ = (x)-> 1/(1+exp(-x)) #logistic function
w_true = [-1,1] #"true" coefficients
prob = φ.(X'*w_true) #prob of observing a 1
y = rand(n) .<= prob #observations ("labels")
```

Then we can run GaussianEP:
```{julia}
G = ep_glm(X,y,Logit())
mean(G) #Approx posterior mean
cov(G) #Approx posterior covariance
```

## Details

### Prior

The default prior over the weights is a zero-mean Gaussian with covariance $\mathbf{I}\tau^{-1}$. $\tau$ is a prior precision and can be set:
```{julia}
G = ep_glm(X,y,Logit(),τ=10.)
```
The (log) marginal likelihood is also approximated and can be extracted via:
```{julia}
log_ml(G)
```
This allows for hyperparameter selection on τ. 

### Likelihoods 

The likelihood is set via the third argument of ep_glm: 
```{julia}
G = ep_glm(X,y,PoisLik()) #switch to Poisson lik. with mean exp(η)
```

It is possible to define your own custom likelihood function. For instance, here we define a likelihood that is logistic for some observations and Poisson for others: 
```{julia}
struct MixedLik{L1,L2} <: GaussianEP.Likelihood
    lik1 :: L1
    lik2 :: L2
end

function GaussianEP.log_dens(ml::MixedLik, η, y)
    return (y.type==1 ? GaussianEP.log_dens(ml.lik1,η,y.value) : GaussianEP.log_dens(ml.lik2,η,y.value))
end

#The observations y can be a vector of anything
#Here we make it a vector of tuples (type,value)
y = [(type=i,value=rand() > .5) for _ in 1:10 for i in 1:2]
X = randn(2,length(y))
G = ep_glm(X,y,MixedLik(Logit(),PoisLik()))
```

## References

A few useful references:

Minka, T. P. (2001). A family of algorithms for approximate Bayesian inference (Doctoral dissertation, Massachusetts Institute of Technology).
Herbrich (2005), On Gaussian Expectation Propagation, https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=17d40397aeedaeb61a2324ec52edaecb93093ca5
Dehaene, G., & Barthelmé, S. (2018). Expectation propagation in the large data limit. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 80(1), 199-217.

