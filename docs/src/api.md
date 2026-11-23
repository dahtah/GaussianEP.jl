# API

## Core EP loop

```@docs
ep
GaussianEP.fit!
log_ml
```

## Models and sites

```@docs
EPModel
Site
```

## Quadrature rules

```@docs
QuadRule
trap_rule
```

## Tilted moments

```@docs
tilted_moments
```

## Likelihoods

```@docs
Logit
PoisLik
ProbitLik
HardSphere
IsingLik
GaussLik
RepulseLik
ConfineLik
LaplaceLik
SpikeSlabLik
```

## Model builders

```@docs
glm_model
ising_model
gas_model
sparse_lm
spike_slab_lm
```

## Hyperparameter selection

```@docs
glm_regpath
glm_autotune
```

## Legacy GLM interface

```@docs
EPApprox
ep_glm
GLMSites
se
```
