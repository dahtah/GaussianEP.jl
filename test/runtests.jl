using GaussianEP,LinearAlgebra,Statistics,SparseArrays
using Test

const testdir = dirname(@__FILE__)
tests = ["glm_gaussian","glm_dim1","asymptotics","general_gaussian","product","ising","moments_extra","glm_equiv","parallel","gas","sparse_lm","spike_slab","fractional"]

@testset "GaussianEP" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end
