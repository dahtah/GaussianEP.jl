using GaussianEP,LinearAlgebra,Statistics,Distributions
using Test

const testdir = dirname(@__FILE__)
#tests = ["glm_gaussian","glm_dim1","asymptotics","test_gen","ep_gaussian"]
tests = ["gaussian_exact"]
@testset "GaussianEP" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end
