using GaussianEP,LinearAlgebra,Statistics
using Test

const testdir = dirname(@__FILE__)
tests = ["glm_gaussian","glm_dim1","asymptotics","test_gen"]

@testset "GaussianEP" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end
