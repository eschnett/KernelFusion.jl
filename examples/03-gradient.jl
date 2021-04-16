# julia> include("examples/03-gradient.jl"); kernel = Gradient.make(); Gradient.run(kernel)

module Gradient

using BenchmarkTools
using InteractiveUtils
using KernelFusion
using ReverseDiff
using StaticArrays

function make()
    # Define some code. The result of a `quote` is a Julia expression. We could read JSON instead.
    code = quote
        function (params, x)
            add1(x) = x + params.offset
            times2′(x) = times2(params, x)
            return (add1 ∘ (x -> max(300, x)) ∘ times2′)(x)
        end
    end
    # This creates actual Julia code from Julia expressions. It's not compiled yet; that only happens when the kernel is called.
    kernel = make_kernel(code)

    return kernel
end

function run(kernel)
    # Define a 2d array
    npoints = 300
    A = Float32[i + 100j for i in 1:npoints, j in 1:npoints]
    B = copy(A)

    # Run the kernel on the 2d array
    params = (factor=2, offset=1)
    run_kernel!(kernel, params, B)

    # Check the result
    @assert B == max.(300, 2 .* A) .+ 1

    function cost(params′)
        local params = (factor=params′[1], offset=params′[2])
        local B1 = kernel(params, A[1])
        local B = typeof(B1).(copy(A))
        run_kernel!(kernel, params, B)
        return sum((B .- 400) .^ 2)
    end
    params′ = SVector{2,Float32}(params.factor, params.offset)

    # We could also use ForwardDiff instead
    grad_cost = params′ -> ReverseDiff.gradient(cost, params′)

    # Calculate gradient via finite differences
    function grad(f, x, δ)
        return SVector{2,Float32}((f(x + [(j == i) * δ for j in 1:length(x)]) -
                                   f(x - [(j == i) * δ for j in 1:length(x)])) /
                                  2δ for i in 1:length(x))
    end
    δ = 0.001
    # @show grad(cost, params′, δ)
    @assert grad_cost(params′) ≈ grad(cost, params′, δ)
    println("Test: Success")

    return nothing
end

end
