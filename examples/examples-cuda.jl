module KernelFusionExamples

using CUDA
using ForwardDiff
using InteractiveUtils
using KernelFusion
using StaticArrays

################################################################################

function make_simple_kernel()
    # Define some code. The result of a `quote` is a Julia expression. We could read JSON instead.
    code = quote
        add1(x) = x + 1
        add1 ∘ (x -> max(300, x)) ∘ times2
    end
    # This creates actual Julia code from Julia expressions. It's not compiled yet; that only happens when the kernel is called.
    kernel = make_kernel(code)

    return kernel
end

function run_simple_kernel(kernel)
    # Define a 2d array
    npoints = 3
    A = Float32[i + 100j for i in 1:npoints, j in 1:npoints]
    B = CuArray(A)

    # Run the kernel on the 2d array
    run_kernel!(kernel, B)

    # Check the result
    @assert Array(B) == max.(300, 2 .* A) .+ 1

    # show_kernel_code(kernel, B)
end

################################################################################

function make_parameterized_kernel()
    code = quote
        function (params, x)
            add1(x) = x + params.offset
            times2′(x) = times2(params, x)
            return (add1 ∘ (x -> max(300, x)) ∘ times2′)(x)
        end
    end
    kernel = make_kernel(code)

    return kernel
end

function run_parameterized_kernel(kernel)
    npoints = 3
    A = Float32[i + 100j for i in 1:npoints, j in 1:npoints]
    B = CuArray(A)

    params = (factor=2, offset=1)
    run_kernel!(kernel, params, B)

    @assert Array(B) == max.(300, 2 .* A) .+ 1

    # show_kernel_code(kernel, params, B)
end

################################################################################

function make_kernel_derivative()
    code = quote
        function (params, x)
            add1(x) = x + params.offset
            times2′(x) = times2(params, x)
            # logsumexp(x, y) = log2(exp2(x) + exp2(y)) # smooth max
            # return (add1 ∘ (x -> logsumexp(300, x)) ∘ times2′)(x)
            return (add1 ∘ (x -> max(300, x)) ∘ times2′)(x)
        end
    end
    kernel = make_kernel(code)

    return kernel
end

function run_kernel_derivative(kernel)
    npoints = 3
    A = Float32[i + 100j for i in 1:npoints, j in 1:npoints]
    params = (factor=2, offset=1)

    function cost(params′)
        params = (factor=params′[1], offset=params′[2])
        B1 = kernel(params, A[1])
        B = typeof(B1).(copy(A))
        run_kernel!(kernel, params, B)
        return sum((B .- 400) .^ 2)
    end
    params′ = SVector{2,Float32}(params.factor, params.offset)
    # @show cost(params′)

    grad_cost = params′ -> ForwardDiff.gradient(cost, params′)
    # @show grad_cost(params′)
    # Calculate gradient via finite differences
    function grad(f, x, δ)
        return SVector{2,Float32}((f(x + [(j == i) * δ for j in 1:length(x)]) -
                                   f(x - [(j == i) * δ for j in 1:length(x)])) /
                                  2δ for i in 1:length(x))
    end
    δ = 0.001
    # @show grad(cost, params′, δ)
    @assert grad_cost(params′) ≈ grad(cost, params′, δ)
end

end
