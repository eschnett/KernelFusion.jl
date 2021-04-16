# julia> include("examples/02-parameters.jl"); kernel = Parameters.make(); Parameters.run(kernel)

module Parameters

using BenchmarkTools
using InteractiveUtils
using KernelFusion

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
    npoints = 3000
    A = Float32[i + 100j for i in 1:npoints, j in 1:npoints]
    B = copy(A)

    # Run the kernel on the 2d array
    params = (factor=2, offset=1)
    run_kernel!(kernel, params, B)

    # Check the result
    @assert B == max.(300, 2 .* A) .+ 1
    println("Test: Success")

    display(@benchmark run_kernel!($kernel, $params, B) setup = (B = copy($A)))
    println("Memory accesses: ", 2 * sizeof(A), " Bytes")

    # show_kernel_code(kernel, params, B)

    return nothing
end

end
