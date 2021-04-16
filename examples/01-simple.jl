# julia> include("examples/01-simple.jl"); kernel = Simple.make(); Simple.run(kernel)

module Simple

using BenchmarkTools
using InteractiveUtils
using KernelFusion

function make()
    # Define some code. The result of a `quote` is a Julia expression. We could read JSON instead.
    code = quote
        add1(x) = x + 1
        add1 ∘ (x -> max(300, x)) ∘ times2
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
    run_kernel!(kernel, B)

    # Check the result
    @assert B == max.(300, 2 .* A) .+ 1
    println("Test: Success")

    display(@benchmark run_kernel!($kernel, B) setup = (B = copy($A)))
    println("Memory accesses: ", 2 * sizeof(A), " Bytes")

    # show_kernel_code(kernel, B)

    return nothing
end

end
