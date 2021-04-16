module KernelFusion

using InteractiveUtils

times2(x) = 2 * x

export make_kernel
make_kernel(expr::Expr) = eval(expr)

export run_kernel!
run_kernel!(kernel, A::Array{T,2}) where {T} = map!(kernel, A, A)
# function run_kernel!(kernel, A::Array{T,2}) where {T}
#     @inbounds for j in 1:size(A, 2)
#         @simd for i in 1:size(A, 1)
#             A[i, j] = kernel(A[i, j])
#         end
#     end
# end

export show_kernel_code
function show_kernel_code(kernel, A::Array{T,2}) where {T}
    # @code_native run_kernel!(kernel, A)
    @code_native map!(kernel, A, A)
end

################################################################################

times2(params, x) = params.factor * x

export run_kernel!
function run_kernel!(kernel, params, A::Array{T,2}) where {T}
    return map!(x -> kernel(params, x), A, A)
end
# function run_kernel!(kernel, A::Array{T,2}) where {T}
#     @inbounds for j in 1:size(A, 2)
#         @simd for i in 1:size(A, 1)
#             A[i, j] = kernel(A[i, j])
#         end
#     end
# end

export show_kernel_code
function show_kernel_code(kernel, params, A::Array{T,2}) where {T}
    # @code_native run_kernel!(kernel, params, A)
    @code_native map!(x -> kernel(params, x), A, A)
end

end
