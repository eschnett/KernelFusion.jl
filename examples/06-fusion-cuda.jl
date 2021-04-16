# julia> include("examples/06-fusion-cuda.jl"); FusionCuda.run()

module FusionCuda

using BenchmarkTools
using CUDA
using KernelFusion
using Random

clamp(x, xlo, xhi) = max(xlo, min(xhi, x))

const S = 3                     # for clipping
const niters = 3

function allreduce_block(op, val::T) where {T}
    thread = threadIdx().x
    nthreads = blockDim().x
    maxnthreads = 512
    sm = @cuStaticSharedMem T maxnthreads
    sm[thread] = val
    CUDA.sync_threads()

    dist = 1
    while dist < nthreads
        if (thread - 1) & dist == 0 && thread + dist <= nthreads
            sm[thread] = op(sm[thread], sm[thread + dist])
        end
        CUDA.sync_threads()
        dist <<= 1
    end

    return sm[1]
end

function broadcast_block(val::T) where {T}
    thread = threadIdx().x
    sm = @cuStaticSharedMem T 1
    if thread == 1
        sm[1] = val
    end
    CUDA.sync_threads()
    if thread > 1
        val = sm[1]
    end
    return val
end

function detrend!(::Val{nelems}, intensities::AbstractArray{T,2},
                  weights::AbstractArray{T,2}) where {nelems,T<:Number}
    # @assert size(intensities) == size(weights)
    ntimes, nfreqs = size(intensities)

    nthreads = blockDim().x
    @cuassert ntimes % nthreads == 0
    nelems′ = ntimes ÷ nthreads
    @cuassert nelems′ == nelems

    freq = blockIdx().x
    time0 = threadIdx().x
    @cuassert 1 ≤ time0 ≤ nthreads ≤ ntimes
    @cuassert 1 ≤ freq ≤ nfreqs

    # reduce
    sumw = zero(T)
    sumi = zero(T)
    for elem in 0:(nelems - 1)
        time = time0 + nthreads * elem
        w = weights[time, freq]
        i = intensities[time, freq]
        sumw += w
        sumi += w * i
    end
    sumw = CUDA.reduce_block(+, sumw, zero(T), Val(true))
    sumi = CUDA.reduce_block(+, sumi, zero(T), Val(true))

    # broadcast
    sumw = broadcast_block(sumw)
    sumi = broadcast_block(sumi)

    avgi = sumi / sumw
    for elem in 0:(nelems - 1)
        time = time0 + nthreads * elem
        intensities[time, freq] -= avgi
    end

    return nothing
end

function clip!(::Val{nelems}, intensities::AbstractArray{T,2},
               weights::AbstractArray{T,2}) where {nelems,T<:Number}
    # @assert size(intensities) == size(weights)
    ntimes, nfreqs = size(intensities)

    nthreads = blockDim().x
    @cuassert ntimes % nthreads == 0
    nelems′ = ntimes ÷ nthreads
    @cuassert nelems′ == nelems

    freq = blockIdx().x
    time0 = threadIdx().x
    @cuassert 1 ≤ time0 ≤ nthreads ≤ ntimes
    @cuassert 1 ≤ freq ≤ nfreqs

    # reduce
    sumw = zero(T)
    sumi2 = zero(T)
    for elem in 0:(nelems - 1)
        time = time0 + nthreads * elem
        w = weights[time, freq]
        i = intensities[time, freq]
        sumw += w
        sumi2 += w * i^2
    end
    sumw = CUDA.reduce_block((x, y) -> x + y, sumw, zero(T), Val(true))
    sumi2 = CUDA.reduce_block(+, sumi2, zero(T), Val(true))

    # broadcast
    sumw = broadcast_block(sumw)
    sumi2 = broadcast_block(sumi2)

    sdvi = sqrt(max(0, sumi2 / sumw))
    for elem in 0:(nelems - 1)
        time = time0 + nthreads * elem
        if abs(intensities[time, freq]) ≥ S * sdvi
            weights[time, freq] = 0
        end
    end

    return nothing
end

function process!(::Val{nelems}, intensities::AbstractArray{T,2},
                  weights::AbstractArray{T,2}) where {nelems,T<:Number}
    for iter in 1:niters
        detrend!(Val(nelems), intensities, weights)
        clip!(Val(nelems), intensities, weights)
    end
    return nothing
end

function run_cuda!(intensities, weights)
    ntimes, nfreqs = size(intensities)
    nthreads = 512
    nblocks = nfreqs
    @assert ntimes % nthreads == 0
    nelems = ntimes ÷ nthreads
    @cuda threads = nthreads blocks = nblocks process!(Val(nelems), intensities,
                                                       weights)
    return nothing
end

function run()
    T = Float32
    ntimes = 4096
    nfreqs = 4096
    Random.seed!(0)
    intensities0 = CuArray(randn(T, ntimes, nfreqs))
    weights0 = CuArray(clamp.(randn(T, ntimes, nfreqs) .+ 1, 0, 1))

    intensities = copy(intensities0)
    weights = copy(weights0)
    println("Running CUDA kernel...")
    run_cuda!(intensities, weights)
    println("Waiting for CUDA kernel...")
    synchronize()
    println("sum(intensities)=$(sum(Array(intensities) .% 1))   sum(weights)=$(sum(Array(weights)))")
    println("Success")

    display(@benchmark (CUDA.@sync run_cuda!($intensities, $weights)) setup = (copy!($intensities,
                                                                                     $intensities0);
                                                                               copy!($weights,
                                                                                     $weights0)))
    println("Memory accesses: ", 2 * (sizeof(intensities) + sizeof(weights)),
            " Bytes")

    return nothing
end

end
