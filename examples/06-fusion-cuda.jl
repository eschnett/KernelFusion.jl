# julia> include("examples/06-fusion-cuda.jl"); FusionCuda.run()

module FusionCuda

using BenchmarkTools
using CUDA
using KernelFusion

clamp(x, xlo, xhi) = max(xlo, min(xhi, x))

const S = 3                     # for clipping
const niters = 3

function detrend!(intensities::CuArray{T,2},
                  weights::CuArray{T,2}) where {T<:Number}
    # @assert size(intensities) == size(weights)
    ntimes, nfreqs = size(intensities)

    freq = blockIdx().x + 1
    time = threadIdx().x + 1

    wtmp = @cuStaticSharedMem(T, ntimes)
    itmp = @cuStaticSharedMem(T, ntimes)
    w = weights[time, freq]
    i = intensities[time, freq]
    wtmp[time] = w
    itmp[time] = i

    sumwtmp = @cuStaticSharedMem(T, ntimes)
    sumitmp = @cuStaticSharedMem(T, ntimes)
    sumwtmp[time] = w
    sumitmp[time] = w * i^2
    CUDA.sync_threads()

    dist = 1
    while dist < ntimes
        if (time - 1) & dist == 0 && time + dist <= ntimes
            sumwtmp[time] += sumwtmp[time + dist]
            sumitmp[time] += sumitmp[time + dist]
            CUDA.sync_threads()
        end
        dist <<= 1
    end

    sumw = sumwtmp[1]
    sumi = sumitmp[1]

    avgi = sumi / sumw
    intensities[time, freq] -= avgi

    return nothing
end

function clip!(intensities::CuArray{T,2},
               weights::CuArray{T,2}) where {T<:Number}
    @assert size(intensities) == size(weights)
    ntimes, nfreqs = size(intensities)

    @inbounds for freq in 1:nfreqs
        sumw = zero(T)
        for time in 1:ntimes
            sumw += weights[time, freq]
        end

        sumi2 = zero(T)
        for time in 1:ntimes
            sumi2 += weights[time, freq] * intensities[time, freq]^2
        end

        sdvi = sqrt(max(0, sumi2 / sumw))
        for time in 1:ntimes
            if abs(intensities[time, freq]) ≥ S * sdvi
                weights[time, freq] = 0
            end
        end
    end

    return nothing
end

function process!(intensities::CuArray{T,2},
                  weights::CuArray{T,2}) where {T<:Number}
    for iter in 1:niters
        detrend!(intensities, weights)
        # clip!(intensities, weights)
    end
    return nothing
end

function run()
    T = Float32
    ntimes = 4096
    nfreqs = 4096
    intensities0 = CuArray(randn(T, ntimes, nfreqs))
    weights0 = CuArray(clamp.(randn(T, ntimes, nfreqs) .+ 1, 0, 1))

    intensities = copy(intensities0)
    weights = copy(weights0)
    @cuda threads = size(intensities, 2) process!(intensities, weights)

    display(@benchmark (CUDA.@sync @cuda threads = size(intensities, 2) process!($intensities,
                                                                                 $weights)) setup = (copy!($intensities,
                                                                                                           $intensities0);
                                                                                                     copy!($weights,
                                                                                                           $weights0)))
    println("Memory accesses: ", 2 * (sizeof(intensities) + sizeof(weights)),
            " Bytes")

    return nothing
end

end
