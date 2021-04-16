# julia> include("examples/05-fusion.jl"); Fusion.run()

module Fusion

using BenchmarkTools
using InteractiveUtils
using KernelFusion

clamp(x, xlo, xhi) = max(xlo, min(xhi, x))

const S = 3                     # for clipping
const niters = 3

function detrend!(intensities::AbstractArray{T,2},
                  weights::AbstractArray{T,2}) where {T<:Number}
    @assert size(intensities) == size(weights)
    ntimes, nfreqs = size(intensities)

    @inbounds for freq in 1:nfreqs
        sumw = zero(T)
        for time in 1:ntimes
            sumw += weights[time, freq]
        end

        sumi = zero(T)
        for time in 1:ntimes
            sumi += weights[time, freq] * intensities[time, freq]
        end

        avgi = sumi / sumw
        for time in 1:ntimes
            intensities[time, freq] -= avgi
        end
    end

    return nothing
end

function clip!(intensities::AbstractArray{T,2},
               weights::AbstractArray{T,2}) where {T<:Number}
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

function process!(intensities::AbstractArray{T,2},
                  weights::AbstractArray{T,2}) where {T<:Number}
    for iter in 1:niters
        detrend!(intensities, weights)
        clip!(intensities, weights)
    end
    return nothing
end

# Manually optimized
function process_fast!(intensities::AbstractArray{T,2},
                       weights::AbstractArray{T,2}) where {T<:Number}
    @assert size(intensities) == size(weights)
    ntimes, nfreqs = size(intensities)

    @inbounds for freq in 1:nfreqs
        for iter in 1:niters
            sumw = zero(T)
            sumi = zero(T)
            @simd for time in 1:ntimes
                sumw += weights[time, freq]
                sumi += weights[time, freq] * intensities[time, freq]
            end

            avgi = sumi / sumw
            @simd for time in 1:ntimes
                intensities[time, freq] -= avgi
            end

            sumi2 = zero(T)
            @simd for time in 1:ntimes
                sumi2 += weights[time, freq] * intensities[time, freq]^2
            end

            sdvi = sqrt(max(0, sumi2 / sumw))
            @simd for time in 1:ntimes
                if abs(intensities[time, freq]) ≥ S * sdvi
                    weights[time, freq] = 0
                end
            end
        end
    end

    return nothing
end

function process_fast2!(intensities::AbstractArray{T,2},
                        weights::AbstractArray{T,2}) where {T<:Number}
    @assert size(intensities) == size(weights)
    ntimes, nfreqs = size(intensities)

    @inbounds for freq in 1:nfreqs
        for iter in 1:niters
            sumw = zero(T)
            sumi = zero(T)
            sumi2 = zero(T)
            @simd for time in 1:ntimes
                sumw += weights[time, freq]
                sumi += weights[time, freq] * intensities[time, freq]
                sumi2 += weights[time, freq] * intensities[time, freq]^2
            end
            avgi = sumi / sumw
            sdvi = sqrt(max(0, sumi2 / sumw - avgi^2))

            @simd for time in 1:ntimes
                intensities[time, freq] -= avgi
                if abs(intensities[time, freq]) ≥ S * sdvi
                    weights[time, freq] = 0
                end
            end
        end
    end

    return nothing
end

function run()
    T = Float32
    ntimes = 4096
    nfreqs = 4096
    intensities0 = randn(T, ntimes, nfreqs)
    weights0 = clamp.(randn(T, ntimes, nfreqs) .+ 1, 0, 1)

    intensities = copy(intensities0)
    weights = copy(weights0)
    process!(intensities, weights)
    good_intensities = copy(intensities)
    good_weights = copy(weights)

    copy!(intensities, intensities0)
    copy!(weights, weights0)
    process_fast!(intensities, weights)
    @assert intensities ≈ good_intensities
    @assert weights ≈ good_weights

    copy!(intensities, intensities0)
    copy!(weights, weights0)
    process_fast2!(intensities, weights)
    @assert intensities ≈ good_intensities
    @assert weights ≈ good_weights

    display(@benchmark process!($intensities, $weights) setup = (copy!($intensities,
                                                                       $intensities0);
                                                                 copy!($weights,
                                                                       $weights0)))
    display(@benchmark process_fast!($intensities, $weights) setup = (copy!($intensities,
                                                                            $intensities0);
                                                                      copy!($weights,
                                                                            $weights0)))
    display(@benchmark process_fast2!($intensities, $weights) setup = (copy!($intensities,
                                                                             $intensities0);
                                                                       copy!($weights,
                                                                             $weights0)))
    println("Memory accesses: ", 2 * (sizeof(intensities) + sizeof(weights)),
            " Bytes")

    return nothing
end

end
