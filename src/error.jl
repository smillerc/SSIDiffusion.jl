
function stage_convergence(u1, un, nghost)
    u1_denom = 0.0
    CartInd = getcartind(u1, nghost)

    @inbounds for I in CartInd
        u1_denom += u1[I]^2
    end
    u1_denom = sqrt(u1_denom)

    if isinf(u1_denom) || iszero(u1_denom)
        resid = -Inf
    else
        numerator = 0.0

        @inbounds for I in CartInd
            numerator += (un[I] - u1[I])^2
        end

        resid = sqrt(numerator) / u1_denom
    end

    @show u1_denom, numerator, resid
    return resid
end

@inline function embedded_error_estimate(
    u0::AbstractArray{T,N}, u1, u2, rtol, nghost
) where {T,N}
    machine_eps = 5eps(Float64)
    atol = rtol * 1e-2
    err = 0.0

    CartInd = getcartind(u0, nghost)

    @inbounds for I in CartInd
        # tol = atol + max(abs(u2[I]), abs(u0[I])) * rtol
        tol = atol + max(abs(u1[I]), abs(u0[I])) * rtol
        du = u1[I] - u2[I]
        du = du * (abs(du) > machine_eps)
        err += abs(du / tol)^2
    end

    return err = sqrt(err / length(CartInd))
end

@inline function getlims(x, nghost)
    lo = first.(axes(x)) .+ nghost
    hi = last.(axes(x)) .- nghost
    return lo, hi
end

@inline function getlooplims(x, nghost)
    lo, hi = getlims(x, nghost)
    N = length(lo)
    return ntuple(i -> lo[i]:hi[i], N)
end

@inline function getcartind(x, nghost)
    lims = getlooplims(x, nghost)
    return CartesianIndices(lims)
end
