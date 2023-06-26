
"""Apply boundary conditions"""
function applybc!(
    u, bcs::NamedTuple{(:ilo, :ihi),Tuple{BC1,BC2}}, nghost::Int=1
) where {BC1,BC2}

    # ilo
    if bcs.ilo === :periodic
        periodic_iloihi(u, nghost)
    elseif bcs.ilo === :reflect
        reflect_ilo(u, nghost)
    elseif bcs.ilo === :zeroflux
        zeroflux_ilo(u, nghost)
    elseif bcs.ilo isa Tuple{Symbol,Number}
        fixed_ilo(u, bcs.ilo[2], nghost)
    else
        error("Unknown ilo BC type, must be :fixed, :periodic, or :zeroflux")
    end

    # ihi
    if bcs.ihi === :periodic
        nothing # ilo already applied the BC
    elseif bcs.ihi === :reflect
        reflect_ihi(u, nghost)
    elseif bcs.ihi === :zeroflux
        zeroflux_ihi(u, nghost)
    elseif bcs.ihi isa Tuple{Symbol,Number}
        fixed_ihi(u, bcs.ihi[2], nghost)
    else
        error("Unknown ihi BC type, must be :fixed, :periodic, or :zeroflux")
    end

    return nothing
end

"""Apply boundary conditions"""
function applybc!(
    u, bcs::NamedTuple{(:ilo, :ihi, :jlo, :jhi),Tuple{BC1,BC2,BC3,BC4}}, nghost::Int=1
) where {BC1,BC2,BC3,BC4}

    # ilo
    if bcs.ilo === :periodic
        periodic_iloihi(u, nghost)
    elseif bcs.ilo === :reflect
        reflect_ilo(u, nghost)
    elseif bcs.ilo === :zeroflux
        zeroflux_ilo(u, nghost)
    elseif bcs.ilo isa Tuple{Symbol,Number}
        if bcs.ilo[1] === :fixed
            fixed_ilo(u, bcs.ilo[2], nghost)
        else
            error("Unknown ilo BC type $(bcs.ilo), must be :fixed, :periodic, or :zeroflux")
        end
    end

    # ihi
    if bcs.ihi === :periodic
        nothing # ilo already applied the BC
    elseif bcs.ihi === :reflect
        reflect_ihi(u, nghost)
    elseif bcs.ihi === :zeroflux
        zeroflux_ihi(u, nghost)
    elseif bcs.ihi isa Tuple{Symbol,Number}
        if bcs.ihi[1] === :fixed
            fixed_ihi(u, bcs.ihi[2], nghost)
        else
            error("Unknown ihi BC type $(bcs.ihi), must be :fixed, :periodic, or :zeroflux")
        end
    end

    # jlo
    if bcs.jlo === :periodic
        periodic_jlojhi(u, nghost)
    elseif bcs.jlo === :reflect
        reflect_jlo(u, nghost)
    elseif bcs.jlo === :zeroflux
        zeroflux_jlo(u, nghost)
    elseif bcs.jlo isa Tuple
        if bcs.jlo[1] === :fixed
            fixed_jlo(u, bcs.jlo[2], nghost)
        else
            error("Unknown jlo BC type $(bcs.jlo), must be :fixed, :periodic, or :zeroflux")
        end
    end

    # jhi
    if bcs.jhi === :periodic
        nothing # jlo already applied the BC
    elseif bcs.jhi === :reflect
        reflect_jhi(u, nghost)
    elseif bcs.jhi === :zeroflux
        zeroflux_jhi(u, nghost)
    elseif bcs.jhi isa Tuple
        if bcs.jhi[1] === :fixed
            fixed_jhi(u, bcs.jhi[2], nghost)
        else
            error("Unknown jhi BC type $(bcs.jhi), must be :fixed, :periodic, or :zeroflux")
        end
    end

    return nothing
end

function hi_indices(A, nhalo)
    hmod = 1 .* (nhalo .== 0)
    hi_halo_end = last.(axes(A))
    hi_halo_start = hi_halo_end .- nhalo .+ 1 .- hmod
    hi_domn_end = hi_halo_start .- 1 .+ hmod
    hi_domn_start = hi_domn_end .- nhalo .+ 1 .- hmod

    return (hi_domn_start, hi_domn_end, hi_halo_start, hi_halo_end)
end

function lo_indices(A, nhalo)
    lmod = 1 .* (nhalo .== 0)
    lo_halo_start = first.(axes(A))
    lo_halo_end = lo_halo_start .+ nhalo .- 1 .+ lmod
    lo_domn_start = lo_halo_end .+ 1 .- lmod
    lo_domn_end = lo_domn_start .+ nhalo .- 1 .+ lmod

    return (lo_halo_start, lo_halo_end, lo_domn_start, lo_domn_end)
end

# -------------------------------------------------------
# Periodic
# -------------------------------------------------------
# FIXME: periodic is currently wrong...
"""Apply periodic boundary conditions on the `i` boundaries. The number of ghost cells, `nghost`, defaults to 1."""
function periodic_iloihi(u::AbstractVector{T}, nghost::Int=1) where {T}
    lo_halo_start, lo_halo_end, lo_domn_start, lo_domn_end = lo_indices(u, nghost)
    hi_domn_start, hi_domn_end, hi_halo_start, hi_halo_end = hi_indices(u, nghost)

    ilo_g = @view u[lo_halo_start[1]:lo_halo_end[1]] # ghost section
    ilo_d = @view u[lo_domn_start[1]:lo_domn_end[1]] # domain section

    ihi_d = @view u[hi_domn_start[1]:hi_domn_end[1]] # ghost section
    ihi_g = @view u[hi_halo_start[1]:hi_halo_end[1]] # domain section

    copy!(ilo_g, ihi_d)
    copy!(ihi_g, ilo_d)

    return nothing
end

"""Apply periodic boundary conditions on the `i` boundaries. The number of ghost cells, `nghost`, defaults to 1."""
function periodic_iloihi(u::AbstractMatrix{T}, nghost::Int=1) where {T}
    lo_halo_start, lo_halo_end, lo_domn_start, lo_domn_end = lo_indices(u, nghost)
    hi_domn_start, hi_domn_end, hi_halo_start, hi_halo_end = hi_indices(u, nghost)

    ilo_hs, jlo_hs = lo_halo_start
    ilo_he, jlo_he = lo_halo_end
    ilo_ds, jlo_ds = lo_domn_start
    ilo_de, jlo_de = lo_domn_end
    ihi_ds, jhi_ds = hi_domn_start
    ihi_de, jhi_de = hi_domn_end
    ihi_hs, jhi_hs = hi_halo_start
    ihi_he, jhi_he = hi_halo_end

    ihi_d = @view u[ihi_ds:ihi_de, jlo_ds:jhi_de]
    ilo_d = @view u[ilo_ds:ilo_de, jlo_ds:jhi_de]
    ilo_g = @view u[ilo_hs:ilo_he, jlo_ds:jhi_de]
    ihi_g = @view u[ihi_hs:ihi_he, jlo_ds:jhi_de]

    copy!(ilo_g, ihi_d)
    copy!(ihi_g, ilo_d)

    return nothing
end

"""Apply periodic boundary conditions on the `j` boundaries. The number of ghost cells, `nghost`, defaults to 1."""
function periodic_jlojhi(u::AbstractMatrix{T}, nghost::Int=1) where {T}
    lo_halo_start, lo_halo_end, lo_domn_start, lo_domn_end = lo_indices(u, nghost)
    hi_domn_start, hi_domn_end, hi_halo_start, hi_halo_end = hi_indices(u, nghost)

    ilo_hs, jlo_hs = lo_halo_start
    ilo_he, jlo_he = lo_halo_end
    ilo_ds, jlo_ds = lo_domn_start
    ilo_de, jlo_de = lo_domn_end
    ihi_ds, jhi_ds = hi_domn_start
    ihi_de, jhi_de = hi_domn_end
    ihi_hs, jhi_hs = hi_halo_start
    ihi_he, jhi_he = hi_halo_end

    jlo_g = @view u[ilo_ds:ihi_de, jlo_hs:jlo_he]
    jhi_g = @view u[ilo_ds:ihi_de, jhi_hs:jhi_he]
    jhi_d = @view u[ilo_ds:ihi_de, jhi_ds:jhi_de]
    jlo_d = @view u[ilo_ds:ihi_de, jlo_ds:jlo_de]

    copy!(jlo_g, jhi_d)
    copy!(jhi_g, jlo_d)

    return nothing
end

# -------------------------------------------------------
# Reflect
# -------------------------------------------------------
# The reflection boundary is exactly the same as the zero flux. Other boundary
# condition-related logic is used elsewhere (e.g. for a, b, Χ, and H) for 
# reflection vs flux, so this is used to help differentiate

"""Apply reflection at the `ilo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
reflect_ilo(u, nghost::Int=1) = zeroflux_ilo(u, nghost)

"""Apply reflection at the `ihi` boundary. The number of ghost cells, `nghost`, defaults to 1."""
reflect_ihi(u, nghost::Int=1) = zeroflux_ihi(u, nghost)

"""Apply reflection at the `jlo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
reflect_jlo(u, nghost::Int=1) = zeroflux_jlo(u, nghost)

"""Apply reflection at the `jhi` boundary. The number of ghost cells, `nghost`, defaults to 1."""
reflect_jhi(u, nghost::Int=1) = zeroflux_jhi(u, nghost)

# -------------------------------------------------------
# Zero Flux
# -------------------------------------------------------

"""Apply zero-flux at the `ilo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function zeroflux_ilo(u::AbstractVector{T}, nghost::Int=1) where {T}
    ilo = first(axes(u, 1))
    ihi = ilo + nghost - 1

    for i in ilo:ihi
        u[i] = u[ihi + 1]
    end

    return nothing
end

"""Apply zero-flux at the `ilo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function zeroflux_ilo(u::AbstractMatrix{T}, nghost::Int=1) where {T}
    ilo = first(axes(u, 1))
    ihi = ilo + nghost - 1

    jlohi = axes(u, 2)
    jlo = first(jlohi) + nghost
    jhi = last(jlohi) - nghost

    for j in jlo:jhi
        for i in ilo:ihi
            u[i, j] = u[ihi + 1, j]
        end
    end

    return nothing
end

"""Apply zero-flux at the `ihi` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function zeroflux_ihi(u::AbstractVector{T}, nghost::Int=1) where {T}
    ihi = last(axes(u, 1))
    ilo = ihi - nghost + 1

    for i in ilo:ihi
        u[i] = u[ilo - 1]
    end

    return nothing
end

"""Apply zero-flux at the `ihi` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function zeroflux_ihi(u::AbstractMatrix{T}, nghost::Int=1) where {T}
    ihi = last(axes(u, 1))
    ilo = ihi - nghost + 1

    jlohi = axes(u, 2)
    jlo = first(jlohi) + nghost
    jhi = last(jlohi) - nghost

    for j in jlo:jhi
        for i in ilo:ihi
            u[i, j] = u[ilo - 1, j]
        end
    end

    return nothing
end

"""Apply zero-flux at the `jlo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function zeroflux_jlo(u::AbstractMatrix{T}, nghost::Int=1) where {T}
    jlo = first(axes(u, 2))
    jhi = jlo + nghost - 1

    ilohi = axes(u, 1)
    ilo = first(ilohi) + nghost
    ihi = last(ilohi) - nghost

    for j in jlo:jhi
        for i in ilo:ihi
            u[i, j] = u[i, jhi + 1]
        end
    end

    return nothing
end

"""Apply zero-flux at the `jhi` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function zeroflux_jhi(u::AbstractMatrix{T}, nghost::Int=1) where {T}
    jhi = last(axes(u, 2))
    jlo = jhi - nghost + 1

    ilohi = axes(u, 1)
    ilo = first(ilohi) + nghost
    ihi = last(ilohi) - nghost

    for j in jlo:jhi
        for i in ilo:ihi
            u[i, j] = u[i, jlo - 1]
        end
    end

    return nothing
end

# -------------------------------------------------------
# Fixed
# -------------------------------------------------------

"""Apply a fixed value `ufix` to the `ilo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function fixed_ilo(u::AbstractVector{T}, ufix, nghost::Int=1) where {T}
    ilo = first(axes(u, 1))
    ihi = ilo + nghost - 1

    for i in ilo:ihi
        u[i] = ufix
    end

    return nothing
end

"""Apply a fixed value `ufix` to the `ilo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function fixed_ihi(u::AbstractVector{T}, ufix, nghost::Int=1) where {T}
    ihi = last(axes(u, 1))
    ilo = ihi - nghost + 1

    for i in ilo:ihi
        u[i] = ufix
    end

    return nothing
end

"""Apply a fixed value `ufix` to the `ilo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function fixed_ilo(u::AbstractMatrix{T}, ufix, nghost::Int=1) where {T}
    ilohi = axes(u, 1)
    jlohi = axes(u, 2)

    jlo = first(jlohi) + nghost
    jhi = last(jlohi) - nghost
    ilo = first(ilohi)
    ihi = ilo + nghost - 1

    for j in jlohi
        for i in ilo:ihi
            u[i, j] = ufix
        end
    end

    return nothing
end

"""Apply a fixed value `ufix` to the `ihi` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function fixed_ihi(u::AbstractMatrix{T}, ufix, nghost::Int=1) where {T}
    ihi = last(axes(u, 1))
    ilo = ihi - nghost + 1

    jlohi = axes(u, 2)
    jlo = first(jlohi) + nghost
    jhi = last(jlohi) - nghost

    for j in jlohi
        for i in ilo:ihi
            u[i, j] = ufix
        end
    end

    return nothing
end

"""Apply a fixed value `ufix` to the `jlo` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function fixed_jlo(u::AbstractMatrix{T}, ufix, nghost::Int=1) where {T}
    jlo = first(axes(u, 2))
    jhi = jlo + nghost - 1

    ilohi = axes(u, 1)
    ilo = first(ilohi) + nghost
    ihi = last(ilohi) - nghost

    for j in jlo:jhi
        for i in ilohi
            u[i, j] = ufix
        end
    end

    return nothing
end

"""Apply a fixed value `ufix` to the `jhi` boundary. The number of ghost cells, `nghost`, defaults to 1."""
function fixed_jhi(u::AbstractMatrix{T}, ufix, nghost::Int=1) where {T}
    jhi = last(axes(u, 2))
    jlo = jhi - nghost + 1

    ilohi = axes(u, 1)
    ilo = first(ilohi) + nghost
    ihi = last(ilohi) - nghost

    for j in jlo:jhi
        for i in ilohi
            u[i, j] = ufix
        end
    end

    return nothing
end
