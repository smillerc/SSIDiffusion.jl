using StaticArrays, LinearAlgebra

struct Mesh1D{A,B,C}
    coords::A
    centroid::B
    volume::C
    nghost::Int
end

struct Mesh2D{A,B,C,D,E}
    coords::A
    centroid::B
    volume::C
    facenorms::D
    facearea::E
    nghost::Int
    A⁻Δᵢⱼ::A
    A⁺Δᵢⱼ::A
    C⁻Δᵢⱼ::A
    C⁺Δᵢⱼ::A
    alpha::Int
    rcoord::Int
end

struct Mesh3D{A,B,C,D,E}
    coords::A
    centroid::B
    volume::C
    facenorms::D
    facearea::E
    nghost::Int
end

"""
    Mesh2D(x::AbstractVector{T}, y::AbstractVector{T}, nhalo::Int=1; add_halo=true, axis_of_symmetry=:x, meshtype=:cartesian) where {T}

The Mesh2D constructor when provided the `x` and `y` 1D coordinate vectors.

# Arguments
 - `x::AbstractVector{T}`: x-coordinate vector (1D)
 - `y::AbstractVector{T}`: y-coordinate vector (1D)
 - `nhalo::Int=1`: Number of halo cells in each dimension

# Keyword Arguments
 - `add_halo=true`: Pad the coordinates with a layer of halo cells
 - `meshtype=:cartesian`: Either `:cartesian` or `:cylindrical` symmetry
 - `axis_of_symmetry=:x`: If `:cylindrical`, which axis is the axis of rotation?
"""
function Mesh2D(
    x::AbstractVector{T},
    y::AbstractVector{T},
    nhalo::Int=1;
    add_halo=true,
    axis_of_symmetry=:x,
    meshtype=:cartesian,
) where {T}
    if add_halo
        x = addhalo(x, nhalo)
        y = addhalo(y, nhalo)
    end

    M = length(x) - 1
    N = length(y) - 1

    xy = zeros(2, M + 1, N + 1)
    for j in 1:(N + 1)
        for i in 1:(M + 1)
            xy[1, i, j] = x[i]
            xy[2, i, j] = y[j]
        end
    end

    mesh = Mesh2D(
        xy, nhalo; axis_of_symmetry=axis_of_symmetry, meshtype=meshtype, add_halo=false
    )

    return mesh
end

"""
    Mesh2D(xy::AbstractArray{T,3}, nhalo::Int=1; axis_of_symmetry=:x, meshtype=:cartesian) where {T}

The Mesh2D constructor when the (x,y) 2D coordinates have already been created. Here `xy` is indexed as `[1:2,i,j]`. This assumes that
the halo region has already been defined in the `xy` coordinate array.

# Arguments
 - `xy::AbstractArray{T,3}`: xy-coordinate array `[xy,i,j]`
 - `nhalo::Int=1`: Number of halo cells in each dimension

# Keyword Arguments
 - `add_halo=false`: This helps the user if set to true -- raises an error to signify that the `xy` array needs to have the halo region included
 - `meshtype=:cartesian`: Either `:cartesian` or `:cylindrical` symmetry
 - `axis_of_symmetry=:x`: If `:cylindrical`, which axis is the axis of rotation?
"""
function Mesh2D(
    xy::AbstractArray{T,3},
    nhalo::Int=1;
    add_halo=false,
    axis_of_symmetry=:x,
    meshtype=:cartesian,
) where {T}
    if add_halo
        error(
            "This constructor method requires the `xy` coordinate array to contain the halo region",
        )
    end

    # cell dimensions
    _, M, N = size(xy) .- 1

    # centroid = round.(quad_centroids(xy), sigdigits=15)
    # volume = round.(quad_volumes(xy), sigdigits=15)
    centroid = quad_centroids(xy)
    volume = quad_volumes(xy)

    x_c = [
        @SVector [centroid[1, i, j], centroid[2, i, j]] for i in axes(centroid, 2),
        j in axes(centroid, 3)
    ]

    @assert meshtype === :cartesian || meshtype === :cylindrical "Unknown meshtype $meshtype, must be `:cartesian` or `:cylindrical`"
    if meshtype === :cartesian
        alpha = 0
        rcoord = 1
    elseif meshtype === :cylindrical
        alpha = 1
        @assert axis_of_symmetry === :x || axis_of_symmetry === :y
        if axis_of_symmetry === :x
            rcoord = 2
        else # axis_of_symmetry === :x
            rcoord = 1
        end
        r = @views centroid[rcoord, :, :]

        volume = @. volume * abs(r)
    end

    facelen, facenorms = quad_face_areas_and_vecs(xy)

    mesh = Mesh2D(
        xy,
        x_c,
        volume,
        facenorms,
        facelen,
        nhalo,
        zeros(2, M, N),
        zeros(2, M, N),
        zeros(2, M, N),
        zeros(2, M, N),
        alpha,
        rcoord,
    )
    update_face_geometric_coeff!(mesh)

    return mesh
end

function Mesh1D(x::AbstractVector{T}, nhalo::Int=1) where {T}
    M = length(x) - 1 #+ 2nhalo

    centroids = zeros(M)
    volumes = zeros(M)

    ilo = first(axes(x, 1))
    ihi = last(axes(x, 1)) - 1

    for i in ilo:ihi
        centroids[i] = 0.5(x[i] + x[i + 1])
        volumes[i] = abs(x[i + 1] - x[i])
    end

    return Mesh1D(x, centroids, volumes, nhalo)
end

function addhalo(x::Vector{T}, nhalo::Integer; dx=nothing) where {T<:AbstractFloat}
    if dx === nothing
        dx_hi = x[end] - x[end - 1]
        dx_lo = x[2] - x[1]
    else
        dx_hi = dx
        dx_lo = dx
    end

    lo_halo = Vector{T}(undef, nhalo)
    hi_halo = Vector{T}(undef, nhalo)

    for i in 1:nhalo
        hi_halo[i] = x[end] + (dx_hi * i)
        lo_halo[i] = x[1] - (dx_lo * i)
    end
    reverse!(lo_halo)
    return vcat(lo_halo, x, hi_halo)
end

function quad_volumes(xy::AbstractArray{T,3}) where {T}
    ni = size(xy, 2) - 1
    nj = size(xy, 3) - 1
    volumes = zeros(T, (ni, nj))

    ϵ = eps(T)

    for j in 1:nj
        for i in 1:ni
            x1, y1 = @views xy[:, i, j]
            x2, y2 = @views xy[:, i + 1, j]
            x3, y3 = @views xy[:, i + 1, j + 1]
            x4, y4 = @views xy[:, i, j + 1]

            dx13 = x1 - x3
            dy24 = y2 - y4
            dx42 = x4 - x2
            dy13 = y1 - y3

            dx13 = (dx13) * (abs(dx13) >= ϵ)
            dy24 = (dy24) * (abs(dy24) >= ϵ)
            dx42 = (dx42) * (abs(dx42) >= ϵ)
            dy13 = (dy13) * (abs(dy13) >= ϵ)

            volumes[i, j] = 0.5 * (dx13 * dy24 + dx42 * dy13)
        end
    end

    return volumes
end

function quad_centroids(xy::AbstractArray{T,3}) where {T}
    ni = size(xy, 2) - 1
    nj = size(xy, 3) - 1
    centroids = similar(xy, (2, ni, nj))

    for j in 1:nj
        for i in 1:ni
            centroids[1, i, j] =
                0.25(xy[1, i, j] + xy[1, i + 1, j] + xy[1, i + 1, j + 1] + xy[1, i, j + 1])
            centroids[2, i, j] =
                0.25(xy[2, i, j] + xy[2, i + 1, j] + xy[2, i + 1, j + 1] + xy[2, i, j + 1])
        end
    end
    return centroids
end

function quad_face_areas_and_vecs(xy::AbstractArray{T,3}) where {T}
    ni = size(xy, 2) - 1
    nj = size(xy, 3) - 1
    facelens = zeros(T, (4, ni, nj))
    normvecs = zeros(T, (2, 4, ni, nj))

    ϵ = eps(T)

    for j in 1:nj
        for i in 1:ni
            x1, y1 = @views xy[:, i, j]
            x2, y2 = @views xy[:, i + 1, j]
            x3, y3 = @views xy[:, i + 1, j + 1]
            x4, y4 = @views xy[:, i, j + 1]

            Δx21 = x2 - x1
            Δx32 = x3 - x2
            Δx43 = x4 - x3
            Δx14 = x1 - x4
            Δy32 = y3 - y2
            Δy43 = y4 - y3
            Δy14 = y1 - y4
            Δy21 = y2 - y1

            Δy21 = Δy21 * (abs(Δy21) >= ϵ)
            Δy32 = Δy32 * (abs(Δy32) >= ϵ)
            Δy43 = Δy43 * (abs(Δy43) >= ϵ)
            Δy14 = Δy14 * (abs(Δy14) >= ϵ)
            Δx21 = Δx21 * (abs(Δx21) >= ϵ)
            Δx32 = Δx32 * (abs(Δx32) >= ϵ)
            Δx43 = Δx43 * (abs(Δx43) >= ϵ)
            Δx14 = Δx14 * (abs(Δx14) >= ϵ)

            Δx12 = -Δx21
            Δx23 = -Δx32
            Δx34 = -Δx43
            Δx41 = -Δx14

            Δs1 = sqrt((Δx21)^2 + (Δy21)^2)
            Δs2 = sqrt((Δx32)^2 + (Δy32)^2)
            Δs3 = sqrt((Δx43)^2 + (Δy43)^2)
            Δs4 = sqrt((Δx14)^2 + (Δy14)^2)

            facelens[1, i, j] = Δs1
            facelens[2, i, j] = Δs2
            facelens[3, i, j] = Δs3
            facelens[4, i, j] = Δs4

            normvecs[1, 1, i, j] = Δy21 / Δs1
            normvecs[2, 1, i, j] = Δx12 / Δs1

            normvecs[1, 2, i, j] = Δy32 / Δs2
            normvecs[2, 2, i, j] = Δx23 / Δs2

            normvecs[1, 3, i, j] = Δy43 / Δs3
            normvecs[2, 3, i, j] = Δx34 / Δs3

            normvecs[1, 4, i, j] = Δy14 / Δs4
            normvecs[2, 4, i, j] = Δx41 / Δs4
        end
    end

    return facelens, normvecs
end

"""

Calculate the geometry coefficients used for the face-adjacent bulk heat 
capacities. For a static mesh, this is only called once.
"""
function update_face_geometric_coeff!(mesh::Mesh2D)
    coords = mesh.coords
    centroids = mesh.centroid
    A⁻Δᵢⱼ = mesh.A⁻Δᵢⱼ
    A⁺Δᵢⱼ = mesh.A⁺Δᵢⱼ
    C⁺Δ = mesh.C⁺Δᵢⱼ
    C⁻Δ = mesh.C⁻Δᵢⱼ
    raxis = mesh.rcoord
    alpha = mesh.alpha

    ilohi = axes(mesh.volume, 1) # cell i-index ranges
    jlohi = axes(mesh.volume, 2) # cell j-index ranges
    ilo = first(ilohi) + mesh.nghost
    jlo = first(jlohi) + mesh.nghost
    ihi = last(ilohi) - mesh.nghost
    jhi = last(jlohi) - mesh.nghost

    iterator_range = CartesianIndices((ilo:(ihi + 1), jlo:(jhi + 1)))

    for idx in iterator_range
        i, j = Tuple(idx)

        x⃗ᵢⱼ = SVector{2,Float64}(view(coords, :, i, j))
        x⃗ᵢ₊₁ⱼ = SVector{2,Float64}(view(coords, :, i + 1, j))
        x⃗ᵢⱼ₊₁ = SVector{2,Float64}(view(coords, :, i, j + 1))

        x⃗cᵢⱼ = centroids[i, j]
        x⃗cᵢⱼ₋₁ = centroids[i, j - 1]
        x⃗cᵢ₋₁ⱼ = centroids[i - 1, j]

        u1 = SVector{2,Float64}(x⃗ᵢⱼ - x⃗cᵢⱼ)
        u2 = SVector{2,Float64}(x⃗ᵢ₊₁ⱼ - x⃗cᵢⱼ)
        A⁺Δᵢⱼ₁ = 0.5abs(u1 × u2)

        u3 = SVector{2,Float64}(x⃗ᵢ₊₁ⱼ - x⃗cᵢⱼ₋₁)
        u4 = SVector{2,Float64}(x⃗ᵢⱼ - x⃗cᵢⱼ₋₁)
        A⁻Δᵢⱼ₁ = 0.5abs(u3 × u4)

        u5 = SVector{2,Float64}(x⃗ᵢⱼ₊₁ - x⃗cᵢⱼ)
        u6 = SVector{2,Float64}(x⃗ᵢⱼ - x⃗cᵢⱼ)
        A⁺Δᵢⱼ₂ = 0.5abs(u5 × u6)

        u7 = SVector{2,Float64}(x⃗ᵢⱼ - x⃗cᵢ₋₁ⱼ)
        u8 = SVector{2,Float64}(x⃗ᵢⱼ₊₁ - x⃗cᵢ₋₁ⱼ)
        A⁻Δᵢⱼ₂ = 0.5abs(u7 × u8)

        Rᵢⱼ = x⃗ᵢⱼ[raxis]^alpha
        Rᵢ₊₁ⱼ = x⃗ᵢ₊₁ⱼ[raxis]^alpha
        Rᵢⱼ₊₁ = x⃗ᵢⱼ₊₁[raxis]^alpha
        Rcᵢⱼ = x⃗cᵢⱼ[raxis]^alpha
        Rcᵢⱼ₋₁ = x⃗cᵢⱼ₋₁[raxis]^alpha
        Rcᵢ₋₁ⱼ = x⃗cᵢ₋₁ⱼ[raxis]^alpha

        A⁻Δᵢⱼ[1, i, j] = A⁻Δᵢⱼ₁
        A⁻Δᵢⱼ[2, i, j] = A⁻Δᵢⱼ₂

        A⁺Δᵢⱼ[1, i, j] = A⁺Δᵢⱼ₁
        A⁺Δᵢⱼ[2, i, j] = A⁺Δᵢⱼ₂

        C⁺Δ[1, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢ₊₁ⱼ + Rcᵢⱼ) * A⁺Δᵢⱼ₁ # C⁺Δᵢⱼ₁
        C⁺Δ[2, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢⱼ₊₁ + Rcᵢⱼ) * A⁺Δᵢⱼ₂ # C⁺Δᵢⱼ₂
        C⁻Δ[1, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢ₊₁ⱼ + Rcᵢⱼ₋₁) * A⁻Δᵢⱼ₁ # C⁻Δᵢⱼ₁
        C⁻Δ[2, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢⱼ₊₁ + Rcᵢ₋₁ⱼ) * A⁻Δᵢⱼ₂ # C⁻Δᵢⱼ₂
    end

    return nothing
end
