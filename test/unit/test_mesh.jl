@testitem "Test Mesh2D Cartesian" begin
    include("common.jl")

    dx = 0.1
    dy = 2dx

    x = collect(0:dx:1)
    y = collect(0:dy:1)

    nhalo = 1
    mesh = Mesh2D(x, y, nhalo; meshtype=:cartesian)
    @test mesh.alpha == 0
    @test mesh.rcoord == 1

    @test all(mesh.volume .≈ dx * dy)

    cell1_norms = mesh.facenorms[:, :, 1, 1]

    @test cell1_norms[:, 1] == [0.0, -1.0]
    @test cell1_norms[:, 2] == [1.0, 0.0]
    @test cell1_norms[:, 3] == [0.0, 1.0]
    @test cell1_norms[:, 4] == [-1.0, 0.0]

    @test all(mesh.facearea[1, :, :] .≈ dx)
    @test all(mesh.facearea[2, :, :] .≈ dy)
    @test all(mesh.facearea[3, :, :] .≈ dx)
    @test all(mesh.facearea[4, :, :] .≈ dy)
    @test mesh.nghost == nhalo

    A⁻Δᵢⱼ₁ = @view mesh.A⁻Δᵢⱼ[1, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁻Δᵢⱼ₂ = @view mesh.A⁻Δᵢⱼ[2, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁺Δᵢⱼ₁ = @view mesh.A⁺Δᵢⱼ[1, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁺Δᵢⱼ₂ = @view mesh.A⁺Δᵢⱼ[2, (begin + 1):(end - 1), (begin + 1):(end - 1)]

    bx = dx
    by = dy
    hx = 0.5dx
    hy = 0.5dy

    @test all(A⁻Δᵢⱼ₁ .≈ 0.5bx * hy)
    @test all(A⁺Δᵢⱼ₁ .≈ 0.5bx * hy)
    @test all(A⁻Δᵢⱼ₂ .≈ 0.5by * hx)
    @test all(A⁺Δᵢⱼ₂ .≈ 0.5by * hx)

    # For the cartesian case all R's are 1
    C⁺Δᵢⱼ₁ = @view mesh.C⁺Δᵢⱼ[1, (begin + 1):(end - 1), (begin + 1):(end - 1)] # = (1 / 3) * (Rᵢⱼ + Rᵢ₊₁ⱼ + Rcᵢⱼ) * A⁺Δᵢⱼ₁ # C⁺Δᵢⱼ₁
    C⁺Δᵢⱼ₂ = @view mesh.C⁺Δᵢⱼ[2, (begin + 1):(end - 1), (begin + 1):(end - 1)] # = (1 / 3) * (Rᵢⱼ + Rᵢⱼ₊₁ + Rcᵢⱼ) * A⁺Δᵢⱼ₂ # C⁺Δᵢⱼ₂
    C⁻Δᵢⱼ₁ = @view mesh.C⁻Δᵢⱼ[1, (begin + 1):(end - 1), (begin + 1):(end - 1)] # = (1 / 3) * (Rᵢⱼ + Rᵢ₊₁ⱼ + Rcᵢⱼ₋₁) * A⁻Δᵢⱼ₁ # C⁻Δᵢⱼ₁
    C⁻Δᵢⱼ₂ = @view mesh.C⁻Δᵢⱼ[2, (begin + 1):(end - 1), (begin + 1):(end - 1)] # = (1 / 3) * (Rᵢⱼ + Rᵢⱼ₊₁ + Rcᵢ₋₁ⱼ) * A⁻Δᵢⱼ₂ # C⁻Δᵢⱼ₂

    @test all(C⁺Δᵢⱼ₁ .≈ 0.5bx * hy)
    @test all(C⁺Δᵢⱼ₂ .≈ 0.5bx * hy)
    @test all(C⁻Δᵢⱼ₁ .≈ 0.5by * hx)
    @test all(C⁻Δᵢⱼ₂ .≈ 0.5by * hx)
end

@testitem "Test Mesh2D Cylindrical (x-axis-of-symmetry)" begin
    include("common.jl")

    dx = 0.1
    dy = 2dx

    x = collect(0:dx:1)
    y = collect(0:dy:1)

    nhalo = 1
    mesh = Mesh2D(x, y, nhalo; axis_of_symmetry=:x, meshtype=:cylindrical)
    @test mesh.rcoord == 2
    @test mesh.alpha == 1

    nhalo = 1

    A⁻Δᵢⱼ₁ = @view mesh.A⁻Δᵢⱼ[1, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁻Δᵢⱼ₂ = @view mesh.A⁻Δᵢⱼ[2, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁺Δᵢⱼ₁ = @view mesh.A⁺Δᵢⱼ[1, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁺Δᵢⱼ₂ = @view mesh.A⁺Δᵢⱼ[2, (begin + 1):(end - 1), (begin + 1):(end - 1)]

    bx = dx
    by = dy
    hx = 0.5dx
    hy = 0.5dy

    @test all(A⁺Δᵢⱼ₁ .≈ 0.5bx * hy)
    @test all(A⁺Δᵢⱼ₂ .≈ 0.5by * hx)
    @test all(A⁻Δᵢⱼ₁ .≈ 0.5bx * hy)
    @test all(A⁻Δᵢⱼ₂ .≈ 0.5by * hx)

    C⁺Δᵢⱼ = zeros(size(mesh.C⁺Δᵢⱼ))
    C⁻Δᵢⱼ = zeros(size(mesh.C⁻Δᵢⱼ))

    ilohi = axes(mesh.volume, 1) # cell i-index ranges
    jlohi = axes(mesh.volume, 2) # cell j-index ranges
    ilo = first(ilohi) + mesh.nghost
    jlo = first(jlohi) + mesh.nghost
    ihi = last(ilohi) - mesh.nghost
    jhi = last(jlohi) - mesh.nghost

    for j in jlo:(jhi + 1)
        for i in ilo:(ihi + 1)
            Rᵢⱼ = mesh.coords[2, i, j]
            Rᵢ₊₁ⱼ = mesh.coords[2, i + 1, j]
            Rᵢⱼ₊₁ = mesh.coords[2, i, j + 1]
            Rcᵢⱼ = mesh.centroid[i, j][2]
            Rcᵢⱼ₋₁ = mesh.centroid[i, j - 1][2]
            Rcᵢ₋₁ⱼ = mesh.centroid[i - 1, j][2]

            C⁺Δᵢⱼ[1, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢ₊₁ⱼ + Rcᵢⱼ) * 0.5bx * hy
            C⁺Δᵢⱼ[2, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢⱼ₊₁ + Rcᵢⱼ) * 0.5by * hx
            C⁻Δᵢⱼ[1, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢ₊₁ⱼ + Rcᵢⱼ₋₁) * 0.5bx * hy
            C⁻Δᵢⱼ[2, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢⱼ₊₁ + Rcᵢ₋₁ⱼ) * 0.5by * hx
        end
    end

    @test all(C⁺Δᵢⱼ .≈ mesh.C⁺Δᵢⱼ)

    xc = [c[1] for c in mesh.centroid]
    yc = [c[2] for c in mesh.centroid]

    r_y = collect((-0.5dy):dy:(0.5dy + 1))

    vol = zeros(size(mesh.volume))

    for j in axes(vol, 2)
        for i in axes(vol, 1)
            vol[i, j] = abs(r_y[j]) * dx * dy
        end
    end

    @test all(vol .≈ mesh.volume)
end
@testitem "Test Mesh2D Cylindrical (y-axis-of-symmetry)" begin
    include("common.jl")

    dx = 0.1
    dy = 2dx

    x = collect(0:dx:1)
    y = collect(0:dy:1)

    nhalo = 1
    mesh = Mesh2D(x, y, nhalo; axis_of_symmetry=:y, meshtype=:cylindrical)
    @test mesh.rcoord == 1
    @test mesh.alpha == 1

    A⁻Δᵢⱼ₁ = @view mesh.A⁻Δᵢⱼ[1, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁻Δᵢⱼ₂ = @view mesh.A⁻Δᵢⱼ[2, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁺Δᵢⱼ₁ = @view mesh.A⁺Δᵢⱼ[1, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    A⁺Δᵢⱼ₂ = @view mesh.A⁺Δᵢⱼ[2, (begin + 1):(end - 1), (begin + 1):(end - 1)]

    bx = dx
    by = dy
    hx = 0.5dx
    hy = 0.5dy

    @test all(A⁻Δᵢⱼ₁ .≈ 0.5bx * hy)
    @test all(A⁺Δᵢⱼ₁ .≈ 0.5bx * hy)
    @test all(A⁻Δᵢⱼ₂ .≈ 0.5by * hx)
    @test all(A⁺Δᵢⱼ₂ .≈ 0.5by * hx)

    C⁺Δᵢⱼ = zeros(size(mesh.C⁺Δᵢⱼ))
    C⁻Δᵢⱼ = zeros(size(mesh.C⁻Δᵢⱼ))

    ilohi = axes(mesh.volume, 1) # cell i-index ranges
    jlohi = axes(mesh.volume, 2) # cell j-index ranges
    ilo = first(ilohi) + mesh.nghost
    jlo = first(jlohi) + mesh.nghost
    ihi = last(ilohi) - mesh.nghost
    jhi = last(jlohi) - mesh.nghost

    for j in jlo:(jhi + 1)
        for i in ilo:(ihi + 1)
            Rᵢⱼ = mesh.coords[1, i, j]
            Rᵢ₊₁ⱼ = mesh.coords[1, i + 1, j]
            Rᵢⱼ₊₁ = mesh.coords[1, i, j + 1]
            Rcᵢⱼ = mesh.centroid[i, j][1]
            Rcᵢⱼ₋₁ = mesh.centroid[i, j - 1][1]
            Rcᵢ₋₁ⱼ = mesh.centroid[i - 1, j][1]

            C⁺Δᵢⱼ[1, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢ₊₁ⱼ + Rcᵢⱼ) * 0.5bx * hy
            C⁺Δᵢⱼ[2, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢⱼ₊₁ + Rcᵢⱼ) * 0.5by * hx
            C⁻Δᵢⱼ[1, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢ₊₁ⱼ + Rcᵢⱼ₋₁) * 0.5bx * hy
            C⁻Δᵢⱼ[2, i, j] = (1 / 3) * (Rᵢⱼ + Rᵢⱼ₊₁ + Rcᵢ₋₁ⱼ) * 0.5by * hx
        end
    end

    @test all(C⁺Δᵢⱼ .≈ mesh.C⁺Δᵢⱼ)

    xc = [c[1] for c in mesh.centroid]
    yc = [c[2] for c in mesh.centroid]

    r_x = collect((-0.5dx):dx:(0.5dx + 1))
    r_y = collect((-0.5dy):dy:(0.5dy + 1))

    vol = zeros(size(mesh.volume))

    for j in axes(vol, 2)
        for i in axes(vol, 1)
            vol[i, j] = abs(r_x[i]) * dx * dy
        end
    end

    @test all(vol .≈ mesh.volume)
end
