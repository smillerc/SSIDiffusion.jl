@testitem "Test Vertex Temperatures & Interpolation Coeff." begin
    dx = 0.1
    dy = dx

    x = collect(0:dx:1)
    y = collect(0:dy:1)

    nhalo = 1
    mesh = Mesh2D(x, y, nhalo; meshtype=:cartesian)

    T_init(T0, x, y) = T0 + T0 * x

    xc = [c[1] for c in mesh.centroid]
    yc = [c[2] for c in mesh.centroid]

    T = zeros(size(mesh.volume))
    κ = similar(T)
    T0 = 1
    T = @. abs.(T_init(T0, xc, yc))

    Ts = 1e-3
    ϵ0 = 0.2
    ϵ1 = 0.02
    solver = SSISolver2D(mesh, Ts, ϵ0, ϵ1)

    η = solver.η
    ξ = solver.ξ

    fill!(κ, 1.0)

    SSIDiffusion.update_bilinear_coeff!(mesh, ξ, η)
    @test all(iszero.(ξ))
    @test all(iszero.(η))

    SSIDiffusion.interp_coeff!(solver.μ, κ, solver.ξ, solver.η, mesh.nghost)
    μ = @view solver.μ[:, (begin + nhalo):(end - nhalo), (begin + nhalo):(end - nhalo)]
    @test all(μ .≈ 0.25)

    SSIDiffusion.vertex_temperatures!(solver.T_vertex, T, solver.μ, mesh.nghost)
    Tv_x = @view solver.T_vertex[(begin + nhalo):(end - nhalo), 2]
    Tv_y = @view solver.T_vertex[3, (begin + nhalo):(end - nhalo)]
    @test all(Tv_x .≈ 1.0:0.1:2.0)
    @test all(Tv_y .≈ 1.1)
end

@testitem "Test Face Conductivity" begin
    include("common.jl")

    AΔ⁺ = 1.0

    AΔ⁻ = 1.0
    κ = 2.0
    κneighbor = 1.0
    κface = SSIDiffusion.face_conductivity(AΔ⁺, AΔ⁻, κ, κneighbor)
    @test κface == 1.5

    AΔ⁺ = 1.0
    AΔ⁻ = 2.0
    κ = 1.0
    κneighbor = 1.0
    κface = SSIDiffusion.face_conductivity(AΔ⁺, AΔ⁻, κ, κneighbor)
    @test κface == 1.0
end

@testitem "Test Face Weighting" begin
    dx = 0.1
    dy = 2dx

    x = collect(0:dx:1)
    y = collect(0:dy:1)

    nhalo = 1
    mesh = Mesh2D(x, y, nhalo; meshtype=:cartesian)

    cv = ones(size(mesh.volume))
    ρ = ones(size(mesh.volume))

    Χ = zeros(2, size(ρ)...)
    SSIDiffusion.faceweights!(Χ, mesh, cv, ρ)

    Χ1 = @view Χ[1, :, :]
    Χ2 = @view Χ[2, :, :]

    @test all(Χ[1, (begin + 1):end, (begin + 1):end] .≈ 0.5)
    @test all(Χ[2, (begin + 1):end, (begin + 1):end] .≈ 0.5)
end

@testitem "Test Face Fluxes" begin end
