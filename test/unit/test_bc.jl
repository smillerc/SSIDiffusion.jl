
@testitem "Test 1D Boundary Conditions" begin
    include("common.jl")

    u = rand(10)
    nghost = 1

    bcs = (ilo=:periodic, ihi=:periodic)

    lo_halo_start, lo_halo_end, lo_domn_start, lo_domn_end = ThermoDiffusionMethods.lo_indices(
        u, nghost
    )
    hi_domn_start, hi_domn_end, hi_halo_start, hi_halo_end = ThermoDiffusionMethods.hi_indices(
        u, nghost
    )

    ilo_hs, = lo_halo_start
    ilo_he, = lo_halo_end
    ilo_ds, = lo_domn_start
    ilo_de, = lo_domn_end
    ihi_ds, = hi_domn_start
    ihi_de, = hi_domn_end
    ihi_hs, = hi_halo_start
    ihi_he, = hi_halo_end

    ilo_g = @view u[lo_halo_start[1]:lo_halo_end[1]] # ghost section
    ilo_d = @view u[lo_domn_start[1]:lo_domn_end[1]] # domain section

    ihi_d = @view u[hi_domn_start[1]:hi_domn_end[1]] # ghost section
    ihi_g = @view u[hi_halo_start[1]:hi_halo_end[1]] # domain section

    ThermoDiffusionMethods.applybc!(u, bcs)

    @test all(ilo_d .== ihi_g)
    @test all(ihi_g .== ilo_d)

    bcs_2 = (ilo=(:fixed, 4.5), ihi=:zeroflux)
    ThermoDiffusionMethods.applybc!(u, bcs_2)
    @test all(u[1] == 4.5)
    @test all(ihi_g .== ihi_d)
end

@testitem "Test 2D Boundary Conditions" begin
    include("common.jl")

    u = rand(10, 10)
    nghost = 1

    bcs = (ilo=:periodic, ihi=:periodic, jlo=:zeroflux, jhi=(:fixed, 123.0))

    lo_halo_start, lo_halo_end, lo_domn_start, lo_domn_end = ThermoDiffusionMethods.lo_indices(
        u, nghost
    )
    hi_domn_start, hi_domn_end, hi_halo_start, hi_halo_end = ThermoDiffusionMethods.hi_indices(
        u, nghost
    )

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

    jlo = @view u[2:9, 1]
    jhi = @view u[2:9, 10]

    ThermoDiffusionMethods.applybc!(u, bcs)

    @test all(ilo_d .== ihi_g)
    @test all(ihi_g .== ilo_d)
    @test all(jhi .== 123.0)
    @test all(jlo .== @view u[2:9, 2])
end
