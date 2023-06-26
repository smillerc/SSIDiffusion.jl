
@testitem "Test Convergence Check" begin
    include("common.jl")

    uⁿ⁺¹ = rand(50, 50)
    uⁿ = rand(50, 50)

    bm = @benchmark SSIDiffusion.stage_convergence($uⁿ⁺¹, $uⁿ, 1)
    @test bm.allocs == 0
end
