
using SSIDiffusion
using Printf
using CairoMakie
using Polyester
# using WriteVTK
using Test
using TimerOutputs

reset_timer!()
# function writecontour(xy_coords::AbstractArray{T,3}, temp_data, iteration, time, name) where {T}
#     x = @views xy_coords[1, :, :]
#     y = @views xy_coords[2, :, :]

#     fn = name * @sprintf("%07i", iteration)
#     vtk_grid(fn, x, y) do vtk
#         vtk["TimeValue"] = time
#         @show time
#         vtk["temperature", VTKCellData()] = temp_data
#     end
# end

"""Update cell conductivity"""
function update_κ_cell!(κ, κ0, T, n=3)
    @batch for i in eachindex(T)
        κ[i] = κ0 * T[i]^n
    end
end

function wavy_grid(dx, nghost)
    x1d = (-(nghost * dx)):dx:(1 + nghost * dx)
    y1d = (-(nghost * dx)):dx:(1 + nghost * dx)

    a0 = 0.1
    xy_wavy = zeros(2, length(x1d), length(y1d))
    for j in axes(xy_wavy, 3)
        for i in axes(xy_wavy, 2)
            x = x1d[i]
            y = y1d[j]

            if 0 < x < 1 && 0 < y < 1
                xy_wavy[1, i, j] = x + a0 * sin(2pi * x) * sin(2pi * y)
                xy_wavy[2, i, j] = y + a0 * sin(2pi * x) * sin(2pi * y)
            else
                xy_wavy[1, i, j] = x
                xy_wavy[2, i, j] = y
            end
        end
    end

    return SSIDiffusion.Mesh2D(xy_wavy, nghost)
end

"""Make a uniform grid"""
function uniform_grid(dx, nghost)
    x1d = (-(nghost * dx)):dx:(1 + nghost * dx)
    y1d = (-(nghost * dx)):dx:(1 + nghost * dx)

    xy = zeros(2, length(x1d), length(y1d))
    for j in axes(xy, 3)
        for i in axes(xy, 2)
            xy[1, i, j] = x1d[i]
            xy[2, i, j] = y1d[j]
        end
    end

    return SSIDiffusion.Mesh2D(xy, nghost)
end

"""
Make a uniform grid where each vertex is perturbed
in a random direction by 20%
"""
function rand_grid(dx, nghost)
    r = 0.2dx

    x1d = (-(nghost * dx)):dx:(1 + nghost * dx)
    y1d = (-(nghost * dx)):dx:(1 + nghost * dx)

    xy_rand = zeros(2, length(x1d), length(y1d))
    for j in axes(xy_rand, 3)
        for i in axes(xy_rand, 2)
            x = x1d[i]
            y = y1d[j]

            if 0 < x < 1 && 0 < y < 1
                theta = rand() * 2pi
                δx = r * cos(theta)
                δy = r * sin(theta)
            else
                δx = 0.0
                δy = 0.0
            end

            xy_rand[1, i, j] = x + δx
            xy_rand[2, i, j] = y + δy
        end
    end

    return SSIDiffusion.Mesh2D(xy_rand, nghost)
end

nghost = 1
dx = 0.025/2
mesh = uniform_grid(dx, nghost)
# mesh = rand_grid(dx, nghost)
# mesh = wavy_grid(dx, nghost)

T0 = 1.0
bcs = (
    ilo=(:fixed, T0),
    ihi=(:fixed, 0.0),
    # jlo=:zero_flux,
    # jhi=:zero_flux,
    jlo=:periodic,
    jhi=:periodic,
)

T = zeros(size(mesh.volume))
Tⁿ⁺¹ = zeros(size(T))
q = zeros(size(T))
ρ = ones(size(T))
κ = zeros(size(T))

κ0 = 1.0
n = 3 # conductivity coeff in κ = κ0 * T^n
cᵥ = ones(size(T))

t0 = 0
tfinal = 0.2
Δt = 5e-6

# Stability criteria
Ts = 1e-3
ϵ0 = 0.2
ϵ1 = 0.02

solver = SSISolver2D(mesh, Ts, ϵ0, ϵ1)

global t = t0
global cycle = 0
cycle_max = Inf
update_mesh = false # the mesh is static
allow_subcycle = true
update_κ_cell!(κ, κ0, T, n)
# Solution loop
while t <= tfinal && cycle <= cycle_max

    # dtnext = SSIDiffusion._solve_single!(solver, mesh, Tⁿ⁺¹, T, ρ, q, cᵥ, κ, Δt, bcs, update_mesh, false)
    @timeit "SSI" begin
        nsubcyles = advance_solution_single_step!(
            solver,
            mesh,
            Tⁿ⁺¹,
            T,
            ρ,
            q,
            cᵥ,
            κ,
            Δt,
            bcs,
            update_mesh,
            allow_subcycle;
            dt_ceiling=false,
        )
    end
    @printf("cycle: %i, t:%.3e, dt:%.3e, subcycles:%d\n", cycle, t, Δt, nsubcyles)
    # if !isfinite(dtnext)
    #     global dtnext = Δt
    # end

    copy!(T, Tⁿ⁺¹)
    update_κ_cell!(κ, κ0, Tⁿ⁺¹, n)

    # if cycle % 1000 == 0
    #     writecontour(mesh.coords, Tⁿ⁺¹, cycle, t, "out")
    # end

    global t += Δt
    # global Δt = dtnext
    global cycle += 1
end

# writecontour(mesh.coords, Tⁿ⁺¹, cycle, t, "out")

front_pos_exact = 0.870571
x = [c[1] for c in mesh.centroid]
y = [c[2] for c in mesh.centroid]

T2d = @view Tⁿ⁺¹[2:(end - 1), 2:(end - 1)]
T1d = @view Tⁿ⁺¹[2:(end - 1), 2]
x1d = @view x[2:(end - 1), 2]
y1d = @view y[2, 2:(end - 1)]

fig = Figure(; resolution=(600, 900))
ax = Axis(
    fig[1, 1];
    xlabel="X",
    ylabel="Y",
    xticks=0:0.2:1,
    yticks=0:0.2:1,
    title="Nonlinear Conduction Into a Cold Medium",
    aspect=DataAspect(),
)

ax2 = Axis(
    fig[2, 1];
    xticks=0:0.2:1,
    yticks=0:0.2:1,
    xlabel="X",
    ylabel="Temperature",
    aspect=DataAspect(),
)

hm = heatmap!(ax, x, y, T2d; colormap=Reverse(:RdBu))
Colorbar(fig[1, 2], hm; label="Temperature")
xlims!(ax, 0, 1)
ylims!(ax, 0, 1)

scatterlines!(ax2, x1d, T1d)

vlines!(ax2, [front_pos_exact]; color=:black, label="Exact Front Position")
axislegend(ax2; position=:lb)
xlims!(ax2, 0, 1)
ylims!(ax2, 0, 1)
save("planar_nonlinear_cold_wall.png", fig)
display(fig)

# Search for the heat front position
function find_heat_front_position(T1d, x1d)
    @assert length(T1d) == length(x1d)
    front_pos = 0.0
    for i in reverse(eachindex(T1d))
        if T1d[i] > 1e-8
            front_pos = x1d[i + 1]
            break
        end
    end

    return front_pos
end

print_timer()

# front_pos = find_hcat_front_position(T1d, x1d)
# @testset "Heat Front Position vs Exact" begin
#     @test 0.98front_pos_exact < front_pos < 1.02front_pos_exact
# end
