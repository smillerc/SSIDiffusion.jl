using SSIDiffusion
using Printf
using CairoMakie
using Polyester
using SpecialFunctions

"""Update cell conductivity"""
function update_κ_cell!(κ, κ0, T, n=3)
    @batch for i in eachindex(T)
        κ[i] = κ0 * T[i]^n
    end
end

"""Dimensionless constant -- used in the analytical solution"""
function ξ(n)
    return (
        ((3n + 2) / (2^(n - 1) * n * π^n))^(1 / (3n + 2)) *
        (gamma(5 / 2 + 1 / n) / (gamma(1 + 1 / n) * gamma(3 / 2)))^(n / (3n + 2))
    )
end

"""Analytical solution for the central temperature"""
function central_temperature(t, n, Q0, ρ, cᵥ, κ0)
    ξ₁ = ξ(n)

    Tc = (
        ((n * ξ₁^2) / (2 * (3n + 2)))^(1 / n) *
        Q0^(2 / (3n + 2)) *
        ((ρ * cᵥ) / (κ0 * t))^(3 / (3n + 2))
    ) # central temperature

    return Tc
end

"""Analytical solution of T(r,t)"""
function analytical_sol(t, r, n, Q0, ρ, cᵥ, κ0)
    T = zeros(size(r))
    Tc = central_temperature(t, n, Q0, ρ, cᵥ, κ0)
    rf² = wave_front_radius(t, n, Q0, ρ, cᵥ, κ0)^2

    for i in eachindex(T)
        rterm = (1 - (r[i]^2 / rf²))
        if rterm > 0
            T[i] = Tc * rterm^(1 / n)
        end
    end

    return T
end

"""Analytical solution for the wave front radius"""
function wave_front_radius(t, n, Q0, ρ, cᵥ, κ0)
    ξ₁ = ξ(n)

    return ξ₁ * ((κ0 * t * Q0^n) / (ρ * cᵥ))^(1 / (3n + 2))
end

"""Make a uniform grid"""
function uniform_grid(dx, nghost)
    x1d = collect(0:dx:1)
    y1d = collect(0:dx:1)

    xy = zeros(2, length(x1d), length(y1d))
    for j in axes(xy, 3)
        for i in axes(xy, 2)
            xy[1, i, j] = x1d[i]
            xy[2, i, j] = y1d[j]
        end
    end

    return SSIDiffusion.Mesh2D(x1d, y1d, nghost; meshtype=:cylindrical, axis_of_symmetry=:x)
end

# Nonlinear thermal conduction
κ0 = 1.0
n = 2

# Mesh setup
nghost = 1
dx = 0.025
mesh = uniform_grid(dx, nghost)

# boundary conditions
bcs = (ilo=:reflect, ihi=:zeroflux, jlo=:reflect, jhi=:zeroflux)

T = zeros(size(mesh.volume))
Tⁿ⁺¹ = zeros(size(T))
q = zeros(size(T))
κ = zeros(size(T))
ρ = zeros(size(T))
cᵥ = zeros(size(T))

Q0 = 1.0
ρ0 = 1.0
cᵥ0 = 1.0

dx
r0 = 0.7dx

r1 = sqrt((dx)^2 + (dx)^2)

dep_vol = (4pi * dx * r0^2)
dep_vol = 4pi / 3 * (r1)^3
T0 = Q0 / (4pi * dx * r0^2)
# dep_vol = (4pi/3) * dx ^ 3
T0 = Q0 / dep_vol

Tfloor = 1e-30

# fill!(T, Tfloor)
fill!(ρ, ρ0)
fill!(cᵥ, cᵥ0)

T[2, 2] = T0
# q[2,2] = Q0

# Time
t0 = 0
tfinal = 0.3
Δt = 1e-5

# SSI solver setup
Ts = 1e-3 # max change in T per timestep
ϵ0 = 0.1 # stability constraints for the SSI method
ϵ1 = 0.02 # stability constraints for the SSI method
solver = SSISolver2D(mesh, Ts, ϵ0, ϵ1)
# copy!(solver.q, q)

# -----------------
# Solve!
# -----------------

global t = t0
global cycle = 0
cycle_max = Inf
global update_mesh = true

update_κ_cell!(κ, κ0, T, n)

while t <= tfinal && cycle <= cycle_max
    if cycle == 0
        dtnext = solve!(solver, Tⁿ⁺¹, T, ρ, cᵥ, mesh, Δt, bcs, true)
        # fill!(solver.q, 0.0)
    else
        dtnext = solve!(solver, Tⁿ⁺¹, T, ρ, cᵥ, mesh, Δt, bcs, false)
    end

    @printf("cycle: %i, t:%.3e, dt:%.3e\n", cycle, t, Δt)

    if !isfinite(dtnext)
        global dtnext = Δt
    end

    global update_mesh = false

    copy!(T, Tⁿ⁺¹)
    update_κ_cell!(κ, κ0, Tⁿ⁺¹, n)

    global t += Δt
    global Δt = dtnext
    global cycle += 1
end

# Plot the results
begin
    x = [c[1] for c in mesh.centroid]
    y = [c[2] for c in mesh.centroid]
    r = @. sqrt(x^2 + y^2)
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
        aspect=DataAspect(),
    )
    ax2 = Axis(
        fig[2, 1];
        xlabel="X",
        ylabel="Temperature",
        xticks=0:0.1:1,
        yticks=0:0.1:1,
        aspect=DataAspect(),
    )
    hm = heatmap!(ax, x1d, y1d, T2d)
    Colorbar(fig[1, 2], hm; label="Temperature")

    # scatterlines!(ax2, x1d, T1d)
    r2d = @view r[2:(end - 1), 2:(end - 1)]
    scatter!(ax2, vec(r2d), vec(T2d); markersize=5, label="SSI Method")
    rsol = 0:0.001:1
    Tsol = analytical_sol(tfinal, rsol, n, Q0, ρ0, cᵥ0, κ0)
    lines!(ax2, rsol, Tsol; color=:red, label="Analytical Solution")
    axislegend(ax2)

    xlims!(ax2, 0, 1)
    ylims!(ax2, 0, 1)
    display(fig)
end

# @benchmark solve!($solver, $Tⁿ⁺¹, $T, $ρ, $cᵥ, $mesh, $Δt, $bcs, false)
