using ThermoDiffusionMethods
using CairoMakie
using Printf

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

    return ThermoDiffusionMethods.Mesh2D(xy_wavy, nghost)
end

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

    return ThermoDiffusionMethods.Mesh2D(xy_rand, nghost)
end

function kershaw_grid() end

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

    return ThermoDiffusionMethods.Mesh2D(xy, nghost)
end

"""L2-norm truncation error"""
function δTʰL2(T, T_exact, mesh)
    err = 0.0
    vol = mesh.volume

    nghost = mesh.nghost
    CartInd = ThermoDiffusionMethods.getcartind(T, nghost)
    for idx in CartInd
        err += vol[idx] * (T[idx] - T_exact[idx])^2
    end

    return sqrt(err)
end

"""Max truncation error"""
function δTʰₘ(T, T_exact, nghost=1)
    err = -Inf
    CartInd = ThermoDiffusionMethods.getcartind(T, nghost)
    for idx in CartInd
        err_local = abs(T[idx] - T_exact[idx])
        err = max(err, err_local)
    end

    return err
end

qm_convergence(δT_h1_m, h1, δT_h2_m, h2) = log(δT_h1_m / δT_h2_m) / log(h1 / h2)
qL2_convergence(δT_h1_L2, h1, δT_h2_L2, h2) = log(δT_h1_L2 / δT_h2_L2) / log(h1 / h2)

"""Advance the solution to tfinal"""
function solve_SSI!(
    mesh, T, κ, ρ, cᵥ, t0, tfinal, Δt, bcs, Texact; fixed_Δt=false, error_target=1e-10
)
    Tⁿ = deepcopy(T)
    Tⁿ⁺¹ = zeros(size(T))

    Ts = 1e-3
    ϵ0 = 0.2
    ϵ1 = 0.04
    solver = SSISolver(mesh, Ts, ϵ0, ϵ1)

    fill!(solver.κ, κ)

    global t = t0
    global dt = Δt
    global cycle = 0
    cycle_max = 5000
    global update_mesh = true

    err = zeros(0)
    while t <= tfinal && cycle <= cycle_max
        if fixed_Δt
            _ = solve!(solver, Tⁿ⁺¹, Tⁿ, ρ, cᵥ, mesh, dt, bcs, false; fixed_dt=true)
            global dtnext = dt
        else
            dtnext = solve!(solver, Tⁿ⁺¹, Tⁿ, ρ, cᵥ, mesh, dt, bcs, false; fixed_dt=false)
            if !isfinite(dtnext)
                global dtnext = dt
            end
        end

        δT_L2 = δTʰL2(Tⁿ⁺¹, Texact, mesh)
        push!(err, δT_L2)
        if δT_L2 <= error_target
            return Tⁿ⁺¹, err
        end
        @printf("cycle: %i, t:%.3e, dt:%.3e, δTʰL2:%.3e\n", cycle, t, dt, δT_L2)

        global update_mesh = false

        copy!(Tⁿ, Tⁿ⁺¹)

        global t += dt
        global dt = dtnext
        global cycle += 1
    end

    return Tⁿ⁺¹, err
end

"""Setup and run the nonlinear steady state problem"""
function steady_state_nonlinear(dx, gridfunc::F) where {F}
    κ = 1
    a = 0
    b = 1 + (1 / (12κ))
    c = -1 / (12κ)
    # κ = 1/30
    # a = (1+8κ) / (6*(1+4κ))
    # b = (1+8κ) / (12κ*(1+4κ))
    # c = 1 / (12κ)

    kabc = (κ, a, b, c)

    Texact(x, y) = a + b * x + c * x^4
    Qsource(x, y) = x^2

    nghost = 1
    mesh = gridfunc(dx, nghost)

    x = [c[1] for c in mesh.centroid]
    y = [c[2] for c in mesh.centroid]

    T_exact = Texact.(x, y)

    bcs = (ilo=:periodic, ihi=:periodic, jlo=(:fixed, a), jhi=(:fixed, a + b + c))

    T = zeros(size(mesh.volume))
    Q = Qsource.(y, x)

    ρ = ones(size(T))
    κall = zeros(size(T))
    fill!(κall, κ)

    cᵥ = ones(size(T))

    t0 = 0
    tf = 2
    Δt = 1e-3

    fixed_dt = false
    Tfinal, errs = solve_SSI!(
        mesh, T, κ, ρ, cᵥ, t0, tf, Δt, bcs, T_exact; fixed_Δt=fixed_dt
    )

    δT_L2 = δTʰL2(Tfinal, T_exact, mesh)
    δT_m = δTʰₘ(Tfinal, T_exact)

    return mesh, kabc, T_exact, Tfinal, δT_L2, δT_m, errs
end

function plot_res(sol, ax, plot_exact, name)
    mesh, kabc, T_exact, Tfinal, δT_L2, δT_m, errs = sol

    κ, a, b, c = kabc

    x = [c[1] for c in mesh.centroid]
    y = [c[2] for c in mesh.centroid]

    label = @sprintf("%s SSI -- L₂norm = %.1e, δTₘ = %.1e", name, δT_L2, δT_m)
    scatter!(ax, vec(y), vec(Tfinal); label=label, markersize=8)
    # heatmap!(x,y,Tfinal)

    if plot_exact
        dx = 0.05
        Texact(x) = a + b * x + c * x^4
        x_exact = (-dx):dx:(1 + dx)
        T_e = Texact.(x_exact)
        lines!(ax, x_exact, T_e; label="Exact")
    end

    return axislegend(ax; position=:lt)

    # xlims!(ax, 0, 1)
    # ylims!(ax, 0, 1)
end

uniform_mesh = steady_state_nonlinear(0.05, uniform_grid)
# rand_mesh = steady_state_linear(0.05, rand_grid)
# wavy_mesh = steady_state_linear(0.05, wavy_grid)

fig = Figure()
ax = Axis(
    fig[1, 1];
    aspect=DataAspect(),
    title="Steady-State Nonlinear Conduction: T(x,y) = a + bx + cx⁴",
)
# plot_res(rand_mesh, ax, false  , "Wavy Mesh")
# plot_res(rand_mesh, ax, false  , "Random Mesh")
plot_res(uniform_mesh, ax, true, "Uniform Mesh")
display(fig)
