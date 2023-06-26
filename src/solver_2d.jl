using Printf, StaticArrays, Polyester
using InteractiveUtils
using TimerOutputs
using BenchmarkTools

"""
The SSIsolver, or Symmetric Semi-Implicit method
"""
struct SSISolver2D{AA1,AA2,AA3}
    # Tⁿ⁺¹::AA1 # next temperature [i,j]
    T_vertex::AA1 # vertex temperature [i,j]
    # κ::AA1 # cell conductivity [i,j]
    τ::AA1 # temperature increment for this step [i,j]
    δ::AA1 # energy lost (per-cell) during the last timestep [i,j]
    # q::AA1 # source term [i,j]
    δ̃::AA2 # energy lost (per-face) during the last timestep [i,j]
    a::AA2 # SSI coefficient [1:2,i,j]
    b::AA2 # SSI coefficient [1:2,i,j]
    H::AA2 # face fluxes [1:2,i,j]
    ξ::AA1 # bilinear coefficient, e.g., (i,j) → (ξ, η)
    η::AA1 # bilinear coefficient, e.g., (i,j) → (ξ, η)
    Χ::AA2 # face split weights [1:2,i,j]
    μ::AA3 # interpolation coeff [1:4,i,j]
    Ts::Vector{Float64} # Temperature variance threshold -- scalar, but the vector allows us to change
    ϵ0::Vector{Float64} # SSI time-step control tolerance τ -- scalar, but the vector allows us to change
    ϵ1::Vector{Float64} # SSI time-step control tolerance for δ -- scalar, but the vector allows us to change
    fluxlimiter::Vector{Float64}
end

set_Ts(SSI, Ts) = SSI.Ts[begin] = Ts
set_ϵ0(SSI, ϵ0) = SSI.ϵ0[begin] = ϵ0
set_ϵ1(SSI, ϵ1) = SSI.ϵ1[begin] = ϵ1

get_Ts(SSI) = SSI.Ts[begin]
get_ϵ0(SSI) = SSI.ϵ0[begin]
get_ϵ1(SSI) = SSI.ϵ1[begin]

"""
    SSISolver2D(Tⁿ::AbstractMatrix{T}, ϵ0, ϵ1, Ts) where {T}

Create the SSI solver type

# Arguments
 - `Tⁿ`: Initial temperature
 - `ϵ0`: time-step control factor
 - `ϵ1`:
 - `Ts`:
"""
function SSISolver2D(mesh::Mesh2D, Ts, ϵ0=0.2, ϵ1=0.05, fluxlimiter=1.0)
    M, N = size(mesh.volume)

    @assert ϵ0 > ϵ1 "The SSI time-step control factor ϵ0 must be > ϵ1"
    @assert ϵ1 > 0 "The SSI time-step control factor ϵ1 must be > 0 [Default is 0.05]"
    @assert Ts > 0 "The SSI temperature variance threshold `Ts` must be > 0"
    @assert fluxlimiter > 0 "The SSI flux limiter factor must be > 0"

    Tvertex = zeros(M + 1, N + 1) # T_vertex
    ξ = zeros(M + 1, N + 1) # ξ::AA1 # bilinear coefficient, e.g., (i,j) → (ξ, η)
    η = zeros(M + 1, N + 1) # η::AA1 # bilinear coefficient, e.g., (i,j) → (ξ, η)

    iterator_range = getcartind(Tvertex, mesh.nghost)
    update_bilinear_coeff!(mesh, ξ, η)

    return SSISolver2D(
        # zeros(M, N), # Tⁿ⁺¹::AA1 # next temperature [i,j]
        Tvertex, # ::AA1 # vertex temperature [i,j]
        # zeros(M, N), # κ::AA1 # cell conductivity [I,j]
        zeros(M, N), # τ::AA1 # temperature increment for this step [i,j]
        zeros(M, N), # δ::AA1 # energy lost (per-cell) last timestep [i,j]
        # zeros(M, N), # q::AA1 # source term [i,j]
        zeros(2, M, N), # δ̃::AA2 # energy lost (per-face) during the last timestep [i,j]
        zeros(2, M, N), # a::AA1 # SSI coefficient [1:2,i,j]
        zeros(2, M, N), # b::AA1 # SSI coefficient [1:2,i,j]
        zeros(2, M, N), # H::AA2 # face fluxes [1:2,i,j]
        ξ, #::AA1 # bilinear coefficient, e.g., (i,j) → (ξ, η)
        η, #::AA1 # bilinear coefficient, e.g., (i,j) → (ξ, η)
        zeros(2, M, N), #Χ::AA2 # face split weights [1:2,i,j]
        zeros(4, M + 1, N + 1), # μ::AA3 # interpolation coeff [1:4,i,j]
        [ϵ0],
        [ϵ1],
        [Ts],
        [fluxlimiter],
    )
end

"""
Options:
1. advance the solution exactly by Δt and allow subcycling determined by max Δt allowable, e.g. given Δt by hydro, split into smaller steps if necessary
2. advance the solution exactly by Δt in a single step regardless of error
"""

"""
    advance_solution_single_step!(SSI::SSISolver2D, mesh, Tⁿ⁺¹, Tⁿ, ρⁿ, q, cᵥ, κ, Δt, boundary_conditions,
    update_geo::Bool, allow_subcycle::Bool, max_nsubcycles=100, dt_drop_tolerance=1e-3, dt_ceiling=true)

# Arguments
 - `SSI::SSISolver2D`: The SSI solver 
 - `mesh`: The mesh type
 - `Tⁿ⁺¹`: New temperature
 - `Tⁿ`: Current temperature
 - `ρⁿ`: Density
 - `q`: Volumetric source term, e.g. J/cm^3
 - `cᵥ`: Specific heat at constant volume
 - `κ`: Cell conductivity
 - `Δt `: desired time-step
 - `boundary_conditions`: 
 - `update_geo::Bool`: 
 - `allow_subcycle::Bool`: 
 - `max_nsubcycles=100`: Maximum allowable subcycles
 - `dt_drop_tolerance=1e-3`: Amount that the timestep is allowed to drop by; this prevents it from tanking to 0
 - `dt_ceiling=true`: Don't let the timestep get larger than the one specified; Setting this to false is useful when you want to let
    the solver choose the largest possible timestep
"""
function advance_solution_single_step!(
    SSI::SSISolver2D,
    mesh,
    Tⁿ⁺¹,
    Tⁿ,
    ρⁿ,
    q,
    cᵥ,
    κ,
    Δt,
    boundary_conditions,
    update_geo::Bool,
    allow_subcycle::Bool;
    max_nsubcycles=Inf,
    dt_drop_tolerance=1e-3,
    dt_ceiling=true,
)
    nsubcycles = 0

    if !allow_subcycle
        @timeit "solve_single" _solve_single!(
            SSI,
            mesh,
            Tⁿ⁺¹,
            Tⁿ,
            ρⁿ,
            q,
            cᵥ,
            κ,
            Δt,
            boundary_conditions,
            update_geo,
            true,
            true,
        )
        return nsubcycles
    else
        t = 0.0
        subcycle_dt = Δt
        while true
            if nsubcycles > max_nsubcycles
                error("Maximum SSI subcycle limit ($max_nsubcycles) reached")
            end

            # when subcycling is allowed, it returns the timestep that it actually took (can be more or less than specified by Δt)
            # and `dt_ceil` limits the timestep to be no more than given by `subcycle_dt`
            dt_ceil = true
            @timeit "solve_single" begin
                dt_actual = _solve_single!(
                SSI,
                mesh,
                Tⁿ⁺¹,
                Tⁿ,
                ρⁿ,
                q,
                cᵥ,
                κ,
                subcycle_dt,
                boundary_conditions,
                update_geo,
                false,
                dt_ceil,
            )
            end
            t += dt_actual

            @timeit "next_timestep" begin
            _, dt_next = next_timestep(Δt, dt_actual)
            end

            # Catch an overshoot of Δt
            if t + dt_next > Δt
                dt_next = Δt - t
            end

            nsubcycles += 1
            subcycle_dt = dt_next

            if iszero(abs(t - Δt))
                break
            end
        end

        return nsubcycles
    end
end

function next_timestep(dt_full, actual_dt, dt_drop_tolerance=1e-3)
    if actual_dt < dt_full
        next_dt = dt_full - actual_dt
        advance = true

        if (next_dt < dt_full * dt_drop_tolerance)
            @warn "The SSI solver timestep is dropping below the threshold given by Δt * $dt_drop_tolerance"
        end
    else
        advance = false
        next_dt = 0.0
    end

    return advance, next_dt
end

"""
3. advance the solution from t0 to tfinal let SSI determine the Δt
4. advance the solution from t0 to tfinal with a pre-determined Δt (regardless of error)
"""
function advance_solution_time_range!(
    SSI::SSISolver2D,
    mesh,
    Tⁿ⁺¹,
    Tⁿ,
    ρⁿ,
    q,
    cᵥ,
    κ,
    t_init,
    Δt,
    t_final,
    boundary_conditions,
    update_geo::Bool,
    allow_subcycle::Bool,
    max_cycles=Inf,
    max_nsubcycles=Inf,
    dt_drop_tolerance=1e-3,
)
    @assert isfinite(t_init)
    @assert isfinite(Δt)
    @assert isfinite(t_final)
    @assert t_final > t_init
    @assert Δt < t_final

    t = t_init
    dt = Δt # the timestep used in this function -- can be updated depending on certain criteria below
    cycle = 0
    while t <= t_final && cycle <= max_cycles
        dtnext = advance_solution_single_step!(
            SSI,
            mesh,
            Tⁿ⁺¹,
            Tⁿ,
            ρⁿ,
            q,
            cᵥ,
            κ,
            dt,
            boundary_conditions,
            update_geo,
            allow_subcycle,
            max_nsubcycles,
            dt_drop_tolerance,
        )
        if !isfinite(dtnext)
            @warn "Non-finite SSI timestep found, defaulting to provided Δt"
            dtnext = Δt
        end

        if allow_subcycle
            # SSI can choose the next dt based on stability constraints 
            # this can be advantageous for speed
            dt = dtnext
        end

        if t + dt > t_final
            dtnext = t_final - t
        end

        t += dt
        cycle += 1
    end

    return nothing
end

"""
"""
function _solve_single!(
    SSI::SSISolver2D,
    mesh,
    Tⁿ⁺¹::AbstractMatrix{NT},
    Tⁿ,
    ρⁿ,
    q,
    cv,
    κ,
    dt::Float64,
    bcs,
    update_geo::Bool,
    fixed_dt=false,
    dt_ceiling=true,
) where {NT}
    ϵ_tol = 10eps(NT)
    
    @timeit "applybc!" applybc!(Tⁿ, bcs, mesh.nghost) # update the ghost cell temperatures

    nghost = mesh.nghost # number of ghost/halo cells

    ilohi = axes(ρⁿ, 1)
    jlohi = axes(ρⁿ, 2)
    ilo = first(ilohi) + nghost
    jlo = first(jlohi) + nghost
    ihi = last(ilohi) - nghost
    jhi = last(jlohi) - nghost

    looplimits = (ilo, ihi, jlo, jhi)
    ξ = SSI.ξ # bilinear coefficient, e.g., (i,j) → (ξ, η)
    η = SSI.η # bilinear coefficient, e.g., (i,j) → (ξ, η)
    μ = SSI.μ # interpolation coeff [1:4,i,j]
    H = SSI.H # face fluxes [1:2,i,j]
    τ = SSI.τ # temperature increment for this step [i,j]
    δ = SSI.δ # energy lost (per-cell) during the last timestep [i,j]
    δ̃ = SSI.δ̃ # energy lost (per-face) during the last timestep
    a = SSI.a # SSI coefficient [i,j]
    b = SSI.b # SSI coefficient [i,j]
    Χ = SSI.Χ # split weight
    vol = mesh.volume # cell volume [i,j]

    ϵ1 = get_ϵ1(SSI)
    ϵ0 = get_ϵ0(SSI)

    # update μ
    @timeit "interp_coeff" interp_coeff!(μ, κ, ξ, η, nghost)

    # Get the face-adjacent data used for splitting the energy corrections.
    # If the mesh is static, only do this once
    if update_geo
        @timeit "update_face_geometric_coeff!" update_face_geometric_coeff!(mesh)
    end

    @timeit "faceweights!" faceweights!(SSI.Χ, mesh, cv, ρⁿ) # update Χ
    @timeit "update_Χ_bc!" update_Χ_bc!(SSI.Χ, nghost) # update Χ for boundary faces
    @timeit "vertex_temperatures!" vertex_temperatures!(SSI.T_vertex, Tⁿ, μ, nghost)
    @timeit "face_centered_flux!" face_centered_flux!(H, a, b, Tⁿ, SSI.T_vertex, κ, SSI.μ, mesh)
    @timeit "update_Hab_bc!" update_Hab_bc!(H, a, b, bcs, nghost) # update fluxes and a/b coeff for boundary faces

    # There are two separate time constraints for the SSI method:
    #   1. ϵ0 - limit τ (based on flux H and a,b SSI coefficients)
    #   2. ϵ1 - limit δᵢⱼ (maximum amount of energy correction)
    # Each constraint must be checked individualy (and in order listed)
    global Δt = dt
    global subcycle_Δt = 0.0
    if fixed_dt
        subcycle_Δt = dt
    else
        Δt_max, _ = calc_timestep(SSI, mesh, Tⁿ, ρⁿ, cv, q)
        if !isfinite(Δt_max)
            Δt_max = dt
        end
        subcycle_Δt = Δt_max
    end

    # don't exceed the specified time step, e.g., when this is dictated by other external constraints
    if dt_ceiling
        subcycle_Δt = min(subcycle_Δt, dt)
    end

    # Loop until all criteria is satisfied (ϵ0, ϵ1, Ts)
    while true
        if fixed_dt
            Δt = dt
        else
            while true
                @timeit "check_τ_constraint" begin
                valid_dt_ϵ0, max_ΔT_ϵ0 = check_τ_constraint(
                    SSI, mesh, Tⁿ, ρⁿ, cv, q, subcycle_Δt
                )
            end


                if valid_dt_ϵ0
                    Δt = subcycle_Δt
                    break
                else
                    new_Δt = 0.5subcycle_Δt * ϵ0 / max_ΔT_ϵ0
                    str = @sprintf(
                        "Reducing Δt based on the ϵ0 constraint from %.3e to %.3e",
                        subcycle_Δt,
                        new_Δt
                    )
                    @info str
                    subcycle_Δt = new_Δt
                end
            end
        end

        # Find τᵢⱼ, the temperature increment for the current cell
        @timeit "temp_increment" begin
            increment_temp!(SSI.τ, ρⁿ, mesh.volume , SSI.H, SSI.a, SSI.b, q, SSI.δ, cv, Δt, looplimits)
        end

        @timeit "finite_val_check(τ)" finite_val_check(τ)

        # Calculate the energy lost (δ̃ᵢⱼₘ) at each face
        @timeit "energy_lost" begin
        for j in jlo:(jhi + 1)
            for i in ilo:(ihi + 1)
                δ̃[1, i, j] = (a[1, i, j] * τ[i, j] + b[1, i, j] * τ[i, j - 1])
                δ̃[2, i, j] = (a[2, i, j] * τ[i, j] + b[2, i, j] * τ[i - 1, j])
            end
        end
        end

        # determine the energy correction (δᵢⱼ) for the next timestep
        @timeit "energy correction" begin
            energy_correction!(SSI.δ, SSI.Χ, SSI.δ̃, Δt, looplimits)
        end

        @timeit "finite_val_check(δ)" finite_val_check(δ)

        # check the ϵ1 constraint based on maximum energy correction (δ)
        if fixed_dt
            break
        end

        @timeit "calc_δ_constraint" begin
            valid_δ_ϵ1, max_ΔT_ϵ1 = calc_δ_constraint(SSI, Tⁿ, ρⁿ, cv, mesh)
        end

        if valid_δ_ϵ1
            break
        else
            new_Δt = 0.5subcycle_Δt * ϵ1 / max_ΔT_ϵ1
            @info @sprintf(
                "Reducing Δt based on the ϵ1 constraint from %.3e to %.3e",
                subcycle_Δt,
                new_Δt
            )
            subcycle_Δt = new_Δt
        end
    end

    # update the temperature
    @timeit "update_next" begin
    @batch for idx in eachindex(Tⁿ⁺¹)
        @inbounds Tⁿ⁺¹[idx] = Tⁿ[idx] + τ[idx]
    end
    end

    return Δt
end


function increment_temp!(τ, ρ, vol, H, a, b, q, δ, cv, Δt, looplimits::NTuple{4,Int})

    ϵ_tol = eps(Float64)
    ilo, ihi, jlo, jhi = looplimits

    @batch for j in jlo:jhi
        for i in ilo:ihi
            # δᵢⱼ is the energy correction from the previous step

            Mᵢⱼ = ρ[i, j] * vol[i, j]

            H1 = H[1, i, j] - H[1, i, j + 1]
            H2 = H[2, i, j] - H[2, i + 1, j]
            H1 = H1 * (abs(H1) >= ϵ_tol)
            H2 = H2 * (abs(H2) >= ϵ_tol)
            # Hsum = round(H1 + H2, sigdigits=SD)
            Hsum = H1 + H2

            τ[i, j] = (
                (Δt * (Hsum + q[i, j] * Mᵢⱼ) + δ[i, j]) / (
                    cv[i, j] * Mᵢⱼ +
                    (a[1, i, j] + a[2, i, j] + b[1, i, j + 1] + b[2, i + 1, j]) * Δt
                )
            )
        end
    end

    nothing
end

function energy_correction!(δ, Χ, δ̃, Δt::Float64, looplimits::NTuple{4,Int})

    ilo, ihi, jlo, jhi = looplimits

    @batch for j in jlo:jhi
        for i in ilo:ihi
            δ[i, j] =
                (
                    Χ[1, i, j] * δ̃[1, i, j] +
                    Χ[2, i, j] * δ̃[2, i, j] +
                    (1.0 - Χ[1, i, j + 1]) * δ̃[1, i, j + 1] +
                    (1.0 - Χ[2, i + 1, j]) * δ̃[2, i + 1, j]
                ) * Δt

            # δ[i, j] = round(δij, sigdigits=SD)
            # δ[i, j] = δij
        end
    end
    return nothing
end

"""Determine the appropriate timestep based on the τ constraint"""
function subcycle_tau_constraint_timestep(SSI::SSISolver2D, mesh, Tⁿ, ρⁿ, cv, q, subcycle_Δt)
    Δt = 0.0
    ϵ0 = get_ϵ0(SSI)

    # Loop until we get a valid timestep
    while true
        valid_dt_ϵ0, max_ΔT_ϵ0 = check_τ_constraint(SSI, mesh, Tⁿ, ρⁿ, cv, q, subcycle_Δt)

        if valid_dt_ϵ0
            Δt = subcycle_Δt
            break
        else
            new_Δt = 0.5subcycle_Δt * ϵ0 / max_ΔT_ϵ0
            @info @sprintf(
                "Reducing Δt based on the ϵ0 constraint from %.3e to %.3e",
                subcycle_Δt,
                new_Δt
            )
            subcycle_Δt = new_Δt
        end
    end

    return Δt
end

"""Check for non-finite values in A with a ghost cell layer `nghost` cells thick"""
function finite_val_check(A, nghost=0)
    iterator_range = getcartind(A, nghost)
    for I in iterator_range
        if !isfinite(A[I])
            error("Non-finite values found at $(I)")
        end
    end
end

"""
Determine the maximum allowable timestep for the next cycle
"""
function calc_timestep(SSI::SSISolver2D, mesh, Tⁿ, ρⁿ, cv, q)
    nghost = mesh.nghost # number of ghost/halo cells

    ϵ0 = get_ϵ0(SSI)
    ϵ1 = get_ϵ1(SSI)
    Ts = get_Ts(SSI)

    # q = SSI.q
    H = SSI.H
    a = SSI.a
    b = SSI.b
    ilohi = axes(q, 1)
    jlohi = axes(q, 2)
    ilo = first(ilohi) + nghost
    jlo = first(jlohi) + nghost
    ihi = last(ilohi) - nghost
    jhi = last(jlohi) - nghost

    Δt_next = Inf
    δ_max = Inf

    for j in jlo:jhi
        for i in ilo:ihi
            M = ρⁿ[i, j] * mesh.volume[i, j]
            cvM = cv[i, j] * M

            H1 = H[1, i, j] - H[1, i, j + 1]
            H2 = H[2, i, j] - H[2, i + 1, j]

            W_T = H1 + H2

            D_T = a[1, i, j] + a[2, i, j] + b[1, i, j + 1] + b[2, i + 1, j]

            Δt = abs(cvM / ((abs(W_T + q[i, j] * M) / ((ϵ0 - ϵ1) * (Ts + Tⁿ[i, j]))) - D_T))

            # @show Δt, W_T, q[i,j], M, D_T, H[:, i, j]
            Δt_next = minimum(x -> isnan(x) ? Inf : x, (Δt_next, Δt))

            δij = abs(cvM * ϵ1 * (Ts + Tⁿ[i, j]))
            δ_max = min(δ_max, δij)
        end
    end

    return Δt_next, δ_max
end

function check_τ_constraint(SSI::SSISolver2D, mesh, T, ρ, cv, q, Δt)
    nghost = mesh.nghost # number of ghost/halo cells

    ϵ0 = get_ϵ0(SSI)
    ϵ1 = get_ϵ1(SSI)
    Ts = get_Ts(SSI)

    # q = SSI.q
    H = SSI.H
    a = SSI.a
    b = SSI.b
    ilohi = axes(T, 1)
    jlohi = axes(T, 2)
    ilo = first(ilohi) + nghost
    jlo = first(jlohi) + nghost
    ihi = last(ilohi) - nghost
    jhi = last(jlohi) - nghost
    vol = mesh.volume

    max_ΔT = -Inf
    for j in jlo:jhi
        for i in ilo:ihi
            Mᵢⱼ = ρ[i, j] * vol[i, j]

            τ0ᵢⱼ = abs(
                (
                    (
                        H[1, i, j] + H[2, i, j] - H[1, i, j + 1] - H[2, i + 1, j] +
                        q[i, j] * Mᵢⱼ
                    ) * Δt
                ) / (
                    cv[i, j] * Mᵢⱼ +
                    (a[1, i, j] + a[2, i, j] + b[1, i, j + 1] + b[2, i + 1, j]) * Δt
                ),
            )

            # check if Δt is too much
            # if τ0ᵢⱼ > (ϵ0 - ϵ1) * (T[i, j] + Ts) return false end
            ΔT = τ0ᵢⱼ / (T[i, j] + Ts)

            # nonallocating max that handles NaNs
            max_ΔT = maximum(x -> isnan(x) ? -Inf : x, (max_ΔT, ΔT))
        end
    end

    valid_Δt = max_ΔT <= (ϵ0 - ϵ1)

    return valid_Δt, max_ΔT
end

function next_subcycle_dt_ϵ0(SSI, mesh, Tⁿ, ρⁿ, cv, q, dt, subcycle_Δt, fixed_dt)
    ϵ1 = get_ϵ1(SSI)
    ϵ0 = get_ϵ0(SSI)

    if fixed_dt
        Δt = dt
    else
        while true
            valid_dt_ϵ0, max_ΔT_ϵ0 = check_τ_constraint(
                SSI, mesh, Tⁿ, ρⁿ, cv, q, subcycle_Δt
            )

            if valid_dt_ϵ0
                Δt = subcycle_Δt
                break
            else
                new_Δt = 0.5subcycle_Δt * ϵ0 / max_ΔT_ϵ0
                str = @sprintf(
                    "Reducing Δt based on the ϵ0 constraint from %.3e to %.3e",
                    subcycle_Δt,
                    new_Δt
                )
                @info str
                subcycle_Δt = new_Δt
            end
        end
    end

    return subcycle_Δt
end

function calc_δ_constraint(SSI::SSISolver2D, T, ρ, cv, mesh)
    nghost = mesh.nghost # number of ghost/halo cells

    ϵ1 = get_ϵ1(SSI)
    Ts = get_Ts(SSI)

    δ = SSI.δ

    ilohi = axes(T, 1)
    jlohi = axes(T, 2)
    ilo = first(ilohi) + nghost
    jlo = first(jlohi) + nghost
    ihi = last(ilohi) - nghost
    jhi = last(jlohi) - nghost
    vol = mesh.volume
    max_ΔT = -Inf

    for j in jlo:jhi
        for i in ilo:ihi
            Mᵢⱼ = ρ[i, j] * vol[i, j]
            τ1ᵢⱼ = abs(δ[i, j] / (cv[i, j] * Mᵢⱼ))
            ΔT = τ1ᵢⱼ / (T[i, j] + Ts)

            # nonallocating max that handles NaNs
            max_ΔT = maximum(x -> isnan(x) ? -Inf : x, (max_ΔT, ΔT))
        end
    end

    valid_Δt = max_ΔT <= ϵ1

    return valid_Δt, max_ΔT
end

# Helper functions

"""
	faceweights!(Χ, CΔ, cv, ρ, iterator_range, nghost)

Find the split weights Χ that are proportional to the bulk heat capacities of each cell adjacent to an edge.
"""
@inline function faceweights!(Χ, mesh, cv, ρ)

    # Split weights proportional to the buld heat capacities of the two triangles
    # on either side of the face. See Eq. 19

    ilohi = axes(ρ, 1)
    jlohi = axes(ρ, 2)
    ilo = first(ilohi) + mesh.nghost
    jlo = first(jlohi) + mesh.nghost
    ihi = last(ilohi) - mesh.nghost+1
    jhi = last(jlohi) - mesh.nghost+1

    C⁺Δ = mesh.C⁺Δᵢⱼ
    C⁻Δ = mesh.C⁻Δᵢⱼ

    @batch for j in jlo:jhi
        for i in ilo:ihi

            C⁺Δᵢⱼ₁ = C⁺Δ[1, i, j] * cv[i, j] * ρ[i, j]
            C⁺Δᵢⱼ₂ = C⁺Δ[2, i, j] * cv[i, j] * ρ[i, j]
            C⁻Δᵢⱼ₁ = C⁻Δ[1, i, j] * cv[i, j - 1] * ρ[i, j - 1]
            C⁻Δᵢⱼ₂ = C⁻Δ[2, i, j] * cv[i - 1, j] * ρ[i - 1, j]

            Χ[1, i, j] = C⁺Δᵢⱼ₁ / (C⁺Δᵢⱼ₁ + C⁻Δᵢⱼ₁)
            Χ[2, i, j] = C⁺Δᵢⱼ₂ / (C⁺Δᵢⱼ₂ + C⁻Δᵢⱼ₂)
        end
    end

    return nothing
end

@inline function face_conductivity(AΔ⁺, AΔ⁻, κ, κneighbor)
    # weighted arithmetic mean
    κ_face = (((κ * AΔ⁻) / (AΔ⁻ + AΔ⁺)) + ((κneighbor * AΔ⁺) / (AΔ⁻ + AΔ⁺)))

    return κ_face
end

"""

Compute the face-centered flux based on the cell-centered temperature `Tc` and vertex temperatures `Tv`, and update the SSI coefficients `a`, and `b`.
"""
function face_centered_flux!(H::AbstractArray{T}, a, b, Tc, Tv, κ, μ, mesh) where {T}
    ϵ = 10eps(T)
    coords = mesh.coords
    centroids = mesh.centroid
    alpha = mesh.alpha
    rcoord = mesh.rcoord

    AΔ⁺ = mesh.A⁺Δᵢⱼ
    AΔ⁻ = mesh.A⁻Δᵢⱼ

    nghost = mesh.nghost # number of ghost/halo cells

    ilohi = axes(Tc, 1)
    jlohi = axes(Tc, 2)
    ilo = first(ilohi) + nghost
    jlo = first(jlohi) + nghost
    ihi = last(ilohi) - nghost
    jhi = last(jlohi) - nghost

    # Loop through the domain cells
    iterator_range = CartesianIndices((ilo:(ihi + 1), jlo:(jhi + 1)))

    @batch for idx in iterator_range
        i, j = Tuple(idx)

        x⃗ᵢⱼ = SVector{2,Float64}(view(coords, :, i, j))
        x⃗ᵢ₊₁ⱼ = SVector{2,Float64}(view(coords, :, i + 1, j))
        x⃗ᵢⱼ₊₁ = SVector{2,Float64}(view(coords, :, i, j + 1))
        x⃗cᵢⱼ = centroids[i, j]
        x⃗cᵢⱼ₋₁ = centroids[i, j - 1]
        x⃗cᵢ₋₁ⱼ = centroids[i - 1, j]

        # vector connecting the vertices that define the face
        Ivᵢⱼₘ = @SVector [
            x⃗ᵢ₊₁ⱼ - x⃗ᵢⱼ, # m = 1
            x⃗ᵢⱼ₊₁ - x⃗ᵢⱼ, # m = 2
        ]

        # vector connecting to the centroids of neighboring faces
        Icᵢⱼₘ = @SVector [
            x⃗cᵢⱼ - x⃗cᵢⱼ₋₁, # m = 1
            x⃗cᵢⱼ - x⃗cᵢ₋₁ⱼ, # m = 2
        ]

        # @show Icᵢⱼₘ x⃗cᵢⱼ x⃗cᵢⱼ₋₁ x⃗cᵢ₋₁ⱼ
        Rᵢⱼ = x⃗ᵢⱼ[rcoord]^alpha
        Rᵢ₊₁ⱼ = x⃗ᵢ₊₁ⱼ[rcoord]^alpha
        Rᵢⱼ₊₁ = x⃗ᵢⱼ₊₁[rcoord]^alpha

        # Eq. 28
        Rfᵢⱼₘ = @SVector [0.5(Rᵢⱼ + Rᵢ₊₁ⱼ), 0.5(Rᵢⱼ + Rᵢⱼ₊₁)]

        # Equations 30 & 31
        ΔTvᵢⱼₘ = @SVector [
            Tv[i + 1, j] - Tv[i, j], # m = 1
            Tv[i, j + 1] - Tv[i, j], # m = 2
        ]

        ΔTcᵢⱼₘ = @SVector [
            Tc[i, j] - Tc[i, j - 1], # m = 1
            Tc[i, j] - Tc[i - 1, j], # m = 2
        ]

        # interpolation coefficients used in aᵢⱼₘ and bᵢⱼₘ
        μaᵢⱼ₁ = μ[1, i, j] - μ[2, i + 1, j]
        μaᵢⱼ₂ = μ[1, i, j] - μ[4, i, j + 1]
        μbᵢⱼ₁ = μ[3, i + 1, j] - μ[4, i, j]
        μbᵢⱼ₂ = μ[3, i, j + 1] - μ[2, i, j]

        # check for machine epsilon (zero-out if less than ~1e-15 for Float64)
        μaᵢⱼ₁ = μaᵢⱼ₁ * (abs(μaᵢⱼ₁) >= ϵ)
        μaᵢⱼ₂ = μaᵢⱼ₂ * (abs(μaᵢⱼ₂) >= ϵ)
        μbᵢⱼ₁ = μbᵢⱼ₁ * (abs(μbᵢⱼ₁) >= ϵ)
        μbᵢⱼ₂ = μbᵢⱼ₂ * (abs(μbᵢⱼ₂) >= ϵ)
        μa = @SVector [μaᵢⱼ₁, μaᵢⱼ₂]
        μb = @SVector [μbᵢⱼ₁, μbᵢⱼ₂]

        κfᵢⱼₘ = @SVector [
            face_conductivity(AΔ⁺[1, i, j], AΔ⁻[1, i, j], κ[i, j], κ[i, j - 1]),
            face_conductivity(AΔ⁺[2, i, j], AΔ⁻[2, i, j], κ[i, j], κ[i - 1, j]),
        ]

        # these terms are re-used in 3 equations
        # crossterm = @SVector [
        #     2(AΔ⁺[1, i, j] + AΔ⁻[1, i, j]),
        #     2(AΔ⁺[2, i, j] + AΔ⁻[2, i, j]),
        # ]

        # Iv_norm_sq = norm(Ivᵢⱼₘ)^2

        crossterm = @. abs(cross(Ivᵢⱼₘ, Icᵢⱼₘ))
        Iv_norm_sq = @. norm(Ivᵢⱼₘ)^2
        κterm = @. (κfᵢⱼₘ * Rfᵢⱼₘ) / crossterm
        dot_term = @. Ivᵢⱼₘ ⋅ Icᵢⱼₘ

        H_ᵢⱼₘ = @view H[:, i, j]
        a_ᵢⱼₘ = @view a[:, i, j]
        b_ᵢⱼₘ = @view b[:, i, j]

        @. H_ᵢⱼₘ = κterm * (dot_term * ΔTvᵢⱼₘ - ΔTcᵢⱼₘ * Iv_norm_sq)
        @. a_ᵢⱼₘ = κterm * (Iv_norm_sq + dot_term * μa)
        @. b_ᵢⱼₘ = κterm * (Iv_norm_sq + dot_term * μb)

        # @inbounds for m in 1:2
        #     H[m, i, j] = H_ᵢⱼₘ[m]
        #     a[m, i, j] = a_ᵢⱼₘ[m]
        #     b[m, i, j] = b_ᵢⱼₘ[m]
        # end
        # @show κterm1, dot_term1, Iv_norm_sq
        # @show norm(Ivᵢⱼₘ)^2, norm.(Ivᵢⱼₘ), (norm.(Ivᵢⱼₘ)).^2
        # @inbounds @simd for m in 1:2
        #     # κterm = (κfᵢⱼₘ[m] * Rfᵢⱼₘ[m]) / crossterm[m]
        #     # dot_term = Ivᵢⱼₘ[m] ⋅ Icᵢⱼₘ[m]

        #     H[m, i, j] = κterm * (dot_term * ΔTvᵢⱼₘ[m] - ΔTcᵢⱼₘ[m] * Iv_norm_sq[m])
        #     a[m, i, j] = κterm * (Iv_norm_sq[m] + dot_term * μa[m])
        #     b[m, i, j] = κterm * (Iv_norm_sq[m] + dot_term * μb[m])
        # end
    end

    # error("done")
    return nothing
end

"""
Find the vertex temperatures using the bi-linear interpolation coefficients
"""
function vertex_temperatures!(
    Tvertex::AbstractArray{T,2}, Tcell::AbstractArray{T,2}, μ::AbstractArray{T,3}, nghost=1
) where {T}
    iterator_range = getcartind(Tvertex, nghost)

    @batch for idx in iterator_range
        i, j = Tuple(idx)
        μᵢⱼ = @view μ[:, i, j]

        Tvertex[i, j] = (
            μᵢⱼ[1] * Tcell[i, j] +
            μᵢⱼ[2] * Tcell[i - 1, j] +
            μᵢⱼ[3] * Tcell[i - 1, j - 1] +
            μᵢⱼ[4] * Tcell[i, j - 1]
        )
    end

    return nothing
end

"""
Find the interpolation coefficients μ, for each node. There are 4 μᵢⱼ per vertex.
"""
function interp_coeff!(
    μ::AbstractArray{T,3},
    κcell::AbstractArray{T,2},
    ξ::AbstractArray{T,2},
    η::AbstractArray{T,2},
    nghost,
) where {T}
    ilohi = axes(κcell, 1)
    jlohi = axes(κcell, 2)
    ilo = first(ilohi) + nghost
    jlo = first(jlohi) + nghost
    ihi = last(ilohi) - nghost+1
    jhi = last(jlohi) - nghost+1

    # This is a vertex-based loop, thus the ihi+1 upper range. Here
    # ihi is the last cell index in the domain
    @batch for j in jlo:jhi
        for i in ilo:ihi

            # Eq. 38
            βᵢⱼ = SVector{4,T}(
                κcell[i, j] * (1 + ξ[i, j]) * (1 + η[i, j]),
                κcell[i - 1, j] * (1 - ξ[i, j]) * (1 + η[i, j]),
                κcell[i - 1, j - 1] * (1 - ξ[i, j]) * (1 - η[i, j]),
                κcell[i, j - 1] * (1 + ξ[i, j]) * (1 - η[i, j]),
            )

            βᵢⱼsum = sum(βᵢⱼ)

            if iszero(βᵢⱼsum)
                μᵢⱼ = @SVector zeros(4)
            else
                μᵢⱼ = βᵢⱼ / βᵢⱼsum
            end

            for q in eachindex(μᵢⱼ)
                μ[q, i, j] = μᵢⱼ[q]
            end
        end
    end

    return nothing
end

function update_Χ_bc!(Χ, nghost)

    # Χ is an cell-based array
    ilohi = axes(Χ, 2)
    jlohi = axes(Χ, 3)
    ilo = first(ilohi) + nghost
    jlo = first(jlohi) + nghost
    ihi = last(ilohi) - nghost
    jhi = last(jlohi) - nghost

    # norm pointing into the domain -- Χ = 1
    # Χij2 for ilo
    # Χij1 for jlo

    # norm pointing out of the domain -- Χ = 0
    # Χij1 for jhi+1
    # Χij2 for ihi+1

    # for all BC types
    for j in jlohi
        Χ[2, ilo, j] = 1
        Χ[2, ihi + 1, j] = 0
    end

    for i in ilohi
        Χ[1, i, jlo] = 1
        Χ[1, i, jhi + 1] = 0
    end

    return nothing
end

function update_Hab_bc!(H, a, b, bcs, nghost)

    # H, a, and b are all cell-based arrays
    ilohi = axes(H, 2)
    jlohi = axes(H, 3)
    ilo = first(ilohi) + nghost
    jlo = first(jlohi) + nghost
    ihi = last(ilohi) - nghost
    jhi = last(jlohi) - nghost

    # for all BC types
    for j in jlohi
        b[2, ilo, j] = 0
        a[2, ihi + 1, j] = 0
    end

    for i in ilohi
        b[1, i, jlo] = 0
        a[1, i, jhi + 1] = 0
    end

    # for flux-specified types

    # ilo BC
    if bcs.ilo === :zeroflux
        for j in jlohi
            H[2, ilo, j] = 0
            a[2, ilo, j] = 0
            b[2, ilo, j] = 0
        end

    elseif bcs.ilo isa Tuple{Symbol,Number}
        if bcs.ilo[1] === :flux
            for j in jlohi
                H[2, ilo, j] = bcs.ilo[2]
                a[2, ilo, j] = 0
                b[2, ilo, j] = 0
            end
        end
    end

    # ihi BC
    if bcs.ihi === :zeroflux
        for j in jlohi
            H[2, ihi + 1, j] = 0
            a[2, ihi + 1, j] = 0
            b[2, ihi + 1, j] = 0
        end

    elseif bcs.ihi isa Tuple{Symbol,Number}
        if bcs.ihi[1] === :flux
            for j in jlohi
                H[2, ihi + 1, j] = bcs.ihi[2]
                a[2, ihi + 1, j] = 0
                b[2, ihi + 1, j] = 0
            end
        end
    end

    # jlo BC
    if bcs.jlo === :zeroflux
        for i in ilohi
            H[1, i, jlo] = 0
            a[1, i, jlo] = 0
            b[1, i, jlo] = 0
        end

    elseif bcs.jlo isa Tuple{Symbol,Number}
        if bcs.jlo[1] === :flux
            for i in ilohi
                H[1, i, jlo] = bcs.jlo[2]
                a[1, i, jlo] = 0
                b[1, i, jlo] = 0
            end
        end
    end

    # jhi BC
    if bcs.jhi === :zeroflux
        for i in ilohi
            H[1, i, jhi + 1] = 0
            a[1, i, jhi + 1] = 0
            b[1, i, jhi + 1] = 0
        end

    elseif bcs.jhi isa Tuple{Symbol,Number}
        if bcs.jhi[1] === :flux
            for i in ilohi
                H[1, i, jhi + 1] = bcs.jhi[2]
                a[1, i, jhi + 1] = 0
                b[1, i, jhi + 1] = 0
            end
        end
    end

    return nothing
end

"""
	update_bilinear_coeff!(mesh, ξ, η)

Calculate all of the (ξ,η) coordinates of the vertices in the mesh. These
can/will be re-used. They only need to be recalculated if the mesh geometry changes
"""
function update_bilinear_coeff!(
    mesh, ξ::AbstractArray{T,N}, η::AbstractArray{T,N}
) where {N,T}
    ilohi = axes(mesh.volume, 1)
    jlohi = axes(mesh.volume, 2)
    ilo = first(ilohi) + mesh.nghost + 1 
    jlo = first(jlohi) + mesh.nghost + 1 
    ihi = last(ilohi) - mesh.nghost
    jhi = last(jlohi) - mesh.nghost

    ϵ = 100eps(T)

    @batch for j in jlo:jhi
        for i in ilo:ihi
            x0 = SVector{2,Float64}(view(mesh.coords, :, i, j))
            x1 = mesh.centroid[i, j]
            x2 = mesh.centroid[i - 1, j]
            x3 = mesh.centroid[i - 1, j - 1]
            x4 = mesh.centroid[i, j - 1]

            # x0 = mesh.centroid[i, j]
            # x1 = SVector{2,Float64}(view(mesh.coords, :, i, j))
            # x2 = SVector{2,Float64}(view(mesh.coords, :, i - 1, j))
            # x3 = SVector{2,Float64}(view(mesh.coords, :, i - 1, j - 1))
            # x4 = SVector{2,Float64}(view(mesh.coords, :, i, j - 1))

            Cx, Cy = get_c.(x0, x1, x2, x3, x4)
            A, B, C = get_quadratic_coeff(Cx, Cy)
            ξ_roots = solve_quadratic(A, B, C)

            # find robust way of doing this
            ξvᵢⱼ = get_valid_root(ξ_roots)
            # ξvᵢⱼ = Inf
            # for q in 1:2
            #     if -1.0 <= ξ_roots[q] <= 1.0
            #         ξvᵢⱼ = ξ_roots[q]
            #         break
            #     end
            # end
            # # if isinf(ξvᵢⱼ) error("unable to find roots!") end

            # # ξv = ξ_roots[-1.0 .<= ξ_roots .<= 1.0]
            # # ξvᵢⱼ = first(ξv)
            # ξvᵢⱼ = ξvᵢⱼ * (abs(ξvᵢⱼ) >= ϵ)

            η_denom = (Cx[3] + Cx[4] * ξvᵢⱼ)
            η_numer = -(Cx[1] + Cx[2] * ξvᵢⱼ)

            η_numer = η_numer * (abs(η_numer) >= ϵ)
            η_denom = η_denom * (abs(η_denom) >= ϵ)

            ηvᵢⱼ = η_numer / η_denom
            ηvᵢⱼ = ηvᵢⱼ * (abs(η_denom) >= ϵ)

            # ηvᵢⱼ = -(Cx[1] + Cx[2] * ξvᵢⱼ) / (Cx[3] + Cx[4] * ξvᵢⱼ)
            # @show ξvᵢⱼ, ηvᵢⱼ, η_numer, η_denom

            ξ[i, j] = ξvᵢⱼ
            η[i, j] = ηvᵢⱼ
        end
    end

    return nothing
end

function get_valid_root(roots)
    ϵ = 100eps(Float64)
    for r in roots
        if -1.0 <= r <= 1.0
            return r * (abs(r) >= ϵ)
        end
    end
    error("unable to find roots!")
end

function get_c(x0, x1, x2, x3, x4)
    C1 = x1 + x2 + x3 + x4 - 4x0
    C2 = x1 - x2 - x3 + x4 # ξ
    C3 = x1 + x2 - x3 - x4 # η
    C4 = x1 - x2 + x3 - x4 # ξη

    ϵ = 1e-14
    # C2 = C2 * (abs(C2) >= ϵ)
    # C3 = C3 * (abs(C3) >= ϵ)
    # C4 = C4 * (abs(C4) >= ϵ)

    return C1, C2, C3, C4
end

function get_quadratic_coeff(Cx, Cy)
    A = Cx[2] * Cy[4] - Cx[4] * Cy[2]
    B = Cy[3] * Cx[2] + Cx[1] * Cy[4] - Cx[4] * Cy[1] - Cy[2] * Cx[3]
    C = Cx[1] * Cy[3] - Cx[3] * Cy[1]
    return A, B, C
end

function diff_of_products(a, b, c, d)
    w = d * c
    e = fma(-d, c, w)
    f = fma(a, b, -w)
    return f + e
end

function solve_quadratic(a, b, c)
    # roots = @MVector [NaN, NaN]
    q = -0.5 * (b + copysign(sqrt(diff_of_products(b, b, 4a, c)), b))
    roots = @SVector [q / a, c / q]
    return roots
end
