module SSIDiffusion

include("error.jl")

export Mesh2D
include("mesh.jl")

include("boundary_conditions.jl")
export applybc!

export SSISolver2D
export advance_solution_single_step!, advance_solution_time_range!
include("solver_2d.jl")

end
