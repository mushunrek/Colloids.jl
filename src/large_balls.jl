module LargeBalls

include("helper.jl")

using .Helper
using StaticArrays
using Tullio, LoopVectorization
using Distributions
using JLD2, DelimitedFiles, Formatting
using Plots

export ColloidSimulation, animate

normal = Normal()

"""
    update_displacement!(displacement, coords, Σ, rl, rs, zs, Δt, noise)

Computes the new displacement in place.

# Arguments
- `displacement::Vector{SVector{2, Float64}}`: pre-allocated container
- `coords::Vector{SVector{2, Float64}}`: coordinates of balls 
- `Σ::Float64`: diffusion coefficient of balls 
- `rl::Float64`: radius of balls 
- `rs::Float64`: radius of fluid particles
- `zs::Float64`: density of fluid particles 
- `Δt::Float64`: time step
- `noise::Vector{SVector{2, Float64}}`: pre-allocated standard normal random variables
"""
function update_displacement!(
        displacement::Vector{SVector{2, Float64}},
        coords::Vector{SVector{2, Float64}},
        Σ, rl,
        rs, zs, 
        Δt,
        noise::Vector{SVector{2, Float64}} 
    )
    # declare constants for performance
    dls = rl + rs
    c1 = 4 * dls^2
    c2 = (Σ*zs*dls*Δt)
    # use Einstein summation
    @tullio displacement[i] = (
                                (1 < Helper.sq_norm(coords[j] - coords[i]) < c1) ?
                                    (
                                        .√(1 .- (coords[j] - coords[i]).^2 ./ c1) 
                                        .* (coords[j] - coords[i]) 
                                        ./ √Helper.sq_norm(coords[j] - coords[i])
                                    ) .* c2 :
                                    @SVector [0.0, 0.0]
                            )
    displacement .= (Σ*√Δt) .* noise .- displacement
    return nothing
end


"""
    update_collision_times!(collision_times, coords, displacement, rl, remaining_time)

Computes the new collision times in place and returns the next collision time.

# Arguments
- `collision_times::Vector{Float64}`: Pre-allocated container to store collision times.
    To reduce memory allocation, we index the lower triangular part of a `n×n` matrix 
    linearly. See also `?triangular_index`.
- `coords::Vector{SVector{2, Float64}}`: coordinates of balls 
- `displacement::Vector{SVector{2, Float64}}`: displacement of balls 
- `rl::Float64`: radius of balls 
- `remaining_time::Float64`: threshold after which collisions are ignored
"""
function update_collision_times!(
                collision_times::Vector{Float64},
                coords::Vector{SVector{2, Float64}},
                displacement::Vector{SVector{2, Float64}},
                rl, remaining_time
            )
    n = length(coords)
    d = 2*rl
    next_collision_time = Inf
    for j in 1:n-1
        for i in j+1:n
            # get triu index
            q = triangular_index(i, j, n)

            # compute collision time between balls `i` and `j`
            @inbounds collision_times[q] = Helper.collision_time(
                                            coords[i] - coords[j],
                                            displacement[i] - displacement[j],
                                            d^2, remaining_time
                                        )
            # update `next_collision_time` if necessary
            @inbounds if collision_times[q] < next_collision_time
                @inbounds next_collision_time = collision_times[q]
            end
        end
    end
    return next_collision_time
end


"""
    handle_collisions!(displacement, coords, collision_times, end_of_collision)

Updates the `displacement` if collisions occur.

# Arguments
- `displacement::Vector{SVector{2, Float64}}`: displacement of balls (to be updated in place)
- `coords::Vector{SVector{2, Float64}}`: coordinates of balls 
- `collision_times::Vector{Float64}`: times of potential collisions, see also `?update_collision_times!`
- `end_of_collision::Float64`: time horizon up to which collisions are handeled
"""
function handle_collisions!(
                displacement::Vector{SVector{2, Float64}},
                coords::Vector{SVector{2, Float64}},
                collision_times::Vector{Float64},
                end_of_collision
            )
    n = length(coords)

    unresolved_collisions = true
    while unresolved_collisions
        unresolved_collisions = false
        for j in 1:n-1
            for i in j+1:n
                # get triu index
                q = triangular_index(i, j, n)

                # determine whether potential collision occurs between balls 
                # `i` and `j` before time horizon `end_of_collision`
                if collision_times[q] ≤ end_of_collision
                    # check whether collision occurs
                    collision, modifier = Helper.check_collision(
                                                            coords[i] - coords[j],
                                                            displacement[i] - displacement[j]
                                                        )
                    # update `displacement` if necessary
                    if collision
                        unresolved_collisions = true
                        displacement[i] -= modifier
                        displacement[j] += modifier
                    end
                end
            end
        end
    end
end

"""
    resolve_overlaps!(displacement, coords, rl)

Resolves overlaps that occur due to imprecisions.

# Arguments
- `displacement::Vector{SVector{2, Float64}}`: displacement of balls (to be updated in place)
- `coords::Vector{SVector{2, Float64}}`: coordinates of balls 
- `rl::Float64`: radius of balls 
"""
function resolve_overlaps!(
                displacement::Vector{SVector{2, Float64}},
                coords::Vector{SVector{2, Float64}},
                rl
            )
    n = length(coords)
    d = 2*rl

    unresolved_overlaps = true
    while unresolved_overlaps
        unresolved_overlaps = false
        for j in 1:n-1
            for i in j+1:n
                # check whether balls `i` and `j` overlap
                collision, modifier = Helper.check_overlap(
                                                coords[i] - coords[j],
                                                displacement[i] - displacement[j],
                                                d^2
                                            )
                # update displacement if necessary
                if collision
                    unresolved_overlaps = true
                    displacement[i] -= modifier
                    displacement[j] += modifier
                end
            end
        end
    end
end

"""
    step!(coords, displacement, collision_times, Σ, rl, rs, zs, t, Δt, time_tolerance, noise)

Computes the `t`-th step of the simulation.

# Arguments
- `coords::Matrix{SVector{2, Float64}}`: Pre-allocated container for the coordinates.
    The entries `1:t` contain the coordinates computed in previous steps. (Index `1`
    is the initial condition.)
- `displacement::Vector{SVector{2, Float64}}`: Pre-allocated container for the 
    displacement of balls.
- `collision_times::Vector{Float64}`: Pre-allocated container for the times of 
    potential collisions.
- `Σ::Float64`: Diffusion coefficient of balls 
- `rl::Float64`: Radius of balls 
- `rs::Float64`: Radius of fluid particles 
- `zs::Float64`: Density of fluid particles 
- `Δt::Float64`: Time step 
- `time_tolerance::Float64`: Temporal resolution of collisions
- `t::Int`: Current time index
- `noise::Matrix{SVector{2, Float64}}`: Pre-allocated standard normal random variables

# Expected Dimensions
Consider `n` balls and a total of `steps` time steps for the simulation. It is 
expected that 
    size(coords) = (n, steps+1)
    size(displacement) = (n,)
    size(collision_times) = (n*(n-1)/2,)
    size(noise) = (n,) 
"""
function step!(
            coords::Matrix{SVector{2, Float64}},
            displacement::Vector{SVector{2, Float64}},
            collision_times::Vector{Float64},
            Σ, rl, rs, zs,
            Δt, time_tolerance, t,
            noise::Vector{SVector{2, Float64}}
        )
    
    # initialize coordinates with old ones
    coords[:, t+1] .= coords[:, t]
    # compute displacement 
    update_displacement!(displacement, coords[:, t+1], Σ, rl, rs, zs, Δt, noise)

    # compute movement with finer resolution given by `time_tolerance`
    remaining_time = Δt
    while remaining_time > 0.0
        # resolve overlaps due to imprecisions
        resolve_overlaps!(displacement, coords[:, t+1], rl)
        # compute times of potential collisions
        next_collision_time = update_collision_times!(collision_times, coords[:, t+1], displacement, rl, remaining_time)

        # move balls
        if next_collision_time ≥ remaining_time
            # if no collision occurs in the remaining time, use full displacement
            coords[:, t+1] .+= displacement 
            remaining_time = 0.0
        else
            # otherwise handle collisions:

            # consider collisions up to remaining time with resolution given by 
            # `time_tolerance`
            end_of_collision = min(
                max(next_collision_time, time_tolerance),
                remaining_time
            )
            # compute fraction of displacement needed
            time_fraction = end_of_collision / remaining_time
            # update coordaintes and displacement accordingly
            coords[:, t+1] .+= time_fraction .* displacement 
            displacement .*= (1 - time_fraction)

            # handle collisions 
            handle_collisions!(displacement, coords[:, t+1], collision_times, end_of_collision)

            remaining_time -= end_of_collision
        end
    end
end

"""
    ColloidSimulation(coords, Σ, rl, rs, zs, T, Δt)

Container for a simulation with given parameters.
"""
struct ColloidSimulation 
    coords::Matrix{SVector{2, Float64}}
    Σ::Float64
    R::Float64
    rs::Float64
    zs::Float64
    T::Float64
    Δt::Float64
end

"""
    ColloidSimulation(initial_configuration, Σ, R, rs, zs, T, Δt, time_tolerance)

Simulates the evolution of balls in a fluid starting from `initial_configuration`.

# Arguments
- `initial_configuration::Vector{SVector{2, Float64}}`: coordinates of balls at time `t=0`
- `Σ`: diffusion coefficient of balls 
- `R`: radius of balls 
- `rs`: radius of fluid particles 
- `zs`: density of fluid particles 
- `T`: time horizon for the simulation 
- `Δt`: time step for the simulation 
- `time_tolerance`: time resolution for collision handling

# Example

```julia
using Colloids 
using StaticArrays
using Plots

# initialize five balls
initial = copy(
    reinterpret(
        SVector{2, Float64},
        [ 1.42*i for i in 1:10 ]
    )
)
Σ = 1.0
R = 1.0
rs = 0.2
zs = 10.0
T = 1.0
Δt = 0.01
time_tolerance = 1e-5

sim = ColloidSimulation(initial, Σ, R, rs, zs, T, Δt, time_tolerance)
animate(sim, "./animation.gif")
```
"""
function ColloidSimulation(
                    initial_configuration::Vector{SVector{2, Float64}},
                    Σ, R, rs, zs,
                    T, Δt, time_tolerance
                )
    n = length(initial_configuration)
    steps = floor(Int, T / Δt)

    # pre-allocate coordinates for the whole simulation
    coords = zeros(SVector{2, Float64}, n, steps+1)
    # initialize coordinates at time `t = 0`
    coords[:, 1] = initial_configuration

    # pre-allocate containers for displacement and times of potential collisions
    displacement = zeros(SVector{2, Float64}, n)
    collision_times = zeros(Float64, n*(n-1)÷2)

    # pre-allocate standard normal random variables
    noise = generate_noise(n, steps)

    # simulate
    for t in 1:steps
        step!(coords, displacement, collision_times, Σ, R, rs, zs, Δt, time_tolerance, t, noise[:, t])
    end

    return ColloidSimulation(coords, Σ, R, rs, zs, T, Δt)
end


"""
    generate_foldername(sim, path)

Generates a suitable name for the folder where the data will be saved.
"""
function generate_foldername(sim::ColloidSimulation, path="./pov/")
    n = length(sim.coords[:, 1])
    "$(path)colloids__n$(n)R$(sim.R)Sigma$(sim.Σ)rs$(sim.rs)zs$(sim.zs)T$(sim.T)dt$(sim.Δt)/"
end

"""
    save_colloids(sim, filename)

Saves the simulation into given file.
"""
function save_colloids(sim::ColloidSimulation, filename)
    save_object(filename, sim)
end

"""
    load_colloids(filename)

Loads a simulation from a given filename
"""
function load_colloids(filename)
    load_object(filename)
end

"""
    save_colloids_to_pov(sim, foldername)

Saves the coordinates of the balls in a format suitable for visualisation with 
the software PovRay.
"""
function save_colloids_to_pov(sim::ColloidSimulation, foldername)
    mkpath(foldername)
    cd(foldername)
    n = length(sim.coords[:, 1])

    for t in 1:size(sim.coords, 2)
        timestamp = fmt("2.10f", t*sim.Δt)
        curr_file = "boules$(timestamp)"
        current_coords = reshape(
            copy(
                reinterpret(
                    Float64,
                    sim.coords[:, t]
                )
            ),
            n, 2
        )
        writedlm(curr_file, current_coords, ",")
    end
end


"""
    circle(c, r)

Returns the circle of center `c` and radius `r`. Used for plotting only.
"""
function circle(c, r)
    θ = LinRange(0.0, 2*π, 100)
    c[1] .+ r*cos.(θ), c[2] .+ r*sin.(θ)
end

"""
    animate(sim, filename)

Produces a GIF of the simulation and saves it at the given location.
"""
function animate(sim::ColloidSimulation, filename)
    anim = @animate for t in 1:size(sim.coords, 2)
        plot(label="t=$t")
        for c in eachrow(sim.coords[:, t])
            plot!(
                circle(c[1], sim.R),
                seriestype=:shape,
                fillaplha=0.2,
                label=false,
                aspect_ratio=1,
                xlims=(-10.0, 20.0),
                ylims=(-10.0, 20.0)
            )
        end
        plot!()
    end
    return gif(anim, filename)
end

end