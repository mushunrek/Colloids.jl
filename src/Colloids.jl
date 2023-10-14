module Colloids

using StaticArrays
using Distributions 
using LoopVectorization
using JLD2, DelimitedFiles, Formatting
using Plots

include("points.jl")
include("potentials.jl")
include("particles.jl")
include("core.jl")

using .Points, .Potentials, .Particles
using .ColloidsCore

export Point, PointList
export Potential, Null, Quadratic, DelayedQuadratic
export Particle, Ball, Fluid 
export ColloidsInFluid, ColloidsInSemicolloids
export animate


struct ColloidsInFluid
    coords::PointMatrix
    colloid::Ball
    T::Float64
    Δt::Float64
end

struct ColloidsInSemicolloids
    colloid_coords::PointMatrix
    semicolloid_coords::PointMatrix
    colloid::Ball
    semicolloid::Ball
    T::Float64
    Δt::Float64
end

function ColloidsInFluid(
                    initial::PointList,
                    colloid::Ball,
                    fluid::Fluid,
                    T, Δt, time_tolerance
                )
    T = convert(Float64, T)
    Δt = convert(Float64, Δt)
    time_tolerance = convert(Float64, time_tolerance)

    n = length(initial)
    steps = floor(Int, T / Δt)

    # pre-allocate coordinates for the whole simulation
    coords = zeros(Point, n, steps+1)
    # initialize coordinates at time `t = 0`
    coords[:, 1] = initial

    # pre-allocate containers for displacement and times of potential collisions
    displacement = zeros(Point, n)
    collision_times = zeros(Float64, n*(n-1)÷2)

    # pre-allocate standard normal random variables and scale them
    scaled_noise = (diffusivity(colloid) * √Δt) .* generate_noise(n, steps)

    # pre-compute constants 
    magic_cst1 = 4 * diameter(colloid, fluid)^2
    magic_cst2 = diffusivity(colloid) * fluid_density(fluid) * diameter(colloid, fluid) * Δt
    sq_diam = diameter(colloid)^2

    # simulate
    for t in 1:steps
        @show t
        step!(
            coords, displacement, collision_times, 
            magic_cst1, magic_cst2, sq_diam, 
            Δt, time_tolerance, t, 
            scaled_noise[:, t]
        )
    end

    return ColloidsInFluid(coords, colloid, T, Δt)
end

function ColloidsInSemicolloids(
            colloid_initial::PointList,
            semicolloid_initial::PointList,
            colloid::Ball,
            semicolloid::Ball,
            T, Δt, time_tolerance;
            estimated_max_travel=missing
        )
    T = convert(Float64, T)
    Δt = convert(Float64, Δt)
    time_tolerance = convert(Float64, time_tolerance)

    n = length(colloid_initial)
    m = length(semicolloid_initial)
    steps = floor(Int, T/Δt)

    colloid_coords = zeros(Point, n, steps+1)
    colloid_coords[:, 1] .= colloid_initial
    semicolloid_coords = zeros(Point, m, steps+1)
    semicolloid_coords[:, 1] .= semicolloid_initial

    colloid_displacement = zeros(Point, n)
    semicolloid_displacement = zeros(Point, m)
    
    potential_collisions_per_colloid = zeros(Int, n)
    colliding_semicolloids = zeros(Int, m)

    colloid_collision_times = zeros(Float64, n*(n-1)÷2)
    semicolloid_collision_times = zeros(Float64, m)

    if estimated_max_travel === missing
        # use extreme value theorem to estimate upper bound for the maximum displacement 
        estimated_max_travel = 2*√Δt * (
            diffusivity(colloid) * √(2*log(n))
            + diffusivity(semicolloid) * √(2*log(m))
        )
    end
    estimated_max_dist_sq = (estimated_max_travel + diameter(colloid, semicolloid))^2

    sq_diam = diameter(colloid)^2
    sq_mixed_diam = diameter(colloid, semicolloid)^2

    scaled_colloid_noise = (diffusivity(colloid)*√Δt) .* generate_noise(n, steps)
    scaled_semicolloid_noise = (diffusivity(semicolloid)*√Δt) .* generate_noise(m, steps)

    for t in 1:steps
        @show t
        step!(
            colloid_coords, semicolloid_coords,
            colloid_displacement, semicolloid_displacement,
            potential_collisions_per_colloid, colliding_semicolloids,
            colloid_collision_times, semicolloid_collision_times,
            colloid, semicolloid,
            sq_diam, sq_mixed_diam,
            Δt, time_tolerance, t,
            scaled_colloid_noise[:, t], scaled_semicolloid_noise[:, t],
            estimated_max_dist_sq
        )
    end

    return ColloidsInSemicolloids(colloid_coords, semicolloid_coords, colloid, semicolloid, T, Δt)
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
function animate(sim::ColloidsInFluid, filename; fps=20, skipframes=0)
    anim = @animate for t in 1:size(sim.coords, 2)
        ((t-1) % (skipframes+1) == 0) || continue

        plot(label="t=$t")
        for c in eachrow(sim.coords[:, t])
            plot!(
                circle(
                    c[1], radius(sim.colloid)
                ),
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
    return gif(anim, filename, fps=fps)
end


"""
    animate(sim, filename)

Produces a GIF of the simulation and saves it at the given location.
"""
function animate(sim::ColloidsInSemicolloids, filename; fps=20, skipframes=0, semicolloids=true)
    anim = @animate for t in 1:size(sim.colloid_coords, 2)
        ((t-1) % (skipframes+1) == 0) || continue

        plot(label="t=$t")
        for i in eachindex(sim.colloid_coords[:, t])
            plot!(
                circle(
                    sim.colloid_coords[i, t], radius(sim.colloid)
                ),
                seriestype=:shape,
                fillaplha=0.8,
                label=false,
                aspect_ratio=1,
                xlims=(-5.0, 5.0),
                ylims=(-5.0, 5.0)
            )
        end
        
        if semicolloids
            for i in eachindex(sim.semicolloid_coords[:, t])
                plot!(
                    circle(
                        sim.semicolloid_coords[i, t], radius(sim.semicolloid)
                    ),
                    seriestype=:shape,
                )
            end
        end
        
        plot!()
    end
    return gif(anim, filename, fps=fps)
end

end
