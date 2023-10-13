module AllBalls 

include("helper.jl")

using .Helper
using StaticArrays
using Tullio, LoopVectorization
using Distributions
using JLD2, DelimitedFiles, Formatting
using Plots


export MixtureSimulation, animate
export no_potential, quadratic, delayed_quadratic
export Particle, Ball, Fluid 
export radius, d, diffusivity, density 
export no_potential, quadratic, delayed_quadratic

const CoordinateList = Vector{SVector{2, Float64}}
const CoordinateMatrix = Matrix{SVector{2, Float64}}

abstract type Particle end

struct Ball <: Particle
    R::Float64
    Σ::Float64
    potential
end

Ball(R, Σ; potential=(x -> zeros(SVector{2, Float64}))) = Ball(R, Σ, potential)

struct Fluid <: Particle
    R::Float64
    density::Float64
end

@inline radius(p::Particle) = p.R
@inline d(p1::Particle, p2::Particle) = radius(p1) + radius(p2)
@inline diffusivity(b::Ball) = b.Σ
@inline density(f::Fluid) = f.density


no_potential(x) = zeros(SVector{2, Float64})
quadratic(x; strength) = strength*x
delayed_quadratic(x; delay, strength) = (
    Helper.sq_norm(x) > delay ?
    strength * (1 - delay/Helper.sq_norm(x)) .* x :
    zeros(SVector{2, Float64})
)

function update_displacement!(
            colloid_displacement::CoordinateList,
            semicolloid_displacement::CoordinateList,
            colloid_coords::CoordinateList,
            semicolloid_coords::CoordinateList,
            colloid::Ball,
            semicolloid::Ball,
            Δt,
            colloid_noise::CoordinateList, 
            semicolloid_noise::CoordinateList
        )
    sqrtΔt = √Δt

    colloid_displacement .= (
        (sqrtΔt * diffusivity(colloid)) .* colloid_noise
        .- colloid.potential.(colloid_coords)
    )
    semicolloid_displacement .= (
        (sqrtΔt * diffusivity(semicolloid)) .* semicolloid_noise
        .- semicolloid.potential.(semicolloid_coords)
    )
    nothing
end

function update_potential_collisions!(
            potential_collisions_per_colloids::Vector{Int},
            colliding_semicolloids::Vector{Int},
            colloid_coords::CoordinateList,
            semicolloid_coords::CoordinateList,
            max_dist_sq
        )
    k = length(colliding_semicolloids)
    curr = 1
    for i in eachindex(colloid_coords)
        potential_collisions_per_colloids[i] = 0
        for j in eachindex(semicolloid_coords)
            rel_pos = colloid_coords[i] - semicolloid_coords[j]
            rel_dist_sq = Helper.sq_norm(rel_pos)
            if rel_dist_sq < max_dist_sq
                potential_collisions_per_colloids[i] += 1
                if k < curr
                    append!(colliding_semicolloids, j)
                else
                    colliding_semicolloids[curr] = j
                end
                curr += 1
            end
        end
    end
    nothing
end

function update_collision_times!(
            colloid_collision_times::Vector{Float64},
            semicolloid_collision_times::Vector{Float64},
            colloid_coords::CoordinateList,
            semicolloid_coords::CoordinateList,
            colloid_displacement::CoordinateList,
            semicolloid_displacement::CoordinateList,
            potential_collisions_per_colloid::Vector{Int},
            colliding_semicolloids::Vector{Int},
            colloid::Ball,
            semicolloid::Ball,
            remaining_time
        )
    dll = 2*radius(colloid)
    dls = d(colloid, semicolloid)
    next_collision_time = Inf 

    diff = length(colliding_semicolloids) - length(semicolloid_collision_times)
    if diff > 0
        append!(semicolloid_collision_times, zeros(Float64, diff))
    end

    n = length(colloid_coords)
    curr = 1
    for j in 1:n
        for i in j+1:n
            q = triangular_index(i, j, n)

            colloid_collision_times[q] = Helper.collision_time(
                colloid_coords[i] - colloid_coords[j],
                colloid_displacement[i] - colloid_displacement[j],
                dll, remaining_time
            )

            if colloid_collision_times[q] < next_collision_time
                next_collision_time = colloid_collision_times[q]
            end
        end

        for k in curr:curr+potential_collisions_per_colloid[j]-1
            i = colliding_semicolloids[k]
            semicolloid_collision_times[k] = Helper.collision_time(
                semicolloid_coords[i] - colloid_coords[j],
                semicolloid_displacement[i] - colloid_displacement[j],
                dls, remaining_time
            )

            if semicolloid_collision_times[k] < next_collision_time
                next_collision_time = semicolloid_collision_times[k]
            end
        end
        curr += potential_collisions_per_colloid[j]
    end
    return next_collision_time
end

function handle_collision!(
            colloid_displacement::CoordinateList,
            semicolloid_displacement::CoordinateList,
            colloid_coords::CoordinateList,
            semicolloid_coords::CoordinateList,
            colloid_collision_times::Vector{Float64},
            potential_collisions_per_colloid::Vector{Int},
            colliding_semicolloids::Vector{Int},
            semicolloid_collision_times::Vector{Float64},
            time_horizon
        )
    n = length(colloid_coords)

    unresolved_collisions = true
    while unresolved_collisions
        unresolved_collisions = false
        curr = 1
        for j in 1:n
            for i in j+1:n 
                q = triangular_index(i, j, n)

                if colloid_collision_times[q] ≤ time_horizon
                    collision, modifier = Helper.check_collision(
                        colloid_coords[i] - colloid_coords[j],
                        colloid_displacement[i] - colloid_displacement[j]
                    )

                    if collision 
                        unresolved_collisions = true
                        colloid_displacement[i] -= modifier 
                        colloid_displacement[j] += modifier
                    end
                end
            end

            for k in curr:curr+potential_collisions_per_colloid[j]-1
                if semicolloid_collision_times[k] ≤ time_horizon
                    i = colliding_semicolloids[k]

                    collision, modifier = Helper.check_collision(
                        colloid_coords[j] - semicolloid_coords[i],
                        colloid_displacement[j] - semicolloid_displacement[i],
                    )

                    if collision
                        unresolved_collisions = true
                        colloid_displacement[j] -= modifier 
                        semicolloid_displacement[i] += modifier
                    end
                end
            end
            curr += potential_collisions_per_colloid[j]
        end
    end
    nothing
end

function resolve_overlaps!(
            colloid_displacement::CoordinateList,
            semicolloid_displacement::CoordinateList,
            colloid_coords::CoordinateList,
            semicolloid_coords::CoordinateList,
            colloid::Ball,
            semicolloid::Ball,
            potential_collisions_per_colloid::Vector{Int},
            colliding_colloids::Vector{Int}
        )
    n = length(colloid_coords)
    dll = 2*radius(colloid)
    dls = d(colloid, semicolloid)

    unresolved_overlaps = true
    while unresolved_overlaps
        unresolved_overlaps = false
        curr = 1
        for j in 1:n
            for i in j+1:n
                overlap, modifier = Helper.check_overlap(
                    colloid_coords[i] - colloid_coords[j],
                    colloid_displacement[i] - colloid_displacement[j],
                    dll
                )

                if overlap
                    unresolved_overlaps = true
                    colloid_displacement[i] -= modifier
                    colloid_displacement[j] += modifier 
                end
            end

            for k in curr:curr+potential_collisions_per_colloid[j]-1
                i = colliding_colloids[k]

                overlap, modifier = Helper.check_overlap(
                    colloid_coords[j] - semicolloid_coords[i],
                    colloid_displacement[j] - semicolloid_displacement[i],
                    dls
                )

                if overlap
                    unresolved_overlaps = true
                    colloid_displacement[j] -= modifier
                    semicolloid_displacement[i] += modifier
                end
            end
            curr += potential_collisions_per_colloid[j]
        end
    end
end

function step!(
            colloid_coords::CoordinateMatrix,
            semicolloid_coords::CoordinateMatrix,
            colloid_displacement::CoordinateList,
            semicolloid_displacement::CoordinateList,
            potential_collisions_per_colloid::Vector{Int},
            colliding_semicolloids::Vector{Int},
            colloid_collision_times::Vector{Float64},
            semicolloid_collision_times::Vector{Float64},
            colloid::Ball,
            semicolloid::Ball,
            Δt, time_tolerance, t,
            colloid_noise::CoordinateList,
            semicolloid_noise::CoordinateList,
            estimated_max_dist_sq
        )

    @. colloid_coords[:, t+1] = colloid_coords[:, t]
    @. semicolloid_coords[:, t+1] = semicolloid_coords[:, t]

    update_displacement!(
            colloid_displacement,
            semicolloid_displacement,
            colloid_coords[:, t+1],
            semicolloid_coords[:, t+1],
            colloid, semicolloid,
            Δt, 
            colloid_noise, semicolloid_noise
        )

    update_potential_collisions!(
        potential_collisions_per_colloid,
        colliding_semicolloids,
        colloid_coords[:, t+1],
        semicolloid_coords[:, t+1],
        estimated_max_dist_sq
    )

    remaining_time = Δt
    while remaining_time > 0.0
        resolve_overlaps!(
                colloid_displacement,
                semicolloid_displacement,
                colloid_coords[:, t+1],
                semicolloid_coords[:, t+1],
                colloid, semicolloid,
                potential_collisions_per_colloid,
                colliding_semicolloids
            )

        next_collision_time = update_collision_times!(
                colloid_collision_times,
                semicolloid_collision_times,
                colloid_coords[:, t+1],
                semicolloid_coords[:, t+1],
                colloid_displacement,
                semicolloid_displacement,
                potential_collisions_per_colloid,
                colliding_semicolloids,
                colloid, semicolloid,
                remaining_time
            )

        if next_collision_time ≥ remaining_time
            colloid_coords[:, t+1] .+= colloid_displacement
            semicolloid_coords[:, t+1] .+= semicolloid_displacement
            remaining_time = 0.0
        else
            time_horizon = min(
                    max(
                        next_collision_time, time_tolerance
                    ),
                    remaining_time
                )

            time_fraction = time_horizon / remaining_time
            @. colloid_coords[:, t+1] += time_fraction  * colloid_displacement
            @. colloid_displacement *= (1 - time_fraction)
            @. semicolloid_coords[:, t+1] += time_fraction * semicolloid_displacement
            @. semicolloid_displacement *= (1 - time_fraction) 

            handle_collision!(
                    colloid_displacement,
                    semicolloid_displacement,
                    colloid_coords[:, t+1],
                    semicolloid_coords[:, t+1],
                    colloid_collision_times,
                    potential_collisions_per_colloid,
                    colliding_semicolloids,
                    semicolloid_collision_times,
                    time_horizon
                )

            remaining_time -= time_horizon
        end
    end
    nothing
end


struct MixtureSimulation
    colloid_coords::CoordinateMatrix
    semicolloid_coords::CoordinateMatrix
    colloid::Ball
    semicolloid::Ball 
    T::Float64
    Δt::Float64
end

function MixtureSimulation(
            colloid_initial::CoordinateList,
            semicolloid_initial::CoordinateList,
            colloid::Ball,
            semicolloid::Ball,
            T, Δt, time_tolerance
        )
    n = length(colloid_initial)
    m = length(semicolloid_initial)
    steps = floor(Int, T/Δt)

    colloid_coords = zeros(SVector{2, Float64}, n, steps+1)
    colloid_coords[:, 1] = colloid_initial
    semicolloid_coords = zeros(SVector{2, Float64}, m, steps+1)
    semicolloid_coords[:, 1] = semicolloid_initial

    colloid_displacement = zeros(SVector{2, Float64}, n)
    semicolloid_displacement = zeros(SVector{2, Float64}, m)

    potential_collisions_per_colloid = zeros(Int, n)
    colliding_semicolloids = zeros(Int, m)

    colloid_collision_times = zeros(Float64, n*(n-1)÷2)
    semicolloid_collision_times = zeros(Float64, m)

    estimated_max_travel = 7√Δt * (diffusivity(colloid) + diffusivity(semicolloid))
    estimated_max_dist_sq = (estimated_max_travel + d(colloid, semicolloid))^2

    colloid_noise = generate_noise(n, steps)
    semicolloid_noise = generate_noise(m, steps)

    for t in 1:steps
        step!(
            colloid_coords, semicolloid_coords,
            colloid_displacement, semicolloid_displacement,
            potential_collisions_per_colloid, colliding_semicolloids,
            colloid_collision_times, semicolloid_collision_times,
            colloid, semicolloid,
            Δt, time_tolerance, t,
            colloid_noise[:, t],
            semicolloid_noise[:, t],
            estimated_max_dist_sq
        )
    end

    return MixtureSimulation(colloid_coords, semicolloid_coords, colloid, semicolloid, T, Δt)
end

function circle(c, r)
    θ = LinRange(0.0, 2*π, 100)
    c[1] .+ r*cos.(θ), c[2] .+ r*sin.(θ)
end

"""
    animate(sim, filename)

Produces a GIF of the simulation and saves it at the given location.
"""
function animate(sim::MixtureSimulation, filename; fps=20)
    anim = @animate for t in 1:size(sim.colloid_coords, 2)
        plot(label="t=$t")
        for i in eachindex(sim.colloid_coords[:, t])
            plot!(
                circle(sim.colloid_coords[i, t], radius(sim.colloid)),
                seriestype=:shape,
                fillaplha=0.8,
                label=false,
                aspect_ratio=1,
                xlims=(-10.0, 20.0),
                ylims=(-10.0, 20.0)
            )
        end
        for i in eachindex(sim.semicolloid_coords[:, t])
            plot!(circle(sim.semicolloid_coords[i, t], radius(sim.semicolloid)),
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

end