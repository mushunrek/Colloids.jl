module AllBalls 

include("helper.jl")

using .Helper
using StaticArrays
using Tullio, LoopVectorization
using Distributions
using JLD2, DelimitedFiles, Formatting
using Plots


export SemiColloidSimulation, animate

const CoordinateList = Vector{SVector{2, Float64}}
const CoordinateMatrix = Matrix{SVector{2, Float64}}

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

    @.colloid_displacement = (
        (sqrtΔt * diffusivity(colloid)) * colloid_noise
        - colloid.potential(colloid_coords)
    )
    @. semicolloid_displacement = (
        (sqrtΔt * diffusivity(semicolloid)) * semicolloid_noise
        - semicolloid.potential(semicolloid_coords)
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
            semicolloid_collision_times::Vector{Float16},
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
        append!(semicolloid_collision_times, zeros(Float64)(diff))
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

        for k in curr:curr+potential_collisions_per_colloid[j]
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
        curr += potential_collisions_per_colloid[j] + 1
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

            for k in curr:curr+potential_collisions_per_colloid[j]
                if semicolloid_collision_times[k] ≤ time_horizon
                    i = colliding_semicolloids[k]

                    collision, modifier = Helper.check_collision(
                        colloid_coords[j] - semicolloid_coords[i],
                        colloid_displacement[j] - semicolloid_displacement[i],
                    )

                    if collision
                        unresolved_collisions = true
                        colloid_displacement[j] -= modifier 
                        semicolloid_displacement += modifier
                    end
                end
            end
            curr += potential_collisions_per_colloid[j] + 1
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
    n = length(coords)
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

            for k in curr:curr+potential_collisions_per_colloid[j]
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
            curr += potential_collisions_per_colloid[j] + 1
        end
    end
end

function step!(
            colloid_coords::CoordinateMatrix,
            semicolloid_coords::CoordinateMatrix,
            colloid_displacement::CoordinateList,
            semicolloid_displacement::CoordinateListm,
            potential_collisions_per_colloid::Vector{Int},
            colliding_semicolloids::Vector{Int},
            colloid::Ball,
            semicolloid::Ball,
            Δt, time_tolerance, t,
            colloid_noise::CoordinateList,
            semicolloid_noise::CoordinateList
        )

        colloid_coords[:, t+1] .= colloid_coords[:, t]
        semicolloid_coords[:, t+1] .= semicolloid_coords[:, t]

        update_displacement!(
            colloid_displacement,
            semicolloid_displacement,
            colloid_coords,
            semicolloid_coords,
            colloid, semicolloid,
            Δt, 
            colloid_noise, semicolloid_noise
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
                colliding_colloids
            )

        next_collision_time = update_collision_times!(
                colloid_collision_times,
                semicolloid_collision_times,
                colloid_coords,
                semicolloid_coords,
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
            @. semicolloid_coords[:, t+1] *= (1- time_fraction)

            handle_collision!(
                    colloid_displacement,
                    semicolloid_displacement,
                    colloid_coords,
                    semicolloid_coords,
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

end