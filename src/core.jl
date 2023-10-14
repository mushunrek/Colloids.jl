module ColloidsCore

using Distributions
using Tullio, LoopVectorization
using ..Points, ..Potentials, ..Particles

export generate_noise, step!

normal = Normal()

function generate_noise(n, m)
    reshape(
        copy(
            reinterpret(
                Point,
                rand(normal, 2*n*m)
            )
        ),
        (n, m)
    )
end


"""
    triangular_index(i, j, n)

Return the index of an element at position `(i,j)`in a lower triangular `n×n` matrix 
when elements are indexed columnwise from left to right.
"""
@inline triangular_index(i, j, n) = @fastmath (j-1)*(2*n - j - 2) ÷ 2 + (i - j)


@inline modifier(rel_pos::Point, rel_prod, rel_dist_sq) = @fastmath (rel_prod / rel_dist_sq) .* rel_pos
@inline modifier(rel_pos::Point, rel_prod) = modifier(rel_pos, rel_prod, sq_norm(rel_pos))


"""
    resolve_overlaps(relative_position, relative_displacement, sq_diam)

Determines whether two balls collide and compute the resulting change in displacement.
`sq_diam` is the squared sum of the radii.
"""
@inline function check_overlap(
        relative_position::Point,
        relative_displacement::Point,
        sq_diam
    )
    relative_product = dot(relative_position, relative_displacement)
    relative_distance_squared = sq_norm(relative_position)
    if ( relative_distance_squared ≤ sq_diam ) && ( relative_product < 0.0 )
        return true, modifier(relative_position, relative_product, relative_distance_squared)
    end
    return false, zeros(Point)
end

@inline function collision_time(
        relative_position::Point,
        relative_displacement::Point,
        sq_diam, remaining_time
    )
    relative_distance_squared = sq_norm(relative_position)

    if relative_distance_squared > sq_diam
        relative_product = dot(relative_position, relative_displacement)
        relative_displacement_squared = sq_norm(relative_displacement)
        @fastmath discriminant = relative_product^2 - relative_displacement_squared * (relative_distance_squared - sq_diam)
        if ( relative_product < 0.0 ) && ( discriminant ≥ 0.0 )
            return @fastmath -remaining_time * ( relative_product + √discriminant ) / relative_displacement_squared
        end
    end
    return Inf
end

@inline function check_collision(
        relative_position::Point,
        relative_displacement::Point
    )
    relative_product = dot(relative_position, relative_displacement)
    if relative_product < 0.0
        return true, modifier(relative_position, relative_product)
    end
    return false, zeros(Point)
end


# implementation of the functions for ColloidsInFluid

"""
    update_displacement!(displacement, coords, Σ, rl, rs, zs, Δt, noise)

Computes the new displacement in place.

# Arguments
- `displacement::PointList`: pre-allocated container
- `coords::PointList`: coordinates of balls 
- `magic_cst1`: equals `4*dls^2`, where `dls = diameter(colloid, fluid)`
- `magic_cst2`: equals `Σ*density(fluid)*dls*Δt`
- `scaled_noise::PointList`: pre-allocated centered normal random variables with
    standard deviation given by `Σ*√Δt`
"""
function update_displacement!(
        displacement::PointList,
        coords::PointList,
        magic_cst1, magic_cst2,
        scaled_noise::PointList 
    )
    # use Einstein summation
    @tullio displacement[i] = (
                                (1 < sq_norm(coords[j] - coords[i]) < magic_cst1) ?
                                    (
                                        .√(1 .- (coords[j] - coords[i]).^2 ./ magic_cst1) 
                                        .* (coords[j] - coords[i]) 
                                        ./ √sq_norm(coords[j] - coords[i])
                                    ) .* magic_cst2 :
                                    zeros(Point)
                            )
    @. displacement = scaled_noise - displacement
    return nothing
end


"""
    resolve_overlaps!(displacement, coords, sq_diam)

Resolves overlaps that occur due to imprecisions.

# Arguments
- `displacement::PointList`: displacement of balls (to be updated in place)
- `coords::PointList`: coordinates of balls 
- `sq_diam::Float64`: squared diameter of balls 
"""
function resolve_overlaps!(
                displacement::PointList,
                coords::PointList,
                sq_diam
            )
    n = length(coords)

    unresolved_overlaps = true
    while unresolved_overlaps
        unresolved_overlaps = false
        @inbounds @simd for j in 1:n-1
            for i in j+1:n
                # check whether balls `i` and `j` overlap
                collision, modifier = check_overlap(
                                                coords[i] - coords[j],
                                                displacement[i] - displacement[j],
                                                sq_diam
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
    update_collision_times!(collision_times, coords, displacement, sq_diam, remaining_time)

Computes the new collision times in place and returns the next collision time.

# Arguments
- `collision_times::Vector{Float64}`: Pre-allocated container to store collision times.
    To reduce memory allocation, we index the lower triangular part of a `n×n` matrix 
    linearly. See also `?triangular_index`.
- `coords::PointList`: coordinates of balls 
- `displacement::PointList`: displacement of balls 
- `sq_diam::Float64`: squared diameter of balls
- `remaining_time::Float64`: threshold after which collisions are ignored
"""
function update_collision_times!(
                collision_times::Vector{Float64},
                coords::PointList,
                displacement::PointList,
                sq_diam, remaining_time
            )
    n = length(coords)
    
    next_collision_time = Inf
    @inbounds @simd for j in 1:n-1
        for i in j+1:n
            # get triu index
            q = triangular_index(i, j, n)

            # compute collision time between balls `i` and `j`
            collision_times[q] = collision_time(
                                            coords[i] - coords[j],
                                            displacement[i] - displacement[j],
                                            sq_diam, remaining_time
                                        )
            # update `next_collision_time` if necessary
            if collision_times[q] < next_collision_time
                next_collision_time = collision_times[q]
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
                displacement::PointList,
                coords::PointList,
                collision_times::Vector{Float64},
                time_horizon
            )
    n = length(coords)

    unresolved_collisions = true
    while unresolved_collisions
        unresolved_collisions = false
        @inbounds @simd for j in 1:n-1
            for i in j+1:n
                # get triu index
                q = triangular_index(i, j, n)

                # determine whether potential collision occurs between balls 
                # `i` and `j` before `time_horizon`
                if collision_times[q] ≤ time_horizon
                    # check whether collision occurs
                    collision, modifier = check_collision(
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
    step!(coords, displacement, collision_times, Σ, rl, rs, zs, t, Δt, time_tolerance, noise)

Computes the `t`-th step of the simulation.

# Arguments
- `coords::PointMatrix`: Pre-allocated container for the coordinates.
    The entries `1:t` contain the coordinates computed in previous steps. (Index `1`
    is the initial condition.)
- `displacement::PointList`: Pre-allocated container for the 
    displacement of balls.
- `collision_times::Vector{Float64}`: Pre-allocated container for the times of 
    potential collisions.
- `magic_cst1`: magic constant, see `?update_displacement!`
- `magic_cst2`: idem
- `Δt::Float64`: Time step 
- `time_tolerance::Float64`: Temporal resolution of collisions
- `t::Int`: Current time index
- `scaled_noise::PointList`: Pre-allocated centered normal random variables with 
    standard deviation `Σ*√Δt`

# Expected Dimensions
Consider `n` balls and a total of `steps` time steps for the simulation. It is 
expected that 
    size(coords) = (n, steps+1)
    size(displacement) = (n,)
    size(collision_times) = (n*(n-1)/2,)
    size(noise) = (n,) 
"""
function step!(
            coords::PointMatrix,
            displacement::PointList,
            collision_times::Vector{Float64},
            magic_cst1, magic_cst2,
            sq_diam,
            Δt, time_tolerance, t,
            scaled_noise::PointList
        )
    
    # initialize coordinates with old ones
    coords[:, t+1] .= coords[:, t]
    # compute displacement 
    update_displacement!(displacement, coords[:, t+1], magic_cst1, magic_cst2, scaled_noise)

    # compute movement with finer resolution given by `time_tolerance`
    remaining_time = Δt
    while remaining_time > 0.0
        # resolve overlaps due to imprecisions
        resolve_overlaps!(displacement, coords[:, t+1], sq_diam)
        # compute times of potential collisions
        next_collision_time = update_collision_times!(collision_times, coords[:, t+1], displacement, sq_diam, remaining_time)

        # move balls
        if next_collision_time ≥ remaining_time
            # if no collision occurs in the remaining time, use full displacement
            coords[:, t+1] .+= displacement 
            remaining_time = 0.0
        else
            # otherwise handle collisions:

            # consider collisions up to remaining time with resolution given by 
            # `time_tolerance`
            time_horizon = min(
                max(next_collision_time, time_tolerance),
                remaining_time
            )
            # compute fraction of displacement needed
            time_fraction = time_horizon / remaining_time
            # update coordaintes and displacement accordingly
            @. coords[:, t+1] += time_fraction * displacement 
            displacement .*= (1 - time_fraction)

            # handle collisions 
            handle_collisions!(displacement, coords[:, t+1], collision_times, time_horizon)

            remaining_time -= time_horizon
        end
    end
end


# implementation of the functions for ColloidsInSemicolloids

@inline function update_displacement!(
            colloid_displacement::PointList,
            semicolloid_displacement::PointList,
            colloid_coords::PointList,
            semicolloid_coords::PointList,
            colloid::Ball,
            semicolloid::Ball,
            scaled_colloid_noise::PointList, 
            scaled_semicolloid_noise::PointList
        )
    @. colloid_displacement = scaled_colloid_noise - colloid.potential(colloid_coords)
    @. semicolloid_displacement = scaled_semicolloid_noise - semicolloid.potential(semicolloid_coords)
    nothing
end

function update_potential_collisions!(
            potential_collisions_per_colloids::Vector{Int},
            colliding_semicolloids::Vector{Int},
            colloid_coords::PointList,
            semicolloid_coords::PointList,
            max_dist_sq
        )
    k = length(colliding_semicolloids)
    curr = 1
    @inbounds for i in eachindex(colloid_coords)
        potential_collisions_per_colloids[i] = 0
        @simd for j in eachindex(semicolloid_coords)
            rel_pos = colloid_coords[i] - semicolloid_coords[j]
            rel_dist_sq = sq_norm(rel_pos)
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


function resolve_overlaps!(
            colloid_displacement::PointList,
            semicolloid_displacement::PointList,
            colloid_coords::PointList,
            semicolloid_coords::PointList,
            sq_diam, sq_mixed_diam,
            potential_collisions_per_colloid::Vector{Int},
            colliding_semicolloids::Vector{Int}
        )
    n = length(colloid_coords)

    unresolved_overlaps = true
    while unresolved_overlaps
        unresolved_overlaps = false
        curr = 1
        @inbounds for j in 1:n
            for i in j+1:n
                overlap, modifier = check_overlap(
                    colloid_coords[i] - colloid_coords[j],
                    colloid_displacement[i] - colloid_displacement[j],
                    sq_diam
                )

                if overlap
                    unresolved_overlaps = true
                    colloid_displacement[i] -= modifier
                    colloid_displacement[j] += modifier 
                end
            end

            for k in curr:curr+potential_collisions_per_colloid[j]-1
                i = colliding_semicolloids[k]

                overlap, modifier = check_overlap(
                    colloid_coords[j] - semicolloid_coords[i],
                    colloid_displacement[j] - semicolloid_displacement[i],
                    sq_mixed_diam
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


function update_collision_times!(
            colloid_collision_times::Vector{Float64},
            semicolloid_collision_times::Vector{Float64},
            colloid_coords::PointList,
            semicolloid_coords::PointList,
            colloid_displacement::PointList,
            semicolloid_displacement::PointList,
            potential_collisions_per_colloid::Vector{Int},
            colliding_semicolloids::Vector{Int},
            sq_diam, sq_mixed_diam,
            remaining_time
        )
    next_collision_time = Inf 

    diff = length(colliding_semicolloids) - length(semicolloid_collision_times)
    if diff > 0
        append!(semicolloid_collision_times, zeros(Float64, diff))
    end

    n = length(colloid_coords)
    curr = 1
    @inbounds for j in 1:n
        for i in j+1:n
            q = triangular_index(i, j, n)

            colloid_collision_times[q] = collision_time(
                colloid_coords[i] - colloid_coords[j],
                colloid_displacement[i] - colloid_displacement[j],
                sq_diam, remaining_time
            )

            if colloid_collision_times[q] < next_collision_time
                next_collision_time = colloid_collision_times[q]
            end
        end

        for k in curr:curr+potential_collisions_per_colloid[j]-1
            i = colliding_semicolloids[k]
            semicolloid_collision_times[k] = collision_time(
                semicolloid_coords[i] - colloid_coords[j],
                semicolloid_displacement[i] - colloid_displacement[j],
                sq_mixed_diam, remaining_time
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
            colloid_displacement::PointList,
            semicolloid_displacement::PointList,
            colloid_coords::PointList,
            semicolloid_coords::PointList,
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
        @inbounds for j in 1:n
            for i in j+1:n 
                q = triangular_index(i, j, n)

                if colloid_collision_times[q] ≤ time_horizon
                    collision, modifier = check_collision(
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

                    collision, modifier = check_collision(
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



function step!(
            colloid_coords::PointMatrix,
            semicolloid_coords::PointMatrix,
            colloid_displacement::PointList,
            semicolloid_displacement::PointList,
            potential_collisions_per_colloid::Vector{Int},
            colliding_semicolloids::Vector{Int},
            colloid_collision_times::Vector{Float64},
            semicolloid_collision_times::Vector{Float64},
            colloid::Ball,
            semicolloid::Ball,
            sq_diam, sq_mixed_diam,
            Δt, time_tolerance, t,
            scaled_colloid_noise::PointList,
            scaled_semicolloid_noise::PointList,
            estimated_max_dist_sq
        )

    @inbounds colloid_coords[:, t+1] .= colloid_coords[:, t]
    @inbounds semicolloid_coords[:, t+1] .= semicolloid_coords[:, t]

    update_displacement!(
        colloid_displacement,
        semicolloid_displacement,
        colloid_coords[:, t+1],
        semicolloid_coords[:, t+1],
        colloid, semicolloid,
        scaled_colloid_noise, scaled_semicolloid_noise
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
                sq_diam, sq_mixed_diam,
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
                sq_diam, sq_mixed_diam,
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



end