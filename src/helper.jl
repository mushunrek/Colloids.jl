module Helper

using StaticArrays
using Tullio, LoopVectorization
using Distributions

export Particle, Ball, Fluid
export radius, d, diffusivity, density
export triangular_index, generate_noise
export check_overlap, collision_time, check_collision

global const normal = Normal()

abstract type Particle end

struct Ball
    R::Float64
    Σ::Float64
end

struct Fluid
    R::Float64
    density::Float64
end

@inline radius(p::Particle) = p.R
@inline d(p1::Particle, p2::Particle) = radius(p1) + radius(p2)
@inline diffusivity(b::Ball) = b.Σ
@inline density(f::Fluid) = f.density


@inline dot(x::SVector{2, Float64}, y::SVector{2, Float64}) = x[1]*y[1] + x[2]*y[2]
@inline sq_norm(x::SVector{2, Float64}) = x[1]^2 + x[2]^2

@inline modifier(rel_pos::SVector{2, Float64}, rel_prod, rel_dist_sq) = (rel_prod / rel_dist_sq) .* rel_pos
@inline modifier(rel_pos::SVector{2, Float64}, rel_prod) = modifier(rel_pos, rel_prod, sq_norm(rel_pos))


"""
    triangular_index(i, j, n)

Return the index of an element at position `(i,j)`in a lower triangular `n×n` matrix 
when elements are indexed columnwise from left to right.
"""
@inline triangular_index(i, j, n) = (j-1)*(2*n - j - 2) ÷ 2 + (i - j)

function generate_noise(n, m)
    reshape(
        copy(
            reinterpret(
                SVector{2, Float64},
                rand(normal, 2*n*m)
            )
        ),
        (n, m)
    )
end


"""
    resolve_overlaps(relative_position, relative_displacement, d2)

Determines whether two balls collide and compute the resulting change in displacement.
`d2` is the squared sum of the radii.
"""
function check_overlap(
        relative_position::SVector{2, Float64},
        relative_displacement::SVector{2, Float64},
        d2
    )
    relative_product = dot(relative_position, relative_displacement)
    relative_distance_squared = sq_norm(relative_position)
    if ( relative_distance_squared ≤ d2 ) && ( relative_product < 0.0 )
        return true, modifier(relative_position, relative_product, relative_distance_squared)
    end
    return false, zeros(SVector{2, Float64})
end

function collision_time(
        relative_position::SVector{2, Float64},
        relative_displacement::SVector{2, Float64},
        d2, remaining_time
    )
    relative_distance_squared = sq_norm(relative_position)

    if relative_distance_squared > d2
        relative_product = dot(relative_position, relative_displacement)
        relative_displacement_squared = sq_norm(relative_displacement)
        discriminant = relative_product^2 - relative_displacement_squared * (relative_distance_squared - d2)
        if ( relative_product < 0.0 ) && ( discriminant ≥ 0.0 )
            return -remaining_time * ( relative_product + √discriminant ) / relative_displacement_squared
        end
    end
    return Inf
end

function check_collision(
        relative_position::SVector{2, Float64},
        relative_displacement::SVector{2, Float64}
    )
    relative_product = dot(relative_position, relative_displacement)
    if relative_product < 0.0
        return true, modifier(relative_position, relative_product)
    end
    return false, zeros(SVector{2, Float64})
end




end