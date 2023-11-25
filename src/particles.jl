module Particles 

using ..Potentials

export Particle, Ball, Fluid 
export radius, diameter, diffusivity, fluid_density

"""
Abstract supertype for all particles.
"""
abstract type Particle end

"""
    Ball <: Particle

Type for spherical particles in a potential.

# Fields
- `R::Float64`: particle radius
- `Σ::Float64`: particle diffusivity 
- `potential::Potential`: environmental potential 
"""
struct Ball <: Particle 
    R::Float64
    Σ::Float64
    potential::Potential
    Ball(R, Σ, potential) = new(convert(Float64, R), convert(Float64, Σ), potential)
end

"""
    Fluid <: Particle

Type for fluids modelling "mean-field" behaviour of small particles.

# Fields
- `R::Float64`: radius of fluid particles 
- `z::Float64`: particle density in fluid 
"""
struct Fluid <: Particle
    R::Float64
    z::Float64
    Fluid(R, z) = new(convert(Float64, R), convert(Float64, z))
end

"""
    Ball(R, Σ [; potential])

Preferred constructor with potential as kwarg. If no potential is specified,
the null potential `Null` is used.
"""
Ball(R, Σ; potential::Potential=Potentials.Null) = Ball(R, Σ, potential)

"""
    radius(p::Particle) -> Float64

Returns the radius `p.R` of the particle.
"""
radius(p::Particle) = p.R 

"""
    diameter(p::Particle) -> Float64

Return the diameter `2*radius(p)` of the particle. See also `?radius`.
"""
diameter(p::Particle) = 2*radius(p)

"""
    diameter(p1::Particle, p2::Particle) -> Float64

Returns the "mixed" diamter `radius(p1) + radius(p2)` of the two particles.
See also `?radius`.
"""
diameter(p1::Particle, p2::Particle) = radius(p1) + radius(p2)

"""
    diffusivity(b::Ball) -> Float64

Returns the diffusivity `b.Σ` of the ball.
"""
diffusivity(b::Ball) = b.Σ

"""
    fluid_density(f::Fluid) -> Float64

Returns the density `f.z` of the fluid.
"""
fluid_density(f::Fluid) = f.z

end