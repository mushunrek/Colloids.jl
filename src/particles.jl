module Particles 

using ..Potentials

export Particle, Ball, Fluid 
export radius, diameter, diffusivity, fluid_density

abstract type Particle end

struct Ball <: Particle 
    R::Float64
    Σ::Float64
    potential::Potential 
end

struct Fluid <: Particle
    R::Float64
    z::Float64
    Fluid(R, z) = new(convert(Float64, R), convert(Float64, z))
end

Ball(R, Σ; potential::Potential=Potentials.Null) = Ball(convert(Float64, R), convert(Float64, Σ), potential)


radius(p::Particle) = p.R 
diameter(p::Particle) = 2*radius(p)
diameter(p1::Particle, p2::Particle) = radius(p1) + radius(p2)
diffusivity(b::Ball) = b.Σ
fluid_density(f::Fluid) = f.z

end