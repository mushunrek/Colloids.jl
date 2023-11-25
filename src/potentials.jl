module Potentials

using ..Points

export Potential 
export Null, Quadratic, DelayedQuadratic

struct Potential <: Function 
    potential
end

(p::Potential)(x::Point) = p.potential(x)

Null = Potential(x -> zeros(Point))
Quadratic(; strength=1.0) = Potential(x -> strength*x)
function DelayedQuadratic(; delay, strength)
    @inline function f(x::Point)
        r = √sq_norm(x)
        ( r > delay ) ? strength * (1 - delay/r) .* x : zeros(Point)
    end
    return Potential(f)
end

end