module Potentials

using ..Points

export Potential 
export Null, Quadratic, DelayedQuadratic

struct Potential <: Function 
    potential
end

(p::Potential)(x::Point) = p.potential(x)

Null = Potential(x -> zeros(Point))
Quadratic(; strength) = Potential(x -> strength*x)
function DelayedQuadratic(; delay, strength)
    @inline function f(x::Point)
        r2 = sq_norm(x)
        ( r2 > delay ) ? strength * (1 - delay/r2) .* x : zeros(Point)
    end
    return Potential(f)
end

end