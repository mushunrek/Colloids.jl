module Potentials

using ..Points

export Potential 
export Null, Quadratic, DelayedQuadratic

"""
    Potential <: Function

Type for environmental potentials.

# Fields
    potential

# Usage
A potential `p` is callable in the sense that `p(x)` is the same as `p.potential(x)`.
Here, `x` is always a `Point` at which the potential will return the opposite of 
the corresponding drift. This is a conscious choice as to force the usual `-` in 
front of the term from the potential.

# Examples
```
using Colloids

q = Quadratic(strength=3.0)
x = Point([4.0, -3.2])
drift = -q(x)
```

# See also 
    ?Null
    ?Quadratic
    ?DelayedQuadratic
"""
struct Potential <: Function 
    potential
end

# Make potentials callable.
(p::Potential)(x::Point) = p.potential(x)

"""
    Null 

Null potential.
"""
Null = Potential(x -> zeros(Point))

"""
    Quadratic([; strength]) -> Potential

A quadratic potential around the origin: a particle at `x` experiences a drift 
`-strength .* x`. The default strength is `1.0`.
"""
Quadratic(; strength=1.0) = Potential(x -> strength*x)

"""
    DelayedQuadratic(; delay, strenght) -> Potential

A quadratic potential with delay: a particle at `x` experiences a drift only if 
`‖x‖₂ > delay` in which case the drift is `-strength * (1 - delay/‖x‖₂) .* x`. 
"""
function DelayedQuadratic(; delay, strength)
    @inline function f(x::Point)
        r = √sq_norm(x)
        ( r > delay ) ? strength * (1 - delay/r) .* x : zeros(Point)
    end
    return Potential(f)
end

end