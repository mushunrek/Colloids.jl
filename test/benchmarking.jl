begin
    include("../src/large_balls.jl")

    using .LargeBalls
    using Distributions
    using StaticArrays 
    using LoopVectorization

    using BenchmarkTools
end

begin
    Σ = 1.3
    rl = 1.
    rs = 0.2
    zs = 10.0
    Δt = 0.1

    n = 10000

    displacement = zeros(SVector{2, Float64}, n);
    coords = copy(reinterpret(
                SVector{2, Float64},
                rand(Normal(), 2*n)
    ));
end

@benchmark LargeBalls.update_displacement!($displacement, coords, $Σ, $rl, $rs, $zs, $Δt) setup=(coords = copy(reinterpret(
    SVector{2, Float64},
    rand(Normal(), 2*n)
)))
@benchmark LargeBalls.update_displacement2!($displacement, coords, $Σ, $rl, $rs, $zs, $Δt) setup=(coords = copy(reinterpret(
    SVector{2, Float64},
    rand(Normal(), 2*n)
)))
