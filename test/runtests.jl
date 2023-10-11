using Colloids
using StaticArrays
using Distributions
using Test

@testset "Colloids.jl" begin
    # Write your tests here.
end

@testset "helper.jl" begin
    include("../src/helper.jl")
    import .Helper

    x = @SVector [0.3; 0.5]
    y = @SVector [0.1; -0.7]
    @test Helper.dot(x, y) ≈ -0.32
    @test Helper.sq_norm(x) ≈ 0.34
    @test Helper.sq_norm(y) ≈ 0.5

    d = 1.
    overlap, modifier = Helper.check_overlap(x, y, d^2)
    @test overlap == true
    @test typeof(modifier) == SVector{2, Float64}
    @test modifier == [ -0.2823529411764706, -0.47058823529411764 ]
    @test Helper.collision_time(x, y, d^2, 0.1) == Inf

    d = 0.01
    overlap, modifier = Helper.check_overlap(x, y, d^2)
    @test overlap == false
    @test typeof(modifier) == SVector{2, Float64}
    @test modifier == [0.0, 0.0]
    @test Helper.collision_time(x, y, d^2, 0.1) == Inf

    x = @SVector [ 1.; 1. ]
    y = @SVector [ -1.; -1. ]
    d = 0.2
    @test 0.0 < Helper.collision_time(x, y, d^2, 1.0) < 1.0
    @test Helper.collision_time(x, y, d^2, 1.0) != Inf


    # still need to test check_collision!
end

@testset "large_balls.jl" begin
    include("../src/large_balls.jl")

    import .LargeBalls

    normal = Normal()
    displacement = zeros(SVector{2, Float64}, 2)
    coords = copy(reinterpret(
        SVector{2, Float64}, 
        [0.0, 0.0, 2.1, 0.0]
    ))
    noise = copy(reinterpret(
        SVector{2, Float64}, 
        [.104, 1.2, -0.43, 0.01]
    ))
    Σ = 1.3
    rl = 1.
    rs = 0.2
    zs = 10.0
    Δt = 0.1
    LargeBalls.update_displacement__testing!(displacement, coords, Σ, rl, rs, zs, Δt, noise)
    @test typeof(displacement) == Vector{SVector{2, Float64}}
    @test displacement == Vector(
                            [
                                SVector{2}([-0.9390472842981036  0.4933153149862672]),
                                SVector{2}([0.8050299570601677   0.004110960958218893])
                            ]
                        )

    
    initial = copy(reinterpret(SVector{2, Float64}, [ 1.42*i for i in -1:10 ]))
    R = 1.0
    Σ = 1.0
    r = 0.5
    T = 5.0
    Δt = 0.01
    time_tolerance = 1e-6
    sim = LargeBalls.ColloidSimulation(initial, Σ, R, rs, zs, T, Δt, time_tolerance)
    LargeBalls.save_colloids_to_pov(sim, LargeBalls.generate_foldername(sim))
    LargeBalls.animate(sim)
    
end

