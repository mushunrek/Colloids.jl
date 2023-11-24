using Colloids
using StaticArrays
using Distributions
#using BenchmarkTools
using Test

normal = Normal()

@testset "Colloids.jl" begin
    initial = [ [1.42*i, 1.42*i] for i in 1:7 ]
    #initial = [ [0.0, 0.0], [ 2.1,0.0 ], [1.0,0.93*2.1], [3.1 ,0.9*2.1] ]

    # radius = 1.0, diffusivity = 1.0
    colloid = Ball(1.0, 1.0, potential=Quadratic(strength=1.0))
    # radius = 0.3, density = 30.0
    fluid = Fluid(0.15, 30.0)

    T = 1.0
    Δt = 0.001
    time_tolerance = 1e-8

    sim = ColloidsInFluid(initial, colloid, fluid, T, Δt, time_tolerance)
    #animate(sim, "test.gif", skipframes=10)
    povray(sim, fps=30)
    #to_csv(sim)

    """colloid_initial = [ [1.42*i, 1.42*i] for i in 1:10 ]
    semicolloid_initial = rand(normal, 10^3, 2)
        
    # radius = 1.0, diffusivity = 1.0
    colloid = Ball(1.0, 1.0)
    # radius = 0.2, diffusivity = 10.0
    semicolloid = Ball(0.2, 1.0)
        
    T = 1.0
    Δt = 0.1
    time_tolerance = 1e-5
        
    @btime ColloidsInSemicolloids(colloid_initial, semicolloid_initial, colloid, semicolloid, T, Δt, time_tolerance)
    #animate(sim, "test2.gif")"""
end

@test_skip @testset "helper.jl" begin
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

@test_skip @testset "large_balls.jl" begin
    include("../src/large_balls.jl")

    import .LargeBalls

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

    displacement = zeros(SVector{2, Float64}, 10^3)
    coords = copy(reinterpret(
        SVector{2, Float64},
        rand(Normal(), 2*10^3)
    ))
    @benchmark LargeBalls.update_displacement!($displacement, $coords, $Σ, $rl, $rs, $zs, $Δt)
    @benchmark LargeBalls.update_displacement2!($displacement, $coords, $Σ, $rl, $rs, $zs, $Δt)

    
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


@test_skip @testset "all_balls.jl" begin
    include("../src/all_balls.jl")
    include("../src/helper.jl")

    using .Helper 
    using .AllBalls

    colloid_initial = copy(reinterpret(SVector{2, Float64}, [ 1.0*i for i in -9:10 ]))
    semicolloid_initial = copy(reinterpret(SVector{2, Float64}, 10.0 .* rand(normal, 2*10^4)))

    colloid = Ball(1.0, 1.0, AllBalls.no_potential)
    semicolloid = Ball(0.2, 1.0, x -> delayed_quadratic(x, delay=20.0, strength=0.01))

    T = 1.0
    Δt = 0.001
    time_tolerance = 1e-10

    sim = MixtureSimulation(colloid_initial, semicolloid_initial, colloid, semicolloid, T, Δt, time_tolerance)
    animate(sim, "./test.gif", fps=5)
end

