"""
    Colloids

A Julia module to simulate colloids.

# API 

- `ColloidsInFluid` simulates colloids in a fluid of microscopic particles by 
    simulating the depletion directly.
- `ColloidsInSemicolloids` simulates colloids in contact with semicolloids 
    (i.e. colloids that do not interact with other particles of the same type).

# Exported names

    Point, Pointlist,
    Potential, Null, Quadratic, DelayedQuadratic,
    Particle, Ball, Fluid,
    ColloidsInFluid, ColloidsInSemicolloids,
    animate
"""
module Colloids

using LoopVectorization
using JLD2, DelimitedFiles, Formatting
using Plots#, PoVRay

include("points.jl")
include("potentials.jl")
include("particles.jl")
include("core.jl")

using .Points, .Potentials, .Particles
using .ColloidsCore

export Point, PointList
export Potential, Null, Quadratic, DelayedQuadratic
export Particle, Ball, Fluid 
export ColloidsInFluid, ColloidsInSemicolloids
export animate, povray, to_csv

"""
# Type docstring 

Abstract type to supertype both `ColloidsInFluid` and `ColloidsInSemicolloids`.
"""
abstract type AbstractSimulation end

"""
# Type docstring
    ColloidsInFluid(coords::PointMatrix, colloid::Ball, fluid::Fluid, T::Float64, Δt::Float64)

Object containing the simulation data with given parameters. `coords` contains the
full evolution with dimensions (coordinates, time).
"""
struct ColloidsInFluid <: AbstractSimulation
    coords::PointMatrix
    colloid::Ball
    fluid::Fluid
    T::Float64
    Δt::Float64
end

"""
# Type docstring
    ColloidsInSemicolloids(
        colloid_coords::PointMatrix, semicolloid_coords::PointMatrix,
        colloid::Ball, semicolloid::Ball,
        T::Float64, Δt::Float64
    )

Object containing the simulation data with given parameters. `colloid_coords` 
and `semicolloid_coords` contain the full evolution with dimensions (coordinates, time).
"""
struct ColloidsInSemicolloids <: AbstractSimulation
    colloid_coords::PointMatrix
    semicolloid_coords::PointMatrix
    colloid::Ball
    semicolloid::Ball
    T::Float64
    Δt::Float64
end

"""
# Constructor 
    ColloidsInFluid(initial, colloid::Ball, fluid::Fluid, T, Δt, time_tolerance)

Constructs the full simulation starting from the initial condition `initial`.
- `initial`: initial configuration of colloids
- `colloid::Ball`: parameters for colloid 
- `fluid::Fluid`: parameters for fluid
- `T`: time horizon for simulation 
- `Δt`: time step for simulation 
- `time_tolerance`: temporal resolution for collisions

# Example 

```
using Colloids 

initial = [ [1.42*i, 1.42*i] for i in 1:10 ]

# radius = 1.0, diffusivity = 1.0
colloid = Ball(1.0, 1.0)
# radius = 0.2, density = 10.0
fluid = Fluid(0.2, 10.0)

T = 1.0
Δt = 0.1
time_tolerance = 1e-5

sim = ColloidsInFluid(initial, colloid, fluid, T, Δt, time_tolerance)
animate(sim, "test.gif")
```
"""
function ColloidsInFluid(
                    initial,
                    colloid::Ball,
                    fluid::Fluid,
                    T, Δt, time_tolerance
                )
    T = Float64(T)
    Δt = Float64(Δt)
    time_tolerance = Float64(time_tolerance)
    initial = PointList(initial)

    n = length(initial)
    steps = floor(Int, T / Δt)

    # pre-allocate coordinates for the whole simulation
    coords = zeros(Point, n, steps+1)
    # initialize coordinates at time `t = 0, converting to `PointList`
    coords[:, 1] = initial

    # pre-allocate containers for displacement and times of potential collisions
    displacement = zeros(Point, n)
    collision_times = zeros(Float64, n*(n-1)÷2)

    # pre-allocate standard normal random variables and scale them
    scaled_noise = (diffusivity(colloid) * √Δt) .* generate_noise(n, steps)

    # pre-compute constants 
    magic_cst1 = 4 * diameter(colloid, fluid)^2
    magic_cst2 = diffusivity(colloid) * fluid_density(fluid) * diameter(colloid, fluid) * Δt
    sq_diam = diameter(colloid)^2

    # simulate
    for t in 1:steps
        #@show t
        step!(
            coords, displacement, collision_times, 
            magic_cst1, magic_cst2, sq_diam, 
            colloid, Δt, time_tolerance, t, 
            scaled_noise[:, t]
        )
    end

    return ColloidsInFluid(coords, colloid, fluid, T, Δt)
end

"""
# Constructor 
    ColloidsInSemicolloids(
        colloid_initial, semicolloid_initial,
        colloid::Ball, semicolloid::Ball,
        T, Δt, time_tolerance
        [, estimated_max_travel]
    )

Constructs the full simulation starting from `colloid_initial` and `semicolloid_initial`.

- `colloid_initial`: initial configuration of colloids
- `semicolloid_initial`: initial configuration of semicolloids
- `colloid::Ball`: parameters for colloids 
- `semicolloid::Ball`: parameters for semicolloids 
- `T`: time horizon for simulation 
- `Δt`: time step for simulation 
- `time_tolerance`: time resolution for collisions 
- `estimated_max_travel`: (default is `missing`) upper bound for the maximal displacement 
    of (semi)colloids. If set `missing`, extremal theory is used to estimate an 
    upper bound. Simulations are only exact if set to `Inf`, but then the complexity
    goes to `O(number of colloids * number of semicolloids)`.

# Example 

```
using Colloids 
using Distributions

normal = Normal()
    
colloid_initial = [ [1.42*i, 1.42*i] for i in 1:10 ]
semicolloid_initial = rand(normal, 10^3, 2)
    
# radius = 1.0, diffusivity = 1.0
colloid = Ball(1.0, 1.0)
# radius = 0.2, diffusivity = 10.0
semicolloid = Ball(0.2, 1.0)
    
T = 1.0
Δt = 0.1
time_tolerance = 1e-5
    
sim = ColloidsInSemicolloids(colloid_initial, semicolloid_initial, colloid, semicolloid, T, Δt, time_tolerance)
animate(sim, "test.gif")
```
"""
function ColloidsInSemicolloids(
            colloid_initial,
            semicolloid_initial,
            colloid::Ball,
            semicolloid::Ball,
            T, Δt, time_tolerance;
            estimated_max_travel=missing
        )
    T = Float64(T)
    Δt = Float64(Δt)
    time_tolerance = Float64(time_tolerance)
    colloid_initial = PointList(colloid_initial)
    semicolloid_initial = PointList(semicolloid_initial)

    n = length(colloid_initial)
    m = length(semicolloid_initial)
    steps = floor(Int, T/Δt)

    colloid_coords = zeros(Point, n, steps+1)
    colloid_coords[:, 1] .= colloid_initial
    semicolloid_coords = zeros(Point, m, steps+1)
    semicolloid_coords[:, 1] .= semicolloid_initial

    colloid_displacement = zeros(Point, n)
    semicolloid_displacement = zeros(Point, m)
    
    potential_collisions_per_colloid = zeros(Int, n)
    colliding_semicolloids = zeros(Int, m)

    colloid_collision_times = zeros(Float64, n*(n-1)÷2)
    semicolloid_collision_times = zeros(Float64, m)

    if estimated_max_travel === missing
        # use extreme value theorem to estimate upper bound for the maximum displacement 
        estimated_max_travel = 4*√Δt * (
            diffusivity(colloid) * √(log(n))
            + diffusivity(semicolloid) * √(log(m))
        )
    end
    estimated_max_dist_sq = (estimated_max_travel + diameter(colloid, semicolloid))^2

    sq_diam = diameter(colloid)^2
    sq_mixed_diam = diameter(colloid, semicolloid)^2

    scaled_colloid_noise = (diffusivity(colloid)*√Δt) .* generate_noise(n, steps)
    scaled_semicolloid_noise = (diffusivity(semicolloid)*√Δt) .* generate_noise(m, steps)

    for t in 1:steps
        #@show t
        step!(
            colloid_coords, semicolloid_coords,
            colloid_displacement, semicolloid_displacement,
            potential_collisions_per_colloid, colliding_semicolloids,
            colloid_collision_times, semicolloid_collision_times,
            colloid, semicolloid,
            sq_diam, sq_mixed_diam,
            Δt, time_tolerance, t,
            scaled_colloid_noise[:, t], scaled_semicolloid_noise[:, t],
            estimated_max_dist_sq
        )
    end

    return ColloidsInSemicolloids(colloid_coords, semicolloid_coords, colloid, semicolloid, T, Δt)
end



"""
    circle(c, r)

Returns the circle of center `c` and radius `r`. Used for plotting only.
"""
function circle(c, r)
    θ = LinRange(0.0, 2*π, 100)
    c[1] .+ r*cos.(θ), c[2] .+ r*sin.(θ)
end

"""
    animate(sim:T, filename [; keyargs]) where T <: AbstractSimulation

Produces a GIF of the simulation and saves it at the given location. 
For the keywords, refer to the documentation of the particular methods.
"""
function animate end

"""
    animate(sim::ColloidsInFluid, filename [; fps, skipframtes])

See `?animate` for the general description.

- `fps` (default: `20`): determines the GIF frame rate. One frame corresponds to 
    one timestamp.
- `skipframes` (default: `0`): Determines the number of timestamps to be 
    skipped between frames.
"""
function animate(sim::ColloidsInFluid, filename; fps=20, skipframes=0)
    anim = @animate for t in 1:size(sim.coords, 2)
        ((t-1) % (skipframes+1) == 0) || continue

        plot(label="t=$t")
        for c in eachrow(sim.coords[:, t])
            plot!(
                circle(
                    c[1], radius(sim.colloid)
                ),
                seriestype=:shape,
                fillaplha=0.2,
                label=false,
                aspect_ratio=1,
                xlims=(-10.0, 20.0),
                ylims=(-10.0, 20.0)
            )
        end
        plot!()
    end
    return gif(anim, filename, fps=fps)
end


"""
    animate(sim::ColloidsInSemicolloids, filename [; fps, skipframes, semicolloids])

See `?animate` for the general description.

- `fps` (default: `20`): determines the GIF frame rate. One frame corresponds to 
    one timestamp.
- `skipframes` (default: `0`): Determines the number of timestamps to be 
    skipped between frames.
- `semicolloids` (default: `true`): Determines whether semicolloids are plotted
    (if `true`) or disregarded in the plots (if `false`).
"""
function animate(sim::ColloidsInSemicolloids, filename; fps=20, skipframes=0, semicolloids=true)
    anim = @animate for t in 1:size(sim.colloid_coords, 2)
        ((t-1) % (skipframes+1) == 0) || continue

        plot(label="t=$t")
        for i in eachindex(sim.colloid_coords[:, t])
            plot!(
                circle(
                    sim.colloid_coords[i, t], radius(sim.colloid)
                ),
                seriestype=:shape,
                fillaplha=0.8,
                label=false,
                aspect_ratio=1,
                xlims=(-5.0, 5.0),
                ylims=(-5.0, 5.0)
            )
        end
        
        if semicolloids
            for i in eachindex(sim.semicolloid_coords[:, t])
                plot!(
                    circle(
                        sim.semicolloid_coords[i, t], radius(sim.semicolloid)
                    ),
                    seriestype=:shape,
                    label=false
                )
            end
        end
        
        plot!()
    end
    return gif(anim, filename, fps=fps)
end

"""
function povray(sim::ColloidsInFluid, output_path::String="./test.mp4"; fps=10)
    tempdir = mktempdir(prefix="pov_")
    center = sum(sim.coords[:, end])/length(sim.coords[:, end])
    k = floor(Int, log(10, length(sim.coords[1, :]))) + 1
    Threads.@threads for t in eachindex(sim.coords[1,:])
        coords = sim.coords[:, t]
        objects = Array{Object}(undef, 2*length(coords))
        @inbounds @simd for i in eachindex(coords)
            p = [ coords[i][1], 0.0, coords[i][2] ]
            objects[i] = Colored(Sphere(p, diameter(sim.colloid, sim.fluid)), RGBFT("yellow", transmit=0.9))
            objects[length(coords) + i] = Colored(Sphere(p, radius(sim.colloid)), RGBFT("red"))
        end
        render(
            CSGUnion(objects), 
            LookAtCamera([center[1], 40.0, center[2]], [center[1], 0.0, center[2]]), 
            ini_path="tempdir/auto_generatedt.ini",
            pov_path="tempdir/auto_generatedt.pov",
            output_path="tempdir/pic(t).png"
        )
    end
    run(`ffmpeg -y -framerate fps -i tempdir/pic%d.png -c:v libx264 -pix_fmt yuv420p output_path`)
end
"""
"""
    to_csv(sim::T [; folder]) where T <: AbstractSimulation

Writes simulation data to CSV files. If no folder is specified, files will be
saved in "./data/". Each file corresponds to one timestamp which is also used
to name the file. In the case of `ColloidsInSemicolloids`, the coordinates of the
semicolloids are appended to the colloid coordinates. As such, it is 
important to save the parameters to extract the information later on.
"""
function to_csv end

function to_csv(sim::ColloidsInFluid; folder="./data/")
    mkpath(folder)
    for i in eachindex(sim.coords[1,:])
        t = (i-1)*sim.Δt
        filename = "$(folder)boules$(fmt("2.10f", t)).csv"
        CSV_writer(sim.coords[:, i], filename)
    end
end

function to_csv(sim::ColloidsInSemicolloids; folder="./data/")
    mkpath(folder)
    for i in eachindex(sim.colloid_coords[1, :])
        t = (i-1)*sim.Δt
        data = vcat( sim.colloid_coords[:, i], sim.semicolloid_coords[:, i] )
        filename = "$(folder)boules$(fmt("2.10f", t)).csv"
        CSV_writer(data, filename)
    end
end


"""
    CSV_writer(coords::PointList, filename::String [; format::String])

Writes a `PointList` to a CSV file. The default format for floating point numbers
is "6.10f".
"""
function CSV_writer(coords::PointList, filename::String; format::String="6.10f")
    open(filename, "w") do f
        for c in coords
            printfmtln(f, "{1:$format},{2:$format},", c...)
        end
    end
end

end
