module LargeBalls

include("helper.jl")

using .Helper
using StaticArrays
using Tullio, LoopVectorization
using Distributions
using JLD2, DelimitedFiles, Formatting
using Plots

export simulate_colloids

normal = Normal()

function update_displacement!(
        displacement::Vector{SVector{2, Float64}},
        coords::Vector{SVector{2, Float64}},
        Σ, rl,
        rs, zs, 
        Δt 
    )
    n = length(displacement)
    noise = copy(reinterpret(SVector{2, Float64}, rand(normal, 2*n)))
    dls = rl + rs
    c1 = 4 * dls^2
    c2 = (Σ*zs*dls*Δt)
    @tullio displacement[i] = (
                                (1 < Helper.sq_norm(coords[j] - coords[i]) < c1) ?
                                    (
                                        .√(1 .- (coords[j] - coords[i]).^2 ./ c1) 
                                        .* (coords[j] - coords[i]) 
                                        ./ √Helper.sq_norm(coords[j] - coords[i])
                                    ) .* c2 :
                                    @SVector [0.0, 0.0]
                            )
    displacement .= (Σ*√Δt) .* noise .- displacement
    return nothing
end


function update_displacement__testing!(
                                    displacement::Vector{SVector{2, Float64}},
                                    coords::Vector{SVector{2, Float64}},
                                    Σ, rl,
                                    rs, zs, 
                                    Δt,
                                    noise::Vector{SVector{2, Float64}}
                                )
    dls = rl + rs
    for i in eachindex(coords)
        displacement[i] = (√Δt * Σ) .* noise[i]
        for j in eachindex(coords)
            rel_pos = coords[j] - coords[i]
            rel_dist_sq = Helper.sq_norm(rel_pos)
            if 1 < rel_dist_sq < 4 * dls^2
                displacement[i] -= (
                                        .√(1 .- rel_pos.^2 ./ (4 * dls^2) ) .* rel_pos ./ √rel_dist_sq
                ) .* (Σ^2 * zs * dls * Δt)
            end
        end
    end
    return nothing
end

"""
    triangular_index(i, j, n)

Return the index of an element in a lower triangular matrix when elements are 
indexed columnwise from left to right.
"""
@inline triangular_index(i, j, n) = (j-1)*(2*n - j - 2) ÷ 2 + (i - j)

function update_collision_times!(
                collision_times::Vector{Float64},
                coords::Vector{SVector{2, Float64}},
                displacement::Vector{SVector{2, Float64}},
                R, remaining_time
            )
    n = length(coords)
    next_collision_time = Inf
    for j in 1:n-1
        for i in j+1:n
            q = triangular_index(i, j, n)

            @inbounds collision_times[q] = Helper.collision_time(
                                            coords[i] - coords[j],
                                            displacement[i] - displacement[j],
                                            2*R, remaining_time
                                        )
            @inbounds if collision_times[q] < next_collision_time
                @inbounds next_collision_time = collision_times[q]
            end
        end
    end
    return next_collision_time
end

function handle_collisions!(
                displacement::Vector{SVector{2, Float64}},
                coords::Vector{SVector{2, Float64}},
                collision_times::Vector{Float64},
                end_of_collision
            )
    n = length(coords)

    unresolved_overlaps = true
    while unresolved_overlaps
        for j in 1:n-1
            for i in j+1:n
                q = triangular_index(i, j, n)

                if collision_times[q] ≤ end_of_collision
                    collision, modifier = Helper.check_collision(
                                                            coords[i] - coords[j],
                                                            displacement[i] - displacement[j]
                                                        )
                    if collision
                        unresolved_overlaps = true
                        displacement[i] -= modifier
                        displacement[j] += modifier
                    end
                end
            end
        end
    end
end

function resolve_overlaps!(
                displacement::Vector{SVector{2, Float64}},
                coords::Vector{SVector{2, Float64}},
                R
            )
    n = length(coords)

    unresolved_overlaps = true
    while unresolved_overlaps
        unresolved_overlaps = false
        for j in 1:n-1
            for i in j+1:n
                collision, modifier = Helper.check_overlap(
                                                coords[i] - coords[j],
                                                displacement[i] - displacement[j],
                                                2*R
                                            )
                if collision
                    unresolved_overlaps = true
                    displacement[i] -= modifier
                    displacement[j] += modifier
                end
            end
        end
    end
end

function step!(
            coords::Matrix{SVector{2, Float64}},
            displacement::Vector{SVector{2, Float64}},
            collision_times::Vector{Float64},
            Σ, R, rs, zs,
            t, Δt, time_tolerance 
        )
    @show t
    coords[:, t+1] .= coords[:, t]
    update_displacement!(displacement, coords[:, t+1], Σ, R, rs, zs, Δt)
    remaining_time = Δt
    while remaining_time > 0
        resolve_overlaps!(displacement, coords[:, t+1], R)
        next_collision_time = update_collision_times!(collision_times, coords[:, t+1], displacement, R, remaining_time)


        if next_collision_time ≥ remaining_time
            coords[:, t+1] .+= displacement 
            remaining_time = 0.0
        else
            end_of_collision = min(
                next_collision_time + time_tolerance,
                remaining_time
            )
            time_fraction = end_of_collision / remaining_time
            coords[:, t+1] .+= time_fraction .* displacement 
            displacement .*= (1 - time_fraction)

            handle_collisions!(displacement, coords[:, t+1], collision_times, end_of_collision)
            remaining_time -= end_of_collision
        end
    end
end

struct ColloidSimulation 
    coords::Matrix{SVector{2, Float64}}
    Σ::Float64
    R::Float64
    rs::Float64
    zs::Float64
    T::Float64
    Δt::Float64
end

function ColloidSimulation(
                    initial_configuration::Vector{SVector{2, Float64}},
                    Σ, R, rs, zs,
                    T, Δt, time_tolerance
                )
    n = length(initial_configuration)
    steps = floor(Int, T / Δt)
    coords = zeros(SVector{2, Float64}, n, steps+1)
    coords[:, 1] = initial_configuration

    displacement = zeros(SVector{2, Float64}, n)
    collision_times = zeros(Float64, n*(n-1)÷2)

    for t in 1:steps
        step!(coords, displacement, collision_times, Σ, R, rs, zs, t, Δt, time_tolerance)
    end

    return ColloidSimulation(coords, Σ, R, rs, zs, T, Δt)
end

function generate_foldername(sim::ColloidSimulation, path="./pov/")
    n = length(sim.coords[:, 1])
    "$(path)colloids__n$(n)R$(sim.R)Sigma$(sim.Σ)rs$(sim.rs)zs$(sim.zs)T$(sim.T)dt$(sim.Δt)/"
end


function save_colloids(coords::Matrix{SVector{2, Float64}}, filename)
    save_object(filename, coords)
end

function load_colloids(filename)
    load_object(filename)
end

function save_colloids_to_pov(sim::ColloidSimulation, foldername)
    mkpath(foldername)
    cd(foldername)
    n = length(sim.coords[:, 1])

    for t in 1:size(sim.coords, 2)
        timestamp = fmt("2.10f", t*sim.Δt)
        curr_file = "boules$(timestamp)"
        current_coords = reshape(
            copy(
                reinterpret(
                    Float64,
                    sim.coords[:, t]
                )
            ),
            n, 2
        )
        writedlm(curr_file, current_coords, ",")
    end
end


function circle(c, r)
    θ = LinRange(0.0, 2*π, 100)
    c[1] .+ r*cos.(θ), c[2] .+ r*sin.(θ)
end

function animate(sim::ColloidSimulation)
    anim = @animate for t in 1:size(sim.coords, 2)
        plot(label="t=$t")
        for c in eachrow(sim.coords[:, t])
            plot!(
                circle(c[1], sim.R),
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
    return gif(anim, "~/Downloads/test.gif")
end

end