# DISCLAIMER
This repository is work in progress, porting Python code to Julia. For more information, please refer to the original [repository](https://lab.wias-berlin.de/zass/dynamics-of-spheres).
## Installation
Install the package by opening Julia's Package manager (by pressing `]` and then `Enter`) and typing
`add "https://github.com/mushunrek/Colloids.jl.git"`. You can remove the package by typing `remove Colloids` in the Package manager environment.

## Example Code

### A simulation of colloids in a fluid

The following code runs a simulation for colloids in a fluid.

```julia
using Colloids

# define the initial colloid coordiantes
initial_configuration = [ [1.42*i, 1.42*i] for i in 1:10 ]

# define a quadratic potential to push colloids towards the center
potential = Quadratic()
# colloid radius = 1.0, colloid diffusivity = 1.0, colloids move in the quadratic potential
colloid = Ball(1.0, 1.0, potential=potential)
# fluid particle radius = 0.15, fluid density = 30.0
fluid = Fluid(0.15, 30.0)

# time horizon
T = 10.0
# time step
Δt = 0.005
# time resolution for collisions
time_tolerance = 1e-8

# run simulation
simulation = ColloidsInFluid(
				initial_configuration,
    			    colloid, fluid,
       			T, Δt, time_tolerance
				)
```

At the moment, there are three options to retrieve the data.
```julia
# animate via Plots.jl
animate(simulation, "output.gif", skipframes=10)

# save the coordinates to CSV files
to_csv(simulation)

# use a rudimentary PoVRay wrapper to directly produce a video via PoVRay
povray(simulation, output_path="output.mp4", fps=30)
```

### A simulation of colloids in semicolloids

TODO

## Authors
- Myriam Fradon
- Julian Kern
- Sylvie Rœlly
- Alexander Zass
## License
This project is licensed under the GNU Affero General Public License v3.0.
