# Dynamics of two-type spheres
>This repository contains the code used to generate the simulations presented in <ins>link to preprint</ins>.

We present here the code for simulating the dynamics of $n$ large spheres that are moving in a bath of infinitely-many small particles.

<img src="media/spheres_still.png" alt="sim_4spheres" width="300"/>

## Simulation of two-type dynamics
In Section 2 of the paper, we study the random dynamics of $n$ hard spheres of radius $\mathring{r}$, centred in $\mathring{X}_1,\dots,\mathring{X}_n$, moving according to $n$ independent Brownian motions $(\mathring{W}_i)_i$, with a gradient drift $\nabla\mathring{\psi}$ to keep them near the origin.

They are moving in a bath of small particles (of radius $\dot{r}$), themselves moving according to indipendent Brownian motions. There is a non-overlap constraint between large spheres and particles; there is no direct interaction between particles and they can overlap.

The resulting dynamics is described by the following infinite-dimensional SDE: $i=1,\dots,n$, $k\geq 1$, $t\in[0,1]$,

$$
\mathring{X}_i(t) = \mathring{X}_i(0) + \mathring{W}_i(t) - \frac{1}{2}\int_0^t\mathring{\psi}(\mathring{X}_i(s))\, d s

+ \displaystyle\sum_{j=1}^n \int_0^t (\mathring{X}_i-\mathring{X}_j)(s)\, dL_{ij}(s)

+ \displaystyle\sum_{k\geq 1} \int_0^t (\mathring{X}_i-\dot{X}_k)(s)\, d\ell_{ik}(s).
$$

$$
\dot{X}_k(t) = \dot{X}_k(0) + \dot{\sigma}\dot{W}_k(t) + \dot{\sigma}^2 \displaystyle{\sum_{i=1}^n}\int_0^t (\dot{X}_k - \mathring{X}-i)(s)\, d\ell_{ki}(s).
$$

The hard-core interaction between spheres induces local times $L_{ij}(t)$ for each pair of spheres. The particle-to-sphere hard-core interaction induces the local times $\ell_{ik}(t)$.

If the distance between two spheres is smaller than twice the radius $\mathring{r}+\dot{r}$ (depletion radius, orange halo on the picture), there are no particles between the spheres. This empty zone in the particle bath induces an attractive pseudo-interaction between the spheres (depletion attraction).

We present here numerical simulations of this phenomenon in $d=2$.
The position of the system of spheres and particles is computed using our colloids Python package. The algorithm is event-driven: the actualization of the motion of the spheres occurs at every collision time. The visualisation is then generated with PoV-Ray and compressed in mp4. The distance unit in the simulation is equal to $\mathring{r}$. The time unit corresponds to the time at which any coordinate of the Brownian motion admits a standard deviation equal to $1$. The running time of the simulation is displayed on the left of the pictures. The simulations run for a finite but large number of particles, using a penalisation function to keep a large enough number of particles around the spheres. We proved in Section 2.1 that such a dynamics approximates the solution of the above infinite-dimensional SDE.

### Examples

In these examples, the constraining potential $\mathring{\psi}$ is equal to zero on a disc around the origin, and increases at a polynomial rate away from the origin. The full-sized files can be found <a href="https://cloud.wias-berlin.de/aotearoa/index.php/s/6AgayPcq7s6i6gJ" target="_blank">here</a> (locked; passcode in MoviePWD.txt).

**4 large spheres in a medium-density soup**

The radius of the small particles is relatively large here (35% of the radius of the large spheres) in order for the depletion area to be clearly visible in the simulation. As a result, the depletion attraction induces not only pair interactions, but also three-body effects. On the left with particles showing, on the right with particles hidden.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="media/4spheres_medium.gif" alt="4spheres_medium.gif" width="200"/>
&nbsp;&nbsp;<img src="media/4spheres_medium_hidden.gif" alt="4spheres_medium_hidden.gif" width="200"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $n=4,\rho=0.35,\dot{\sigma}=0.8,\dot{z}\simeq 3$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Particles showing (left) and particles hidden (right)

**6 large spheres in a high-density soup**

The collisions with the numerous small particles around the clusters does not allow the large spheres to leave their cluster: the larger the particle density, the stronger the depletion effect. The macroscopic resulting effect on the large spheres is similar to the effect of a short-range attractive potential between the spheres. This effect is particularly visible when the small particles are hidden, as done in the second part of the simulation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="media/6spheres_high_hidden.gif" alt="6spheres_high_hidden.gif" width="200"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $n=6,\rho=0.35,\dot{\sigma}=0.3,\dot{z}\simeq 6$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3 depletion clusters

**8 large spheres in a very high-density soup**

The radius of the particles here is only 20% of the radius of the spheres. Hence the depletion only induces a very short range attraction: there is no attraction as soon as spheres are separated by a distance larger than twice the depletion radius. The pairs of nearly colliding sphere are more stable than in the previous simulations thanks to the higher density of particles.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="media/8spheres_high.gif" alt="8spheres_high.gif" width="200"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $n=8,\rho=0.2,\dot{\sigma}=1,\dot{z}\simeq 50$

**4 large spheres moving around minimal energy configurations**
As explained in Section 3 of the paper, the equilibrium measure of the system has a strong probability to put the spheres in the vicinity of their minimum energy configurations. These configurations have maximum contact number (5 contact points for 4 spheres here). The higher the density, the closer the configurations tend to be to the minimum: it takes time here for the contact number of the configuration to drop from 5 to 4.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="media/4optimal.gif" alt="4optimal.gif" width="200"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $n=4,\rho=0.25,\dot{\sigma}=0.8,\dot{z}\simeq 25$

## Usage
The above simulations are only a small sample of possible interesting effects. The interested reader can perform simulations with their own set of parameters using the Python class colloid created for the paper.

Run balls_in_particles.py for the spheres in random medium (or balls_in_fluid.py for the drifted sphere system) with parameters file corresponding to the given initial configurations of the spheres and the particles. The numerical result is a collection of .csv files containing the coordinates of each balls at each time.

For a visualisation of the result, use the .ini and .pov files to run <a href="http://povray.org/" target="_blank">PoV-Ray</a> with the same parameters (or these .ini and .pov for the drifted dynamics instead of random medium dynamics).


## Authors
- Myriam Fradon
- Julian Kern
- Sylvie Rœlly
- Alexander Zass
## License
This project is licensed under the GNU Affero General Public License v3.0.
