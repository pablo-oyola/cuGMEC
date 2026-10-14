# cuGMEC

**cuGMEC** is a GPU code for gyrokinetic-MHD hybrid simulations of tokamaks and stellarators, written in CUDA C++. It is
the GPU reconstruction of GMEC, the Gyrokinetic-MHD Energetic-particle Code
{cite:p}`jiang2024gmec1,liu2024gmec2,liu2026cugmec`:

- **fluid part:** a reduced, nonlinear two-fluid model with fluid electrons: gyrokinetic vorticity equation, gyrokinetic
  Poisson equation, parallel Ampère's law, parallel Ohm's law, electron continuity and the electron isothermal
  condition;
- **kinetic part:** up to three gyrokinetic ion species (thermal ions, alphas, beam ions), advanced with the $\delta f$
  particle-in-cell method;
- **coupling:** the perturbed pressures of the kinetic species enter the vorticity equation through the
  pressure-curvature term.

It is used for energetic-particle-driven Alfvén eigenmodes (TAE, RSAE, BAE) and for drift-wave and electromagnetic
micro-instabilities (ITG, KBM).

The numerics use field-aligned coordinates with a shifted metric, fourth-order finite differences on five points,
fourth-order Runge–Kutta time stepping, and a sparse direct solver for the Poisson equation. The grid is split along $y$
(the poloidal, field-aligned direction) across MPI processes and GPUs, which communicate through NCCL.

:::{admonition} About this documentation
:class: note

The physics and numerics chapters were written from the source code of this branch (`src/`), cross-checked against the
published papers; where the two differ, the code is described and the difference is noted. The ITPA example uses a run
made with this branch. cuMEGA, a cylindrical-coordinate code built on cuGMEC's structure with MEGA's MHD model, has
its own documentation and is compared with cuGMEC on the ITPA case.
:::

```{toctree}
:caption: Physics model
:maxdepth: 2

physics/coordinates
physics/fluid
physics/particles
```

```{toctree}
:caption: Numerical methods
:maxdepth: 2

numerics/discretization
numerics/solvers
numerics/time_integration
numerics/parallel
```

```{toctree}
:caption: User guide
:maxdepth: 2

guide/workflow
en-US/environment
en-US/parameters
guide/python
guide/diagnostics
```

```{toctree}
:caption: Examples and performance
:maxdepth: 2

examples/itpa
performance
```

```{toctree}
:caption: Back matter
:maxdepth: 1

references
```

```{toctree}
:caption: 中文文档
:maxdepth: 1

zh-CN/environment
zh-CN/parameters
```
