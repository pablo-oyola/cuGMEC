# Fluid model

The bulk plasma is described by a reduced, two-fluid gyrokinetic-MHD model {cite:p}`jiang2024gmec1,liu2024gmec2`. The
evolved fields are

| Field | Code name | Meaning |
|---|---|---|
| $\varpi$ | `w` | vorticity (polarization charge density) |
| $A$ | `A` | $\delta A_\parallel$ |
| $\delta n_e$ | `dNe` | electron density perturbation |
| $\delta T_e$ | `dTe` | electron temperature perturbation |
| $\delta p$ | `dP` | single-fluid pressure (only with `ifReducedMHD`) |

and after every Runge–Kutta stage the code derives the potential $\Phi$ (`Phi`), the parallel current
$\tilde J = \delta J_\parallel/B$ (`dJpB`) and the electron pressure $\delta p_e$ (`dPe`). The kinetic species provide
their perturbed pressures $\delta p_s$ (`dPi`, `dPa`, `dPb`) and, with quasi-neutrality, their gyrocentre densities.

The equations below are in SI form ($\mu_0$ and $e$ explicit); see [](coordinates.md) for the normalized form. They
were reconstructed from the coefficient arrays of `src/cuGMEC_model.h` (`compressCoefficient`), which are generated
symbolically and contracted with finite differences in `src/kernels/mhd.cuh`.

## Operators

With $\mathbf b = \mathbf B/B$, the perturbed field line and the drifts are

$$
\nabla_\parallel = \mathbf b\cdot\nabla + B^{-1}\delta\mathbf B\cdot\nabla,\qquad
\delta\mathbf B = \nabla\times(A\mathbf b) = \nabla A\times\mathbf b + A\nabla\times\mathbf b,\qquad
\mathbf v_E = \frac{\mathbf B\times\nabla\Phi}{B^2},
$$

and three geometric operators appear:

$$
\begin{aligned}
&\nabla_\perp\cdot(c\nabla_\perp f)\quad(\partial_y\ \text{of perturbations dropped}),\\
&\mathcal K(h) = \frac1B\,(\mathbf b\times\boldsymbol\kappa)\cdot\nabla h,\\
&\mathcal C(h) = \nabla\times\frac{\mathbf B}{B^2}\cdot\nabla h = \nabla\cdot\frac{\mathbf B\times\nabla h}{B^2}.
\end{aligned}
$$

In the perpendicular operators, $\partial_y$ acting on a perturbation is dropped ($k_\parallel \ll k_\perp$), while
$\partial_y$ of the equilibrium coefficients, of $J$ and of $g^{iy}$ is kept. $\mathcal K$ uses the exact curvature from
the metric, and $\mathcal C$ is the exact compressibility of the $E\times B$ and diamagnetic flows.

## Gyrokinetic vorticity equation

$$
\frac{\partial\varpi}{\partial t} =
  \mathbf B\cdot\nabla\frac{\delta J_\parallel}{B}
  + \delta\mathbf B\cdot\nabla\frac{J_{\parallel0}}{B}
  + \frac{2}{B}\,(\mathbf b\times\boldsymbol\kappa)\cdot\nabla\Big(\delta p_e + \sum_s\delta p_s\Big)
  - \mathbf v_{*i}\cdot\nabla\varpi
  + \Big[\delta\mathbf B\cdot\nabla\frac{\delta J_\parallel}{B} - \mathbf v_E\cdot\nabla\varpi\Big]_\mathrm{NL}
$$ (eq-vorticity)

| Term | Physics | Switch |
|---|---|---|
| $\mathbf B\cdot\nabla(\delta J_\parallel/B)$ | field-line bending | always |
| $\delta\mathbf B\cdot\nabla(J_{\parallel0}/B)$ | kink drive by the equilibrium current | always |
| $(2/B)\,\mathbf b\times\boldsymbol\kappa\cdot\nabla\delta p$ | pressure-curvature (ballooning) drive, fluid electrons and every enabled kinetic species $s$ | `if<Species>` |
| $\mathbf v_{*i}\cdot\nabla\varpi$ | ion diamagnetic drift, $\mathbf v_{*i} = \mathbf B\times\nabla p_{i0}/(e n_{i0}B^2)$ | `ifDiaDrift` |
| $\delta\mathbf B\cdot\nabla(\delta J_\parallel/B)$ | Maxwell stress | `ifNonlinearMHD`, `ifMaxwellStress` (× `MaxwellStressCoef`) |
| $\mathbf v_E\cdot\nabla\varpi$ | Reynolds stress ($E\times B$ advection of the vorticity) | `ifNonlinearMHD`, `ifReynoldsStress` (× `ReynoldsStressCoef`) |

Remarks:

- The kinetic species enter the fields only here, through their pressure in the curvature term (and through
  quasi-neutrality if enabled). All species share the coefficients of the fluid electron pressure.
- The ion diamagnetic term is the simplified gyroviscous cancellation, $-i\omega_{*p,i}\delta\varpi$ in Fourier form,
  with the total equilibrium ion pressure gradient. It applies whether or not the thermal ions are also kinetic.
- The Reynolds stress is implemented as $E\times B$ advection of $\varpi$, an approximation of the polarization
  nonlinearity $\nabla\cdot[(\rho_m/B^2)(\mathbf v_E\cdot\nabla)\nabla_\perp\Phi]$.
- There is no equilibrium flow, equilibrium potential or radial electric field.

## Gyrokinetic Poisson equation

$$
\varpi = \nabla_\perp\cdot\left(\frac{m_i n_i}{B^2}\nabla_\perp\Phi_0\right)
  = \nabla_\perp\cdot\left(\frac{1}{\mu_0 V_A^2}\nabla_\perp\Phi_0\right)
$$ (eq-poisson)

with the local Alfvén speed $V_A(\rho, \theta)$ (block `Va`). $\Phi_0$ is found by a sparse direct solve on each $y$
plane ([](../numerics/solvers.md)).

**Thermal-ion FLR** (`ifFLRMHD`). The potential is then corrected as

$$
\Phi = \Phi_0 - \rho_i^2 V_A^2\,\mu_0\varpi \approx (1 - \rho_i^2\nabla_\perp^2)\,\Phi_0,\qquad \rho_i = \frac{\sqrt{m_i T_i}}{eB},
$$

the Padé form of the polarization density, $(1 - \Gamma_0(b))/b \approx 1/(1 + b)$ with $b = k_\perp^2\rho_i^2$.

**Quasi-neutrality** (`ifQNeutrality`). Instead of evolving it, the electron density is diagnosed after every stage
from

$$
e\,\delta n_e = \varpi + \sum_s Z_s e\,\delta\bar n_s ,
$$

with $\delta\bar n_s$ the deposited gyrocentre densities of the kinetic species.

## Parallel Ampère's law

$A$ is evolved, and the current follows from it explicitly (no solve):

$$
\mu_0\,\delta J_\parallel = -\frac1B\,\nabla_\perp\cdot\left[B^2\nabla_\perp\left(\frac{A}{B}\right)\right].
$$ (eq-ampere)

## Parallel Ohm's law

$$
\frac{\partial A}{\partial t} = -\nabla_\parallel\Phi + \frac{T_e}{e\,n_e}\nabla_\parallel n_e,
\qquad\text{i.e.}\qquad E_\parallel = -\frac{\nabla_\parallel p_e}{e\,n_e}\Big|_{\nabla_\parallel T_e = 0}.
$$ (eq-ohm)

- The pressure term is switched by `ifEparallel`; without it the model is ideal, $E_\parallel = 0$.
- Linearized, $\nabla_\parallel n_e \to \mathbf b\cdot\nabla\delta n_e + \delta\mathbf B\cdot\nabla n_{e0}/B$ and the
  prefactor is $T_{e0}/(e n_{e0})$. With `ifNonlinearMHD` the full $\nabla_\parallel$, $T_e = T_{e0} + \delta T_e$ and
  $n_e = n_{e0} + \delta n_e$ are used.
- There is no electron inertia, no $\nabla_\parallel\delta T_e$ term and no explicit resistivity. A parallel
  resistivity can be applied by the implicit step `ifNablaPerp2A` ([](../numerics/solvers.md)).

The same right-hand side gives $\partial_t A_\parallel$, which the particles need for the inductive electric field.

## Electron continuity equation

$$
\begin{aligned}
\frac{\partial\,\delta n_e}{\partial t} ={}& -\mathbf v_E\cdot\nabla n_{e0} - n_{e0}\,\mathcal C(\Phi)
  + \frac{1}{e}\,\mathcal C(\delta p_e)
  + \frac{1}{e}\left[\mathbf B\cdot\nabla\frac{\delta J_\parallel}{B} + \delta\mathbf B\cdot\nabla\frac{J_{\parallel0}}{B}\right]\\
  &+ \Big[-\mathbf v_E\cdot\nabla\delta n_e - \delta n_e\,\mathcal C(\Phi) + \frac1e\,\delta\mathbf B\cdot\nabla\frac{\delta J_\parallel}{B}\Big]_\mathrm{NL}
\end{aligned}
$$ (eq-continuity)

This is $\partial_t n_e = -\nabla\cdot(n_e\mathbf v_E) - \nabla\cdot(n_e\mathbf v_{*e}) + e^{-1}\nabla\cdot(J_\parallel\mathbf b_\mathrm{tot})$:
the whole parallel current is carried by the electrons. $\mathcal C(\Phi) = \nabla\cdot\mathbf v_E$ is exact, including
the $\nabla B$ and curvature compressions. The nonlinear $\delta\mathbf B\cdot\nabla\tilde J$ term is always included
in the nonlinear model; it does not follow `ifMaxwellStress`. With `ifQNeutrality`, the result is replaced by the
quasi-neutral $\delta n_e$.

## Electron temperature equation

The README calls it the electron isothermal condition. As implemented it advects $T_e$ with the perpendicular drifts of
the perturbed field:

$$
\frac{\partial T_e}{\partial t} = -(\mathbf v_E + \mathbf v_{*e})\cdot\nabla T_e,\qquad
\mathbf v_{*e} = -\frac{\mathbf b_t\times\nabla p_e}{e n_e B},\quad \mathbf b_t = \mathbf b + \delta\mathbf B/B ,
$$ (eq-te)

with $T_e = T_{e0} + \delta T_e$. Linearized, only $-\mathbf v_E\cdot\nabla T_{e0}$ and the diamagnetic cross terms
$\propto T_{e0}n_{e0}'\,\delta T_e$ and $T_{e0}T_{e0}'\,\delta n_e$ remain. There is no parallel conduction, heating
or compression. In the papers {cite:p}`jiang2024gmec1` the electrons are instead isothermal along the field,
$\nabla_\parallel T_e = 0$, with no $T_e$ equation.

The electron pressure is closed by

$$
\delta p_e = \delta n_e\,T_{e0} + n_{e0}\,\delta T_e\ \big[+\,\delta n_e\,\delta T_e\big]_\mathrm{NL}.
$$

## Reduced-MHD pressure (`ifReducedMHD`)

With `ifReducedMHD`, a single-fluid pressure replaces $\delta p_e$ in the vorticity equation:

$$
\frac{\partial p}{\partial t} + \mathbf v_E\cdot\nabla p + \Gamma\,p\,\frac{2}{B}\,(\mathbf b\times\boldsymbol\kappa)\cdot\nabla\Phi = 0,
\qquad p_0 = p_{i0} + p_{e0},
$$ (eq-reduced)

i.e. $\Gamma p\nabla\cdot\mathbf v_E$ in its curvature form, with $\Gamma$ = `Gamma` (the papers use 5/3; the example
file has 0). $\delta n_e$ and $\delta T_e$ are still advanced but feed back only through Ohm's law, so
`ifEparallel = falseType` is recommended with this option.

## Dissipation

Optional dissipation is applied after every completed step, by operator splitting
([](../numerics/solvers.md), [](../numerics/discretization.md)):

- `perp2<Field>`: perpendicular diffusion $\partial_t f = \nu_\perp\nabla_\perp^2 f$, one backward-Euler step. For
  `Phi` it acts on the potential, and $\varpi$ is then recomputed from the smoothed $\Phi$ (the papers add
  $D\nabla_\perp^2\delta\varpi$ to the vorticity equation instead). For `A` it is a parallel resistivity,
  $\partial_t A = -\eta\,\delta J_\parallel$ with $\eta$ = `perp2A`.
- `para4<Field>`: parallel hyper-diffusion, one explicit Euler step $f \leftarrow f - \Delta t\,\nu_4\,\partial_y^4 f$
  along the field line. It is a grid operator ($\partial_y^4$ at fixed field line, without metric factors), so
  $\nu_4$ is in units of $1/t_0$.
- `perp2dP`, `para4dP`: the same on the deposited kinetic pressures (and `perp2dNe` on the kinetic densities with
  `ifQNeutrality`).

## Boundary conditions

The radial edges $\rho_0$ and $\rho_1$ are perfectly conducting walls. The right-hand sides are evaluated only at the
interior nodes, so every fluid field keeps its initial value of zero on both walls:
$\Phi = \varpi = A = \delta n_e = \delta T_e = \delta p = \delta J_\parallel = 0$, and the deposited kinetic pressures
vanish there too. The papers {cite:p}`jiang2024gmec1,liu2024gmec2` state $\delta J_\parallel = 0$ and zero radial
derivatives of $\delta p$ and $\delta n_e$; the code uses zero values. The directions $y$ and $z$ are periodic (with the
twist-shift for the field-aligned representation).

## Model summary

| Physics | In the model | Not in the model |
|---|---|---|
| shear Alfvén waves, field-line bending, kink drive | yes | |
| pressure-curvature drive (fluid electrons, kinetic species) | yes | |
| ion diamagnetic drift (gyroviscous cancellation) | `ifDiaDrift` | |
| $E_\parallel$ from $\nabla_\parallel p_e$, kinetic Alfvén waves with $\rho_i$ FLR | `ifEparallel`, `ifFLRMHD` | electron inertia |
| thermal-ion kinetics (Landau damping, ITG) | kinetic `Ion` species | |
| perpendicular compression of the bulk pressure | `ifReducedMHD` ($\Gamma p\nabla\cdot\mathbf v_E$) | |
| parallel ion flow, sound waves, fluid $\delta p_i$ | | not evolved |
| resistivity | implicit `perp2A` | in Ohm's law |
| equilibrium flow, $E_r$ | | yes |
