# Gyrokinetic particles

Up to three kinetic ion species are simulated with the $\delta f$ particle-in-cell method {cite:p}`liu2024gmec2`:

| Species | Switch | Typical use | Note |
|---|---|---|---|
| `Ion` | `ifIon` | thermal ions (Landau damping, ITG, kinetic thermal-ion effects) | |
| `Alpha` | `ifAlpha` | fusion alphas, or any second energetic species | |
| `Beam` | `ifBeam` | beam ions | loaded with $v_\parallel \ge 0$ only |

Each species has a mass `<Species>Mass` ($m_p$), a charge `<Species>Char` ($e$), a distribution type and its
parameters, a $\beta$ and a speed range.

## Units and marker state

Lengths are in $L_0$, velocities in $v_{A0}$, masses in $m_p$, charges in $e$, energies in $m_p v_{A0}^2$ and $\mu$ in
$m_p v_{A0}^2/B_0$. The small parameter of the guiding-centre expansion is

$$
\varepsilon = \frac{v_{A0}}{L_0\,eB_0/m_p} = \frac{\rho_{A,p}}{L_0},
$$

the proton Larmor radius at $v_{A0}$ over $L_0$. A marker carries seven numbers:

| Slot | Content |
|---|---|
| $x$, $y = \theta$, $z$ | position; in a tokamak $z = \alpha = \varphi - q\theta$ (field-aligned), in a stellarator $z = \varphi$ |
| $v_\parallel$ | parallel velocity ($v_\parallel$ formulation) |
| $w$ | weight, $\delta f/g$ |
| $p$ | $f/g$ at load time, constant afterwards |
| $\mu$ | magnetic moment, an exact invariant |

where $g$ is the marker distribution in phase space.

## Fields seen by the particles

Before each particle step the code prepares, from the current MHD state:

1. $\partial_t A_\parallel$ from the right-hand side of Ohm's law ([](fluid.md));
2. $\Phi$, $A$ and $\partial_t A$ converted to the field-aligned representation and all-gathered over $y$, so every GPU
   holds the whole domain;
3. their fourth-order derivatives in $(x, y, z)$, packed per node as
   $[A, \partial_x A, \partial_y A, \partial_z A, \partial_x\Phi, \partial_y\Phi, \partial_z\Phi, \partial_t A]$.

Fields are interpolated to the markers with trilinear (cloud-in-cell) weights, the equilibrium with bilinear weights in
$(x, y)$ and the profiles linearly in $x$. The fields are frozen for the whole particle step.

## Guiding-centre equations

With the modified field

$$
\mathbf B^* = B\mathbf b + \varepsilon\frac{m}{Z}v_\parallel\nabla\times\mathbf b + \delta\mathbf B,\qquad
B^*_\parallel = \mathbf b\cdot\mathbf B^*,\qquad \delta\mathbf B = A\nabla\times\mathbf b + \nabla A\times\mathbf b,
$$

the markers follow

$$
\frac{d\mathbf X}{dt} = \frac{1}{B^*_\parallel}\left[v_\parallel\mathbf B^* - \nabla\Phi\times\mathbf b
  - \frac{\varepsilon\mu}{Z}\nabla B\times\mathbf b\right],
$$ (eq-gc-x)

$$
\frac{dv_\parallel}{dt} = -\frac{Z}{\varepsilon m}\left[\partial_t A_\parallel
  + \frac{\mathbf B^*\cdot\left(\nabla\Phi + \frac{\varepsilon\mu}{Z}\nabla B\right)}{B^*_\parallel}\right],
$$ (eq-gc-v)

i.e., in physical units, $m\,dv_\parallel/dt = (Ze/B^*_\parallel)\,\mathbf B^*\cdot(\mathbf E - (\mu/Ze)\nabla B)$ with
$E_\parallel = -\mathbf b\cdot\nabla\Phi - \partial_t A_\parallel$. The geometry ($\mathbf b$, $\nabla\times\mathbf b$,
$\nabla B$) comes from the field-aligned metric; the mirror force and the $\nabla B$ drift use the equilibrium $|B|$
only, and $\delta\mathbf B$ enters $\mathbf B^*$ only in the nonlinear model.

This is the $v_\parallel$ (symplectic) formulation with $A_\parallel$ evolved by the fluid Ohm's law, so there is no
$p_\parallel$ pull-back and no cancellation problem.

- **Nonlinear particles** (`ifNonlinearPIC`): markers follow the perturbed orbits above. The perturbed velocities
  $\delta\dot{\mathbf X} = \dot{\mathbf X} - \dot{\mathbf X}_0$ and $\delta\dot v_\parallel = \dot v_\parallel - \dot v_{\parallel0}$
  are formed against the field-free equilibrium motion.
- **Linear particles**: markers follow the unperturbed orbits, and the weights are driven by the linearized
  velocities $\delta\dot{\mathbf X} = (v_\parallel\delta\mathbf B - \nabla\Phi\times\mathbf b)/B^*_{\parallel0}$ and
  $\delta\dot v_\parallel = -(Z/\varepsilon m)[\partial_t A + (\mathbf B^*_0\cdot\nabla\Phi + (\varepsilon\mu/Z)\,\delta\mathbf B\cdot\nabla B)/B^*_{\parallel0}]$.

In a stellarator the same equations are solved in straight $(x, \theta, \varphi)$ PEST coordinates, with the field
derivatives converted from the aligned grid by $\partial_x|_S = \partial_x|_A - q'\theta\,\partial_z|_A$,
$\partial_\theta|_S = \partial_y|_A - q\,\partial_z|_A$, $\partial_\varphi|_S = \partial_z|_A$.

## $\delta f$ weight equation

With $\mathcal E = \tfrac12 m v_\parallel^2 + \mu B$,

$$
\frac{dw}{dt} = -(p - w)\left[\delta\dot x\,\frac{\partial\ln f_0}{\partial x}\Big|_{\mathcal E,\mu}
  + \delta\dot{\mathcal E}\,\frac{\partial\ln f_0}{\partial\mathcal E}\Big|_{x,\mu}\right],\qquad
\delta\dot{\mathcal E} = \mu\,\delta\dot{\mathbf X}\cdot\nabla B + m v_\parallel\,\delta\dot v_\parallel .
$$ (eq-weight)

In the linear model the factor $(p - w)$ becomes $p$. Since $p = f/g$ is conserved along exact orbits, $p - w = f_0/g$:
this is $dw/dt = -(f/g - w)\,d\ln f_0/dt$ restricted to the perturbed motion {cite:p}`liu2024gmec2`. $f_0$ is a function
of $(x, \mathcal E, \mu)$, not of the canonical momentum $P_\varphi$, and the equilibrium radial drift term
$\dot{\mathbf X}_0\cdot\nabla x\,\partial_x f_0$ is not kept.

## Equilibrium distributions

`<Species>Type` selects $f_0$. Its radial profiles $n_s(x)$ and $T_s(x)$ come from the equilibrium file; for
slowing-down species the "T" slot holds the critical speed $v_c/v_{A0}$.

| Type | $f_0$ (up to a constant) |
|---|---|
| `Maxwell` | $n\,T^{-3/2}\exp(-\mathcal E/T)$ |
| `Slowing0` | $n/(v^3 + v_c^3)$ |
| `Slowing1` | `Slowing0` $\times\,[1 + \operatorname{erf}((V_b - v)/\Delta V)]$ |
| `Slowing2` | `Slowing0` $\times\,\exp[-(\Lambda - \Lambda_0)^2/\Delta\Lambda^2]$, $\Lambda = \mu B_0/\mathcal E$ |
| `Slowing3` | `Slowing0` $\times$ both factors |

Parameters: `<Species>Vb` ($V_b$), `DeltaV` ($\Delta V$), `Lambda0` ($\Lambda_0$), `DeltaLambda2` ($\Delta\Lambda^2$),
`Vmin`, `Vmax`. The derivatives $\partial_x\ln f_0$ and $\partial_{\mathcal E}\ln f_0$ are analytic, including the
radial variation of $n$, $T$ and $v_c$. The constant normalization of $f_0$ is not used: the amplitude is set by the
pressure calibration below. The anisotropic `Slowing2` and `Slowing3` are cuGMEC additions to the published model.

## Marker loading

- **Position.** Rejection sampling in $(x, y)$ with acceptance $\propto n_s J$ (`spaceReal`, markers follow the
  density) or $\propto J$ (`spaceUniform`, uniform in volume).
- **Toroidal replication (tokamak).** $z$ is drawn in one cell, and each marker is copied to all `gridNz` cells with
  the same $(x, y, v_\parallel, \mu, p)$. Only one marker in `gridNz` is an independent sample, and the copies are
  correlated around the torus. In a stellarator $z$ is drawn over the whole wedge.
- **Velocity, `velocityReal`.** Markers approximate $f_0$: Maxwellian $v_\parallel$ and $v_\perp$ restricted to
  $[V_\mathrm{min}, V_\mathrm{max}]$; slowing-down speeds by inverse-CDF sampling of $v^2/(v^3 + v_c^3)$ with an isotropic
  pitch, plus rejection on the erf and $\Lambda$ factors. Then $p = 1$ for `spaceReal` (or $n_s$ for `spaceUniform`).
- **Velocity, `velocityUniform`.** Markers are uniform in $d^3v$ on $[V_\mathrm{min}, V_\mathrm{max}]$ with an isotropic
  pitch, and $p$ is the unnormalized $f_0$.
- $\mu = m v^2(1 - \xi^2)/(2B)$; initial weights are zero.

The random generator is seeded from the system's entropy source, so two runs are not bit-for-bit identical.

:::{note}
With `velocityReal`, $p$ equals $f_0/g$ only up to the normalization of the truncated or rejected velocity
distribution. That normalization depends on $x$ through $v_c$, $T$ and $B$, and its radial derivative is not in the
weight equation.
:::

**Pressure calibration.** After loading, the equilibrium pressure
$\sum p\,(m v_\parallel^2 + \mu B)/(2J)$ is deposited, and a constant `partConst` is set so that its average over the
$x = 0$ nodes equals `<Species>Beta`. The deposited density and pressure profiles are written to
`initial/<Species>Density` and `initial/<Species>Pressure`.

## Gyro-averaging (`ifFLRPIC`)

With `ifFLRPIC`, every field quantity in the equations of motion is averaged over `gyroNums` points on the gyro-ring
(phases $2\pi k/$`gyroNums`, no random offset), with Larmor radius $\rho_L = \sqrt{2m\mu B}/(ZeB)$ at the guiding
centre, and the deposit is spread over the same ring points.

In a tokamak the ring is built in the local flux-surface frame: a displacement $\rho_L\cos\alpha$ along $\nabla x$ and
$\rho_L\sin\alpha$ in the surface perpendicular to $\mathbf B$, using the precomputed factors $1/|\nabla x|$ and the
binormal length; the displaced point is mapped back to $(x, \theta, \alpha)$ with the twist-shift. This first-order
construction differs from the ring of {cite:t}`liu2024gmec2`. Ring points outside $x\in[0, 1)$ contribute zero, but
the average still divides by `gyroNums`.

## Deposition and coupling

Each marker deposits, at the end of its push and in the same kernel,

$$
\delta p_s \leftarrow \frac{\mathrm{partConst}}{2}\,\frac{(m v_\parallel^2 + \mu B)\,w}{J},
$$

the scalar pressure $(\delta P_\parallel + \delta P_\perp)/2$, with trilinear weights on the aligned grid. The
deposited pressure is then summed over all GPUs, its ghost rows folded back with the twist-shift, converted to the
shifted representation, and optionally smoothed and filtered. It enters the vorticity equation through the curvature
term,

$$
\frac{2}{B}\,(\mathbf b\times\boldsymbol\kappa)\cdot\nabla\frac{\delta P_\parallel + \delta P_\perp}{2},
$$

the usual scalar-pressure coupling, which merges the $\nabla B$ drift of $\delta P_\perp$ into the curvature drift of
$\delta P_\parallel$ ($\nabla_\perp\ln B\approx\boldsymbol\kappa$). No kinetic current is deposited. With
`ifQNeutrality`, the gyrocentre density $\delta\bar n_s$ is deposited as well. The nodes $x = 0$ and $x = 1$ receive
nothing, so $\delta p_s = 0$ on the walls.

The atomic additions go into 16 copies of the deposit buffer, chosen by marker index, to reduce contention; they are in
`mhdReal` (double) even when the particles are in single precision.

## Time step and boundaries

- Markers are advanced by classical RK4 on $(x, y, z, v_\parallel, w)$ with $\Delta t_\mathrm{PIC}$ = `ratioDt`·`dt`,
  one particle step every `ratioDt` MHD steps ([](../numerics/time_integration.md)).
- **Radial boundary.** Markers are never removed. A marker that would leave $x\in[0, 1)$ is put back at its
  start-of-step position with $\theta\to-\theta$ (its up-down mirror) and $w = 0$. {cite:t}`liu2024gmec2` describes
  such markers as discarded.
- **Poloidal.** In a tokamak, $(y, z)\to(y\mp2\pi, z\pm2\pi q)$ (twist-shift); in a stellarator, plain periodicity.
- **Toroidal.** Periodic over the wedge.
- Markers are sorted by cell every `sortSteps` particle steps (CUB radix sort) for memory locality.
